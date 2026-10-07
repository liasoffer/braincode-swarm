"""Helpers for spawn_batch.py, kept out of the dispatcher to keep it lean:
env loading, output-folder naming (content hash + LLM-generated slug), and the
startup bookkeeping that makes reruns skip already-done records instead of
redoing them. Needs `litellm` installed (`pip install litellm`) — the one
dependency this repo has beyond the standard library, isolated here rather
than in spawn_batch.py itself.
"""
import hashlib
import itertools
import json
import os
import re
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import litellm

SLUG_PROMPT = """Read the trajectory below and produce a short filesystem-safe \
slug describing it: lowercase words separated by single hyphens, 3 to 6 words, \
no punctuation, no quotes. It is one of many similar trajectories in the same \
batch, so name what's actually distinct about this one — the specific subject, \
action, or detail — not a generic description that could apply to any of them. \
Use Latin letters a-z and digits only, even when the trajectory itself is in \
another script or language: describe it in English rather than transliterating, \
since the slug is only there to be recognizable in a directory listing. \
Output only the slug, nothing else.

Trajectory:
{content}"""


def load_dotenv(path: Path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Config:
    harness_name: str
    task_name: str
    model: str
    concurrency: int
    stagger_s: float
    timeout_s: int
    experiment: str
    base_url: str
    api_key: str
    harness_dir: Path
    task_file: Path


def load_config(self_dir: Path) -> Config:
    """Read spawn_batch.py's configuration from the environment (self_dir/.env
    fills in anything not already set) and resolve/validate the pieces that
    need to name a real file. Raises ValueError with a user-facing message if
    anything required is missing — the caller decides how to report it.
    """
    load_dotenv(self_dir / ".env")

    harness_name = os.environ.get("SWARM_HARNESS", "pi")
    task_name = os.environ.get("SWARM_TASK", "discovery")
    model = os.environ.get("SWARM_MODEL", "vertex-proxy/gemini-3.5-flash")
    concurrency = int(os.environ.get("SWARM_CONCURRENCY", "16"))
    # Off by default. This was added on the theory that the pool's jitter-free
    # retries arrive at a restarting proxy as one burst and knock it over again;
    # that was not borne out — the proxy usually survives the returning traffic,
    # so the stagger only adds latency. Kept as a knob, not a default.
    stagger_s = float(os.environ.get("SWARM_STAGGER", "0"))
    # A ceiling on one record, not a target. Normal records finish in 2-5
    # minutes; one riding out proxy outages through the harness's own retries
    # can legitimately take ~15, so this is set well above that and exists
    # only to stop a container that will never finish — an agent burning a
    # core on `grep -rn / ` held its pool slot for 37 minutes before anyone
    # noticed.
    timeout_s = int(os.environ.get("SWARM_TIMEOUT", "1200"))
    experiment = os.environ.get("SWARM_EXPERIMENT") or None
    base_url = os.environ.setdefault("PROXY_BASE_URL", "https://vertex-proxy-v26q.onrender.com/v1")
    api_key = os.environ.get("PROXY_API_KEY")
    if not api_key:
        raise ValueError("PROXY_API_KEY not set (checked environment and .env)")

    harness_dir = self_dir / "harnesses" / harness_name
    if not (harness_dir / "Dockerfile").exists():
        raise ValueError(f"no such harness: {harness_dir}")
    task_file = self_dir / "tasks" / f"{task_name}.md"
    if not task_file.exists():
        raise ValueError(f"no such task file: {task_file}")

    return Config(harness_name, task_name, model, concurrency, stagger_s,
                  timeout_s, experiment, base_url, api_key, harness_dir,
                  task_file)


def load_batch(batch_path: Path, done_hashes: set) -> tuple:
    """Read a batch file into (line, hash) pairs, skipping blank lines and any
    record whose content hash is already in done_hashes. Returns
    (records, skipped_count).
    """
    records = []
    skipped = 0
    with batch_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            full_hash = content_hash(line)
            if full_hash in done_hashes:
                skipped += 1
                continue
            records.append((line, full_hash))
    return records, skipped


def prune_incomplete_folders(out_dir: Path) -> list:
    """Remove every folder without a metadata.json — both crash orphans and
    folders holding a recorded harness failure (stdout.log/stderr.log).

    Failure logs are kept only until the next run starts, not indefinitely:
    a retry can't reuse the failed folder's name, because the slug half is
    LLM-generated per attempt and comes out different for the same record
    (`352a9d-uncensored-russian-llm-huggingface-recommendation` and
    `352a9d-uncensored-russian-llms-huggingface` are one record, twice), so
    keeping them meant every re-run of a failing batch left another
    near-duplicate behind and the count grew without bound. Read the logs
    before re-running; a successful record's folder is never touched.
    Returns the names removed, for the caller to log.
    """
    removed = []
    if not out_dir.exists():
        return removed
    for entry in out_dir.iterdir():
        if not entry.is_dir():
            continue
        if (entry / "metadata.json").exists():
            continue
        shutil.rmtree(entry)
        removed.append(entry.name)
    return removed


def failures_dir(out_dir: Path) -> Path:
    """Where a failed record's logs are kept: `failures/<namespace>/`, mirroring
    out_dir's own `output/<namespace>/` shape one level up.

    Deliberately outside out_dir so the two never interfere — out_dir ends up
    holding successes only, and nothing in here is subject to
    prune_incomplete_folders. Assumes out_dir is a namespace subfolder of a
    top-level output dir (`<root>/output/<namespace>`) — spawn_batch.py always
    constructs it that way, so `out_dir.name` is the namespace and
    `out_dir.parent.parent` is `<root>`.
    """
    return out_dir.parent.parent / "failures" / out_dir.name


def claim_failure_dir(out_dir: Path, full_hash: str) -> Path:
    """Create and return a fresh failure directory: `<hash6>`, else `<hash6>-2`,
    `-3`, … taking the first free name.

    The suffix is assigned blindly, without checking whose record already holds
    the shorter name, so a suffixed name means only "that name was taken". Two
    unrelated things cause that, and neither is safe to assume from the name:

    - **The same record failing again** — the common case, since a re-run
      retries exactly the records that failed. Each attempt gets its own
      directory, so `failures/` is a history of attempts rather than a snapshot
      of the latest one.
    - **A different record whose hash shares the first 6 hex chars** — rare per
      pair but near-certain across a corpus this size (24 bits collides on the
      order of a hundred times at ~70k records).

    So don't read `<hash6>` and `<hash6>-2` as the same record twice, and don't
    read them as two different records either. Each directory's failure.json
    records the full `hash` it belongs to; that field, not the name, is what
    groups attempts by record — count the directories sharing a full hash to
    see how many times one record has failed.

    mkdir(exist_ok=False) is the claim, which makes it atomic: concurrent
    workers racing for the same name can't both win, so no lock is needed.
    """
    base_dir = failures_dir(out_dir)
    base = full_hash[:6]
    for i in itertools.count(1):
        candidate = base_dir / (base if i == 1 else f"{base}-{i}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue


def record_failure(out_dir: Path, full_hash: str, attempt_name: str,
                   record_line: str, stdout: str, stderr: str,
                   returncode: int, produced: bool, duration_s: float,
                   harness_name: str, task_name: str, model: str,
                   experiment: str, produced_dir=None) -> Path:
    """Preserve a failed attempt under `failures/`, and return its path.

    Kept outside out_dir, so out_dir holds successes only and none of this is
    exposed to prune_incomplete_folders — a failure's logs survive the next run
    without anything having to be rescued by hand first. One directory per
    *attempt* (see claim_failure_dir for how the name is chosen).

    `failure.json` records what the logs cannot: the container's exit code and
    whether it wrote anything. A run that exits 0 having produced no files is a
    different problem from one that was killed, and without the exit code the
    two are indistinguishable after the fact. `error` is null exactly when the
    harness reported no error at all, which is its own diagnostic signature
    rather than missing data.
    """
    dest = claim_failure_dir(out_dir, full_hash)
    (dest / "source.json").write_text(record_line)
    (dest / "stdout.log").write_text(stdout)
    (dest / "stderr.log").write_text(stderr)

    # Whatever the harness managed to write before it died, under partial/.
    # A run that gets three of four files out and then hits a 429 has done
    # almost all the work; discarding it left nothing to inspect and no way to
    # tell a near-miss from a container that never started. It stays a failure
    # either way — the task's contract is all four files, so no metadata.json
    # is written and a rerun still retries the record from scratch.
    partial_names = []
    if produced_dir is not None and Path(produced_dir).is_dir():
        partial = dest / "partial"
        for item in sorted(Path(produced_dir).iterdir()):
            partial.mkdir(exist_ok=True)
            target = partial / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
            partial_names.append(item.name)

    (dest / "failure.json").write_text(json.dumps(
        {
            "hash": full_hash,
            "attempt_name": attempt_name,
            "returncode": returncode,
            "produced_files": produced,
            "partial_files": partial_names,
            "duration_s": round(duration_s, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "harness": harness_name,
            "task": task_name,
            "model": model,
            "experiment": experiment,
            "error": last_error_line(stderr),
        },
        indent=2,
    ))
    return dest


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


HTTP_STATUS_RE = re.compile(r"^(\d{3})\b")


def last_error_line(stderr: str) -> str | None:
    """A short description of how the harness failed, or None if it said nothing.

    Deliberately not tied to one harness's wording. opencode prefixes
    `Error: ...`; pi writes a bare `503 status code (no body)` or dumps a whole
    Cloudflare challenge page starting `429 <!DOCTYPE html>`. Recognizing only
    the first shape reported pi's perfectly clear HTTP errors as "no error
    reported", which collides with the genuinely silent shape — a container
    that exits having produced neither output nor complaint — and those two
    need to stay distinguishable.

    None therefore means the harness really printed nothing usable, which is
    itself the diagnostic signal.
    """
    lines = [l.strip() for l in ANSI_ESCAPE.sub("", stderr).splitlines() if l.strip()]
    for line in reversed(lines):
        if line.startswith("Error:"):
            return line[len("Error:"):].strip() or None
    for line in lines:
        status = HTTP_STATUS_RE.match(line)
        if status:
            # Don't return the body: a Cloudflare challenge is ~20KB of HTML,
            # and naming the mitigation is the part worth recording.
            if "Just a moment" in stderr or "cf-mitigated" in stderr:
                return f"HTTP {status.group(1)} (Cloudflare challenge)"
            return f"HTTP {status.group(1)}"
    # Anything left that isn't the harness narrating itself. Both CLIs stream
    # progress to stderr (a banner, tool calls, todo lists), so a transcript
    # made of nothing but that is the silent shape — treating its banner as an
    # error message would erase the distinction this function exists to
    # preserve. Tested structurally rather than against a list of glyphs:
    # narration is prefixed with punctuation or symbols (> → ← ✱ ✗ # [ • $)
    # and that set differs per harness and version, while a message meant for a
    # human starts with a word.
    speech = [l for l in lines if l[0].isalnum()]
    return speech[0][:160] if speech else None


def scan_existing_output(out_dir: Path):
    """One pass over out_dir at startup: which content hashes are already
    done (have a metadata.json with a timestamp — written only on success),
    and every folder name already in use (for reserve_name's -n collision
    avoidance, regardless of whether that folder succeeded, failed, or is
    unrelated).
    """
    done_hashes = set()
    names = set()
    if not out_dir.exists():
        return done_hashes, names
    for entry in out_dir.iterdir():
        if not entry.is_dir():
            continue
        names.add(entry.name)
        meta_path = entry / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if meta.get("hash") and meta.get("timestamp"):
            done_hashes.add(meta["hash"])
    return done_hashes, names


def trajectory_text(record_line: str) -> str:
    """The record's `content` as readable text, for mounting into the container.

    A batch record is one JSON object per line, so handed over raw the
    trajectory arrives as a single line with its newlines escaped to `\\n` and
    its turns buried under `id`/`platform`/`timestamp` fields the task never
    asks about. Agents given that spend turns shelling out to
    python3/node/jq/perl to pretty-print it — none of which exist in the
    harness images — so they reach the trajectory several wasted model calls
    later, or not at all. Handing over the decoded content leaves nothing to
    parse. The raw record is still kept, as each output folder's source.json.

    Falls back to the raw line for anything that isn't a JSON object with a
    non-empty string `content`: an unexpected record shape should still reach
    the agent rather than fail the run outright.
    """
    try:
        record = json.loads(record_line)
    except (TypeError, ValueError):
        return record_line
    if not isinstance(record, dict):
        return record_line
    content = record.get("content")
    if not isinstance(content, str) or not content.strip():
        return record_line
    return content


def content_hash(record_line: str) -> str:
    """Full sha256 hex digest of the raw record line — the stable identity a
    folder name's prefix and its metadata.json's `hash` field both derive
    from, regardless of whether the record has its own `id` field.
    """
    return hashlib.sha256(record_line.encode()).hexdigest()


def slugify(text: str, max_words: int = 6, max_len: int = 50) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    text = "-".join(text.split("-")[:max_words])[:max_len].strip("-")
    return text or "example"


def generate_slug(record_line: str, model: str, api_key: str, base_url: str) -> str:
    """Ask the model for a short, specific slug for this one trajectory.
    Never raises — a naming nicety isn't worth failing a whole record over;
    falls back to a generic slug if the call fails for any reason.
    """
    # SWARM_MODEL is "provider/id", meaningful to the harnesses' own config
    # (e.g. opencode's provider block). litellm talks to base_url directly
    # here, so only the id half means anything to it — the "openai/" prefix
    # tells litellm to speak the OpenAI-compatible wire format against
    # whatever base_url points at, same protocol every harness already uses.
    model_id = model.rsplit("/", 1)[-1]
    try:
        response = litellm.completion(
            model=f"openai/{model_id}",
            api_key=api_key,
            api_base=base_url,
            messages=[{"role": "user", "content": SLUG_PROMPT.format(content=record_line)}],
            # Generous on purpose, not because the slug itself is long: a
            # reasoning model's thinking counts against this same budget, and
            # a too-small cap means it spends everything thinking and returns
            # no visible text at all (verified — 30 reliably produced empty
            # output; 1024 reliably left room for both).
            max_tokens=1024,
            timeout=45,
        )
        raw = response.choices[0].message.content or ""
    except Exception:
        raw = ""
    return slugify(raw)


def build_folder_name(full_hash: str, slug: str) -> str:
    return f"{full_hash[:6]}-{slug}"


def reserve_name(base: str, names: set, names_lock: threading.Lock) -> str:
    """Thread-safe: claims `base`, or `base-2`, `base-3`, ... if taken —
    covers both names already on disk at startup and ones claimed by a
    sibling worker earlier in this same run.
    """
    with names_lock:
        if base not in names:
            names.add(base)
            return base
        n = 2
        while f"{base}-{n}" in names:
            n += 1
        name = f"{base}-{n}"
        names.add(name)
        return name


SECRET_ENV_NAMES = ("PROXY_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "HF_TOKEN")


def scrub_secrets(root: Path) -> int:
    """Replace API key values from this process's env with REDACTED in every
    file under `root` (an agent that runs `env` would otherwise write them
    into its session log). Returns the number of files rewritten."""
    secrets = [v for v in (os.environ.get(n) for n in SECRET_ENV_NAMES) if v and len(v) >= 8]
    if not secrets or not root.exists():
        return 0
    changed = 0
    for path in (root.rglob("*") if root.is_dir() else [root]):
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        new = data
        for s in secrets:
            new = new.replace(s.encode(), b"REDACTED")
        if new != data:
            path.write_bytes(new)
            changed += 1
    return changed


def host_uid_gid():
    """(uid, gid) to run containers as, or None on hosts without POSIX ids
    (Windows/Docker Desktop, where bind mounts aren't uid-checked anyway)."""
    if hasattr(os, "getuid") and hasattr(os, "getgid"):
        return os.getuid(), os.getgid()
    return None


def build_docker_cmd(uid, gid, model: str, reference_dir: Path,
                      traj_path: Path, prompt_file: Path, scratch, image: str,
                      container_name: str | None = None,
                      extra_mounts=(), extra_env=None, add_host: bool = False,
                      memory: str = "1g", writable_mounts=(), passthrough_env=()) -> list:
    """The exact `docker run` invocation for one record: non-root (matches
    the host uid/gid, so the writable /output mount just works), all
    capabilities dropped, no privilege escalation, memory/CPU capped. Network
    is deliberately not restricted — every harness needs to reach the model
    API. The base mount surface is the 3 read-only paths + 1 writable dir
    below; callers add more read-only mounts through `extra_mounts`
    ((host_path, container_path) pairs — always mounted read-only, so /output
    stays the only writable path). See ADVANCED.md's Security section.

    reference_dir is a directory, not a file: the specification is split across
    reference/language-spec.md plus the glossary files, so an agent reads the
    spec and only the parts of the glossary it needs.

    `add_host` maps host.docker.internal to the host gateway, so a container
    can reach a service on the host — the glossary RAG server (rag/server.py).
    Docker Desktop provides that name already; on Linux it needs this flag.

    container_name is what makes a timeout enforceable: killing the `docker
    run` process only detaches the CLI, leaving the container running, so
    spawn_batch needs a name to `docker kill`.

    uid/gid may be None (Windows hosts): the --user flag is then omitted.

    `passthrough_env` names host variables (API keys) handed to the container
    by name only (`-e NAME`), so their values never appear on a command line.
    """
    env_flags = []
    for key, value in (extra_env or {}).items():
        env_flags += ["-e", f"{key}={value}"]
    mount_flags = []
    for host_path, container_path in extra_mounts:
        mount_flags += ["-v", f"{host_path}:{container_path}:ro"]
    # Besides /output, only state the harness itself must persist across
    # attempts (pi's session dir for resume) is ever mounted writable.
    for host_path, container_path in writable_mounts:
        mount_flags += ["-v", f"{host_path}:{container_path}"]
    return [
        "docker", "run", "--rm",
        *(("--name", container_name) if container_name else ()),
        *(("--user", f"{uid}:{gid}") if uid is not None and gid is not None else ()),
        "-e", "HOME=/tmp",
        "--cap-drop=ALL",
        "--security-opt", "no-new-privileges:true",
        "--memory", memory,
        "--cpus", "1",
        *(("--add-host", "host.docker.internal:host-gateway") if add_host else ()),
        "-e", "PROXY_API_KEY", "-e", "PROXY_BASE_URL",
        *[flag for name in passthrough_env for flag in ("-e", name)],
        "-e", f"SWARM_MODEL={model}",
        *env_flags,
        "-v", f"{reference_dir}:/reference:ro",
        "-v", f"{traj_path}:/trajectory.txt:ro",
        "-v", f"{prompt_file}:/prompt.md:ro",
        *mount_flags,
        "-v", f"{scratch}:/output",
        image,
    ]
