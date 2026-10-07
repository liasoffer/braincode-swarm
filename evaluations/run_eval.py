#!/usr/bin/env python3
"""Run the BrainCode evaluations: translators of six models on unseen test
items, with exactly the swarm translators' setup (compact spec, frozen
glossary, glossary RAG, kit, formats, context limits, host check).

    python run_eval.py setup         --run main                 # freeze reference, RAG, needs, contexts
    python run_eval.py translate     --run main [--models a,b] [--items k1,k2] [--runs 3] [--max-usd 150]
    python run_eval.py backtranslate --run main [--models ...]   # expressivity (Gemini models)
    python run_eval.py status        --run main

Everything of a run lives under runs/<run>/; every step is resumable (done
runs are skipped). API keys: Gemini through the vertex proxy (PROXY_API_KEY,
swarm/.env); Anthropic from ANTHROPIC_API_KEY or the user variable
CLAUDE_API_KEY; OpenAI from OPENAI_API_KEY or OPENAI_API_KEY_PERSONAL. Keys
reach containers by name only (`docker run -e NAME`), never as values.
"""
import argparse
import itertools
import json
import os
import random
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
SWARM = EVAL_DIR.parent / "swarm"
sys.path.insert(0, str(SWARM))
sys.path.insert(0, str(EVAL_DIR))

import loop  # noqa: E402
import loop_files as lf  # noqa: E402
import sample as sample_mod  # noqa: E402
import translate_batch as tb  # noqa: E402
import utils  # noqa: E402
from rag import needs as needs_mod  # noqa: E402

RUNS_DIR = EVAL_DIR / "runs"
MODELS_PATH = EVAL_DIR / "models.json"
RAG_PORT = int(os.environ.get("EVAL_RAG_PORT", "8775"))
THROTTLE_PORT = os.environ.setdefault("THROTTLE_PORT", "8786")
# Setup v2 (from the 216th Gemini run on): proxy concurrency 6 -> 4 (fewer
# throttle cooldowns), per-attempt timeout 1200 -> 2400 s (high-effort
# SWE-bench runs were killed mid-work), and the translator prompt asks for
# batched lookups and attaches examples.jsonl. Every result.json records the
# setup it ran under (`setup`).
SETUP_VERSION = "v2"
ROUTE_CONCURRENCY = {"proxy": int(os.environ.get("EVAL_PROXY_CONCURRENCY", "4")),
                     "anthropic": int(os.environ.get("EVAL_ANTHROPIC_CONCURRENCY", "3")),
                     "openai": int(os.environ.get("EVAL_OPENAI_CONCURRENCY", "3"))}
ROUTE_KEYS = {"anthropic": ("ANTHROPIC_API_KEY",), "openai": ("OPENAI_API_KEY",)}


def route_keys(model: dict) -> tuple:
    """Provider keys a container needs: only its own route's (the proxy key is
    always passed by build_docker_cmd)."""
    return ROUTE_KEYS.get(model["route"], ())
TIMEOUT_S = int(os.environ.get("EVAL_TIMEOUT", "2400"))
_cost_lock = threading.Lock()
# Evaluation containers are named <prefix><model>-...; a process cleans up only
# its own models' containers, so several evaluation processes (and the swarm
# loop, whose containers are swarm-tr-/swarm-mig-) can run side by side.
CONTAINER_PREFIX = "swarm-ev-"


def kill_model_containers(model_names) -> int:
    import subprocess
    try:
        names = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True,
                               timeout=60).stdout.split()
    except Exception:
        return 0
    mine = [n for n in names if any(n.startswith(f"{CONTAINER_PREFIX}{m}-") for m in model_names)]
    if mine:
        subprocess.run(["docker", "kill", *mine], capture_output=True, text=True, timeout=120)
    return len(mine)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------- config

def load_models() -> list:
    return json.loads(MODELS_PATH.read_text(encoding="utf-8"))["models"]


def _user_env(name: str):
    """A Windows user/machine environment variable not in this process's env."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        import winreg
        for hive, path in ((winreg.HKEY_CURRENT_USER, "Environment"),
                           (winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")):
            try:
                with winreg.OpenKey(hive, path) as key:
                    return winreg.QueryValueEx(key, name)[0]
            except OSError:
                continue
    except ImportError:
        pass
    return None


def prepare_keys() -> dict:
    """Put provider keys into this process's env under the names the harness
    expects; returns {route: available}. Values are never printed."""
    utils.load_dotenv(SWARM / ".env")
    anthropic = _user_env("ANTHROPIC_API_KEY") or _user_env("CLAUDE_API_KEY")
    openai = _user_env("OPENAI_API_KEY") or _user_env("OPENAI_API_KEY_PERSONAL")
    if anthropic:
        os.environ["ANTHROPIC_API_KEY"] = anthropic
    if openai:
        os.environ["OPENAI_API_KEY"] = openai
    return {"proxy": bool(os.environ.get("PROXY_API_KEY")), "anthropic": bool(anthropic), "openai": bool(openai)}


def run_dir(run: str) -> Path:
    return RUNS_DIR / run


def items_of(run: str) -> list:
    return [json.loads(line) for line in (run_dir(run) / "items.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]


def model_items(model: dict, items: list) -> list:
    return items if model["sample"] == "all" else [i for i in items if i["all_models"]]


def price_of(model: dict, usage: dict):
    p = model.get("price")
    if not p or not usage:
        return None
    write = p[3] if len(p) > 3 else p[0]   # cache writes; billed only where the provider reports them
    return round((usage.get("input", 0) * p[0] + usage.get("cacheRead", 0) * p[1] + usage.get("cacheWrite", 0) * write
                  + usage.get("output", 0) * p[2]) / 1e6, 4)


# ---------------------------------------------------------------------- services

class Services:
    """The frozen reference snapshot's RAG server (for the kit inside
    containers) and the model-proxy throttle, for the duration of a command."""

    def __init__(self, run: str, need_rag: bool = True, models=()):
        self.run = run
        self.need_rag = need_rag
        self.models = [m["name"] if isinstance(m, dict) else m for m in models]
        # the proxy throttle only when a proxy model runs (or for setup's need extraction)
        self.routes = {m["route"] for m in models if isinstance(m, dict)}
        self.retriever = self.server = self.throttle = None

    def __enter__(self):
        from rag.retrieve import Retriever
        from rag.server import RagServer
        d = run_dir(self.run)
        self.retriever = Retriever(dense=True, glossary_path=d / "reference" / "glossary.jsonl",
                                   index_dir=d / "rag_index")
        if self.need_rag:
            self.server = RagServer(self.retriever, "0.0.0.0", RAG_PORT).start()
            log(f"eval: RAG server on :{RAG_PORT} ({len(self.retriever.index.records)} records, frozen snapshot)")
        if not self.routes or "proxy" in self.routes:
            cfg = loop.load_loop_config()
            self.throttle = loop.start_throttle(cfg)
        return self

    def __exit__(self, *exc):
        if self.throttle is not None:
            self.throttle.stop()
        if self.server is not None:
            self.server.stop()
        kill_model_containers(self.models)


# ---------------------------------------------------------------------- setup

def cmd_setup(args):
    d = run_dir(args.run)
    items = sample_mod.load()
    if args.items:
        wanted = set(args.items.split(","))
        items = [i for i in items if i["item_key"] in wanted]
    d.mkdir(parents=True, exist_ok=True)
    ref = d / "reference"
    if not ref.exists():
        tb.snapshot_reference(ref)
        from glossary import manifest
        (d / "meta.json").write_text(json.dumps({
            "glossary_version": manifest.current_version(), "created_at": lf.now_iso(),
            "sample": sample_mod.SAMPLE_PATH.name, "items": len(items)}, indent=1), encoding="utf-8")
        log(f"eval: froze reference ({manifest.current_version()}) -> {ref}")
    (d / "items.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    with Services(args.run, need_rag=False) as svc:
        def one(item):
            idir = d / "items" / item["item_key"]
            if (idir / "rag_context.md").exists():
                return "cached"
            idir.mkdir(parents=True, exist_ok=True)
            content = lf.item_content(item)
            segments, needs, method = needs_mod.extract_needs(content, use_llm=True)
            stripped = [{k: v for k, v in n.items() if k not in ("id", "context")} for n in needs]
            result = svc.retriever.retrieve(content, needs=stripped, needs_method=method)
            needs_full = result["needs"]
            (idir / "trajectory.txt").write_text(needs_mod.numbered_text(result["segments"]), encoding="utf-8")
            (idir / "item_raw.txt").write_text(content, encoding="utf-8")
            (idir / "needs.json").write_text(json.dumps({"translator_id": item["item_key"], "needs": needs_full},
                                                        ensure_ascii=False, indent=1), encoding="utf-8")
            from rag.retrieve import render_context
            (idir / "rag_context.md").write_text(render_context(result, svc.retriever, meta["glossary_version"]),
                                                 encoding="utf-8")
            (idir / "meta.json").write_text(json.dumps({**{k: item[k] for k in ("item_key", "dataset", "item_id",
                                                                               "stratum", "chars", "all_models")},
                                                        "needs": len(needs_full), "needs_method": method,
                                                        "glossary_sha": result["glossary_sha"]}, indent=1),
                                            encoding="utf-8")
            return method
        with ThreadPoolExecutor(max_workers=4) as pool:
            done = list(pool.map(one, items))
    log(f"eval: setup {args.run}: {len(items)} items; needs by method {dict((m, done.count(m)) for m in set(done))}")


# ---------------------------------------------------------------------- translate

def _route_semaphores():
    return {route: threading.Semaphore(n) for route, n in ROUTE_CONCURRENCY.items()}


def translate_one(model: dict, item: dict, k: int, run: str, image: str, retriever, budget: dict, sems: dict) -> dict:
    d = run_dir(run)
    out_dir = d / model["name"] / item["item_key"] / f"r{k}"
    result_path = out_dir / "result.json"
    if result_path.exists():
        return {**json.loads(result_path.read_text(encoding="utf-8")), "already_done": True}
    with _cost_lock:
        if budget["spent"].get(model["name"], 0) >= budget["max_usd"]:
            return {"status": "skipped", "detail": "model budget reached"}
    idir = d / "items" / item["item_key"]
    ref = d / "reference"
    if out_dir.exists():
        shutil.rmtree(out_dir)   # an interrupted run starts over
    work = out_dir / "work"
    for sub in ("session", "out"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    for name in ("trajectory.txt", "item_raw.txt", "rag_context.md", "needs.json"):
        shutil.copy2(idir / name, work / name)
    tid = f"{item['item_key']}-r{k}"
    prompt = tb.fill_template((lf.TASKS_DIR / "translator.md").read_text(encoding="utf-8"),
                              {"TRANSLATOR_ID": tid, "BATCH_ID": "eval", "TNUM": k, "DATASET": item["dataset"]})
    attach = tb.build_attachments(work / "attach", ref, work)
    needs = json.loads((work / "needs.json").read_text(encoding="utf-8"))["needs"]
    usage, feedback, last_problem = {}, "", None
    started = time.monotonic()
    for attempt in (1, 2):
        resume = attempt > 1 and any((work / "session").rglob("*.jsonl"))
        prompt_path = work / f"prompt{attempt}.md"
        prompt_path.write_text((tb.CONTINUE_PROMPT.format(problem=last_problem or "interrupted") if resume else prompt)
                               + feedback, encoding="utf-8")
        container = f"{CONTAINER_PREFIX}{model['name']}-{tid}-{attempt}-{random.randint(1000, 9999)}"
        env = {"TRANSLATOR_ID": tid, "DATASET": item["dataset"], "BATCH_ID": "eval", "RAG_PORT": RAG_PORT,
               "PI_JSON": "1", "PI_SESSION_DIR": "/session", **tb.CONTEXT_LIMITS_ENV}
        if resume:
            env["PI_RESUME"] = "1"
        cmd = utils.build_docker_cmd(
            None, None, model["pi_model"], ref, work / "trajectory.txt", prompt_path, work / "out", image,
            container_name=container, add_host=True,
            extra_mounts=[(work / "item_raw.txt", "/item_raw.txt"), (work / "rag_context.md", "/rag_context.md"),
                          (work / "needs.json", "/needs.json"), (lf.KIT_DIR, "/kit"),
                          (lf.DOC_FORMATS_DIR, "/doc_formats"), (attach, "/attach")],
            writable_mounts=[(work / "session", "/session")], extra_env=env, passthrough_env=route_keys(model))
        with sems[model["route"]]:
            proc, duration, used = tb.run_container_logged(cmd, container, TIMEOUT_S, out_dir / f"attempt{attempt}.log")
        utils.scrub_secrets(out_dir)   # an agent running `env` would log the keys
        used = {**used, **tb.collect_context_log(work / "session", used, out_dir / f"attempt{attempt}.context.jsonl")}
        for key, value in used.items():
            usage[key] = usage.get(key, 0) + value
        status, body, sugg, problem = tb.validate_output(work / "out")
        tail = utils.last_error_line(proc.stderr or "") or ""
        if problem is None and proc.returncode != 0:
            problem = f"harness exited {proc.returncode}" + (f" ({tail})" if tail else "")
        report = None
        if problem is None:
            report = retriever.check(body, needs)
            if status == "success":
                gate = tb.success_gate_problems(report)
                if gate:
                    problem = "claimed success but the host check found: " + "; ".join(gate)
                    feedback = ("\n\n## Your previous attempt was rejected\n\nIt declared `Status: success`, but the "
                                "host's check of its BrainCode found problems. Fix every one, or report a failure "
                                "with suggestions:\n\n" + tb.render_check(report))
            elif status == "failed":
                gate = tb.failure_gate_problems(report)
                if gate:
                    problem = "failed translation rejected: " + "; ".join(gate)
                    feedback = tb.FAILURE_GATE_FEEDBACK + tb.render_check(report)
        if problem is None:
            (out_dir / "translation.md").write_text(body, encoding="utf-8")
            if status == "failed":
                (out_dir / "suggestions.md").write_text(sugg or "", encoding="utf-8")
            (out_dir / "check.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
            break
        last_problem = problem
        status = "error"
        if attempt == 1 and tb.is_proxy_error(tail):
            time.sleep(random.uniform(*tb.PROXY_BACKOFF_S))
    tb.keep_session(work / "session", out_dir / "session.jsonl")
    for sub in ("out",):
        if (work / sub).exists():
            shutil.copytree(work / sub, out_dir / "output", dirs_exist_ok=True)
    shutil.rmtree(work, ignore_errors=True)
    cost = price_of(model, usage)
    res = {"model": model["name"], "item": item["item_key"], "dataset": item["dataset"], "run": k,
           "status": status, "problem": last_problem if status == "error" else None, "attempts": attempt,
           "duration_s": round(time.monotonic() - started, 1), "usage": usage, "cost_usd": cost,
           "setup": SETUP_VERSION, "timeout_s": TIMEOUT_S}
    if status == "failed":
        res["failure"] = failure_summary(out_dir)
    result_path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    if cost:
        with _cost_lock:
            budget["spent"][model["name"]] = budget["spent"].get(model["name"], 0) + cost
    return res


def failure_summary(out_dir: Path) -> dict:
    """What a failed translation says the glossary lacks: the terms its
    suggestions would add (symbol, dimension) and refine (target), with counts,
    from suggestions.md (the translator's own failure documentation)."""
    sugg = out_dir / "suggestions.md"
    blocks = lf.parse_suggestions(sugg.read_text(encoding="utf-8")) if sugg.exists() else []
    adds = [{"term": b["value"], "dimension": b["dimension"]} for b in blocks if b["type"] == "add"]
    refines = [{"target": b["value"], "dimension": b["dimension"]} for b in blocks if b["type"] == "refine"]
    return {"add_count": len({a["term"] for a in adds}), "refine_count": len({r["target"] for r in refines}),
            "would_add": adds, "would_refine": refines,
            "documented": bool(blocks), "suggestions_file": "suggestions.md" if sugg.exists() else None}


def cmd_document(args):
    """Add the failure summary to every failed run's result.json (backfill)."""
    n = 0
    for p in run_dir(args.run).glob("*/*/r*/result.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("status") == "failed":
            r["failure"] = failure_summary(p.parent)
            p.write_text(json.dumps(r, indent=1), encoding="utf-8")
            n += 1
            if not r["failure"]["documented"]:
                log(f"eval: {p.parent} failed without documented suggestions")
    log(f"eval: documented {n} failed runs")


def _image() -> str:
    return loop.build_image("pi")


def spent_so_far(run: str) -> dict:
    spent = {}
    for p in run_dir(run).glob("*/*/r*/result.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        spent[r["model"]] = spent.get(r["model"], 0) + (r.get("cost_usd") or 0)
    return spent


def select_models(arg: str, available: dict) -> list:
    models = load_models()
    if arg and arg != "all":
        wanted = set(arg.split(","))
        models = [m for m in models if m["name"] in wanted]
    out = []
    for m in models:
        if not m.get("pi_model"):
            log(f"eval: skipping {m['name']}: {m.get('note') or 'no harness model'}")
        elif not available.get(m["route"]):
            log(f"eval: skipping {m['name']}: no API key for route {m['route']}")
        else:
            out.append(m)
    return out


def cmd_translate(args):
    available = prepare_keys()
    models = select_models(args.models, available)
    items = items_of(args.run)
    if args.items:
        wanted = set(args.items.split(","))
        items = [i for i in items if i["item_key"] in wanted]
    image = _image()
    budget = {"max_usd": args.max_usd, "spent": spent_so_far(args.run)}
    # Interleaved across models (round-robin), so every route's slots fill at
    # once: model-by-model order left the pool's threads all waiting on the
    # first model's route while the other routes sat idle.
    per_model = [[(m, it, k) for it in model_items(m, items) for k in range(1, args.runs + 1)] for m in models]
    jobs = [j for group in itertools.zip_longest(*per_model) for j in group if j is not None]
    done = sum((run_dir(args.run) / m["name"] / it["item_key"] / f"r{k}" / "result.json").exists() for m, it, k in jobs)
    log(f"eval: {len(jobs)} translator runs ({done} already done, {len(jobs) - done} to run; setup {SETUP_VERSION}, "
        f"timeout {TIMEOUT_S}s, proxy concurrency {ROUTE_CONCURRENCY['proxy']}): " + ", ".join(
        f"{m['name']} {sum(j[0] is m for j in jobs)}" for m in models))
    sems = _route_semaphores()
    with Services(args.run, models=models) as svc:
        workers = sum(ROUTE_CONCURRENCY.values())
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(translate_one, m, it, k, args.run, image, svc.retriever, budget, sems): (m, it, k)
                       for m, it, k in jobs}
            for f in as_completed(futures):
                m, it, k = futures[f]
                try:
                    r = f.result()
                except Exception as e:   # one broken run must not stop the others
                    import traceback
                    traceback.print_exc()
                    r = {"status": "error", "problem": str(e)}
                if r.get("already_done"):
                    continue   # finished in an earlier session: not logged again
                log(f"eval: {m['name']} {it['item_key']} r{k} -> {r.get('status')}"
                    + (f" (${r['cost_usd']:.3f}, {r.get('duration_s', 0):.0f}s)" if r.get("cost_usd") else "")
                    + (f" [{str(r.get('problem'))[:160]}]" if r.get("status") == "error" else ""))


# ---------------------------------------------------------------------- backtranslate

def backtranslate_one(model: dict, fwd_dir: Path, item: dict, run: str, image: str, retriever, sems: dict) -> dict:
    from rag.server import render_entries
    import metrics
    back = fwd_dir / "back"
    result_path = back / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    translation = (fwd_dir / "translation.md").read_text(encoding="utf-8")
    code = metrics.braincode_code(translation)
    d = run_dir(run)
    if back.exists():
        shutil.rmtree(back)
    work = back / "work"
    for sub in ("session", "out", "attach"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    (work / "trajectory.txt").write_text(code, encoding="utf-8")
    kinds = metrics.glossary_kinds(d / "reference" / "glossary.jsonl")
    symbols, _ = metrics.symbol_counts(translation, kinds)
    keys = sorted({s.split("::", 1)[0] if "::" in s else s for s in symbols})
    entries = render_entries({k: retriever.entry(k) for k in keys}) if keys else "(no glossary symbols)"
    shutil.copy2(d / "reference" / "language-spec.compact.md", work / "attach" / "1-language-spec.md")
    (work / "attach" / "2-glossary-entries.md").write_text("# Glossary entries used by the document\n\n" + entries,
                                                         encoding="utf-8")
    shutil.copy2(EVAL_DIR / "doc_formats" / "reconstruction.md", work / "attach" / "3-format-reconstruction.md")
    prompt = tb.fill_template((EVAL_DIR / "tasks" / "backtranslator.md").read_text(encoding="utf-8"),
                              {"DATASET": item["dataset"]})
    (work / "prompt.md").write_text(prompt, encoding="utf-8")
    tid = f"{item['item_key']}-{fwd_dir.name}-back"
    container = f"{CONTAINER_PREFIX}{model['name']}-{tid}-{random.randint(1000, 9999)}"
    env = {"RAG_PORT": RAG_PORT, "PI_JSON": "1", "PI_SESSION_DIR": "/session", **tb.CONTEXT_LIMITS_ENV}
    cmd = utils.build_docker_cmd(
        None, None, model["pi_model"], d / "reference", work / "trajectory.txt", work / "prompt.md", work / "out",
        image, container_name=container, add_host=True,
        extra_mounts=[(lf.KIT_DIR, "/kit"), (work / "attach", "/attach")],
        writable_mounts=[(work / "session", "/session")], extra_env=env, passthrough_env=route_keys(model))
    started = time.monotonic()
    with sems[model["route"]]:
        proc, duration, used = tb.run_container_logged(cmd, container, TIMEOUT_S, back / "attempt1.log")
    utils.scrub_secrets(back)   # an agent running `env` would log the keys
    used = {**used, **tb.collect_context_log(work / "session", used, back / "attempt1.context.jsonl")}
    recon = work / "out" / "reconstruction.md"
    status = "ok" if recon.exists() and recon.read_text(encoding="utf-8").strip() and proc.returncode == 0 else "error"
    if recon.exists():
        shutil.copy2(recon, back / "reconstruction.md")
    tb.keep_session(work / "session", back / "session.jsonl")
    shutil.rmtree(work, ignore_errors=True)
    res = {"model": model["name"], "item": item["item_key"], "run": fwd_dir.name, "status": status,
           "duration_s": round(time.monotonic() - started, 1), "usage": used, "cost_usd": price_of(model, used),
           "problem": None if status == "ok" else (utils.last_error_line(proc.stderr or "") or "no reconstruction")}
    result_path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def cmd_backtranslate(args):
    available = prepare_keys()
    models = [m for m in select_models(args.models, available) if m.get("expressivity")]
    items = {i["item_key"]: i for i in items_of(args.run)}
    image = _image()
    # One back-translation per model and item: its first forward run with a
    # usable translation (r1, else r2, else r3).
    jobs = []
    for m in models:
        for item_key in sorted(items):
            if args.items and item_key not in args.items.split(","):
                continue
            for res in sorted(run_dir(args.run).glob(f"{m['name']}/{item_key}/r*/result.json")):
                r = json.loads(res.read_text(encoding="utf-8"))
                if r["status"] in ("success", "failed") and (res.parent / "translation.md").exists():
                    jobs.append((m, res.parent, items[item_key]))
                    break
    log(f"eval: {len(jobs)} back-translations (one per model and item)")
    sems = _route_semaphores()
    with Services(args.run, models=models) as svc:
        with ThreadPoolExecutor(max_workers=sum(ROUTE_CONCURRENCY.values())) as pool:
            futures = {pool.submit(backtranslate_one, m, fd, it, args.run, image, svc.retriever, sems): (m, fd)
                       for m, fd, it in jobs}
            for f in as_completed(futures):
                m, fd = futures[f]
                try:
                    r = f.result()
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    r = {"status": "error", "problem": str(e)}
                log(f"eval: back {m['name']} {fd.parent.name} {fd.name} -> {r['status']}"
                    + (f" [{r.get('problem')}]" if r["status"] != "ok" else ""))


# ---------------------------------------------------------------------- status

def cmd_status(args):
    from collections import Counter
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in run_dir(args.run).glob("*/*/r*/result.json")]
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    for name, rs in sorted(by_model.items()):
        c = Counter(r["status"] for r in rs)
        cost = sum(r.get("cost_usd") or 0 for r in rs)
        tokens = sum((r.get("usage") or {}).get("input", 0) + (r.get("usage") or {}).get("cacheRead", 0) for r in rs)
        print(f"{name:18} runs {len(rs):4}  {dict(c)}  ${cost:.2f}  input+cached {tokens:,}")
    backs = [json.loads(p.read_text(encoding="utf-8")) for p in run_dir(args.run).glob("*/*/r*/back/result.json")]
    if backs:
        print("back-translations:", dict(Counter((b["model"], b["status"]) for b in backs)))


def cmd_regate(args):
    """Re-check every stored success and failure with the current host gates
    (the opaque / quoted-text rules came after the forward runs). A run a gate
    now rejects becomes an error; its original status and the gate's reasons
    are kept in result.json (`regated`). A run an earlier regate rejected is
    re-checked from its original status and restored if it now passes."""
    from rag.retrieve import Retriever
    d = run_dir(args.run)
    retriever = Retriever(dense=True, glossary_path=d / "reference" / "glossary.jsonl", index_dir=d / "rag_index")
    rejected = restored = 0
    for res in sorted(d.glob("*/*/r*/result.json")):
        r = json.loads(res.read_text(encoding="utf-8"))
        doc = res.parent / "translation.md"
        original = (r.get("regated") or {}).get("original_status") or r["status"]
        if original not in ("success", "failed") or not doc.exists():
            continue
        needs = json.loads((d / "items" / r["item"] / "needs.json").read_text(encoding="utf-8"))["needs"]
        report = retriever.check(doc.read_text(encoding="utf-8"), needs)
        success = original == "success"
        gate = tb.success_gate_problems(report) if success else tb.failure_gate_problems(report)
        if gate:
            if r.get("regated", {}).get("problems") == gate and r["status"] == "error":
                continue
            lead = "claimed success but the host check found: " if success else "failed translation rejected: "
            r.update({"status": "error", "problem": lead + "; ".join(gate),
                      "regated": {"original_status": original, "problems": gate}})
            rejected += 1
            log(f"eval: regate {r['model']} {r['item']} r{r['run']} -> error [{'; '.join(gate)[:160]}]")
        elif r.get("regated"):
            r.update({"status": original, "problem": None})
            r.pop("regated")
            restored += 1
            log(f"eval: regate {r['model']} {r['item']} r{r['run']} -> {original} (restored)")
        else:
            continue
        res.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        (res.parent / "check.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"eval: regate rejected {rejected} run(s), restored {restored}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["setup", "translate", "backtranslate", "status", "document", "regate"])
    p.add_argument("--run", default="main")
    p.add_argument("--models", default="all", help="comma-separated model names from models.json, or all")
    p.add_argument("--items", default="", help="comma-separated item keys (default: the run's items)")
    p.add_argument("--runs", type=int, default=3, help="independent translations per item")
    p.add_argument("--max-usd", type=float, default=150.0, help="per-model cost cap (priced models)")
    args = p.parse_args(argv)
    {"setup": cmd_setup, "translate": cmd_translate, "backtranslate": cmd_backtranslate,
     "status": cmd_status, "document": cmd_document, "regate": cmd_regate}[args.command](args)


if __name__ == "__main__":
    main()
