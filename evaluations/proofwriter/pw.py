#!/usr/bin/env python3
"""ProofWriter experiment: does a Qwen 2.5 fine-tuned on BrainCode reason better
when it works from a BrainCode translation of the problem than the plain
Qwen 2.5 does from the English problem?

Benchmark: ProofWriter, OWA, the depth-5 dataset's test split
(tasksource/proofwriter), questions of proof depth 3 and 4 (35 each), labels
True / False / Unknown balanced within each depth.

Phases
  1  fine-tune Qwen 2.5 on the BrainCode translations     Colab (finetune-qwen/)
  2  Gemini translators translate the 70 items            here: setup, translate
  3  plain Qwen 2.5 answers the English items (baseline)  Colab (condition `baseline`)
  4  the fine-tuned Qwen solves the successful            Colab (condition `braincode_ft`)
     translations and writes its conclusion in BrainCode;
     Gemini back-translators turn that conclusion back   here: backtranslate
     into English and read off its verdict

Commands (in order)
  sample         70 items -> sample.jsonl (refuses to overwrite)
  setup          freeze glossary g19 + RAG index into ../runs/proofwriter, needs + retrieval per item
  translate      phase 2: Gemini translators (Docker + vertex proxy), resumable
  status         translation and run progress
  package        the items + successful translations -> finetune-qwen/benchmark/ for Colab
  backtranslate  phase 4b, after copying Colab's runs into runs/: the answers' BrainCode -> English -> verdict
  analyze        accuracy, paired tests, per depth and label -> results/

    python pw.py sample
    python pw.py setup
    python pw.py translate [--runs 1] [--max-usd 20]
    python pw.py package
    python pw.py backtranslate
    python pw.py analyze
"""
import argparse
import json
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parent
REPO = EVAL_DIR.parent
sys.path.insert(0, str(REPO / "swarm"))
sys.path.insert(0, str(EVAL_DIR))

DATA = HERE / "data" / "proofwriter_test.parquet"
DATA_URL = ("https://huggingface.co/datasets/tasksource/proofwriter/resolve/refs%2Fconvert%2Fparquet/"
            "default/test/0000.parquet")
SAMPLE = HERE / "sample.jsonl"
RUNS = HERE / "runs"                      # Colab's runs, copied back: runs/<condition>/<item>/r<k>/
RESULTS = HERE / "results"
TASKS = HERE / "tasks"
FORMATS = HERE / "doc_formats"
PACKAGE = REPO / "finetune-qwen" / "benchmark"
RUN = "proofwriter"                       # the translation run: ../runs/proofwriter (run_eval layout)
REF_RUN = "main"                          # its frozen reference (glossary g19) and RAG index are copied from here
TRANSLATOR = "gemini-flash-3.7"           # the swarm's own translator model (models.json)
BACKTRANSLATOR = "gemini-flash-3.7"
DEPTHS = (3, 4)
PER_DEPTH = 35
LABELS = ("True", "False", "Unknown")
SEED = 2026
BRAINCODE_CONDITIONS = ("braincode_ft", "braincode_base")
VERDICT_RE = re.compile(r"^\s*Verdict:\s*(True|False|Unknown|None)\s*$", re.I | re.M)
BLOCK_RE = re.compile(r"```braincode\s*\n(.*?)```", re.S)


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------- sample

def item_text(theory: str, question: str) -> str:
    """The English item: one theory sentence per line, then the question. This
    is what the translators translate and what the baseline answers."""
    sentences = [s.strip() + "." for s in theory.strip().rstrip(".").split(". ") if s.strip()]
    return ("<|user|>Theory:\n" + "\n".join(sentences) + "\n\nQuestion: Based only on the theory, is the "
            "following statement True, False, or Unknown?\n" + question.strip())


def cmd_sample(args):
    import pandas as pd
    if SAMPLE.exists():
        sys.exit(f"pw: {SAMPLE.name} exists; delete it to resample")
    if not DATA.exists():
        import urllib.request
        DATA.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA)
    d = pd.read_parquet(DATA)
    d = d[(d["config"] == "depth-5") & d["id"].str.contains("-OWA-")]
    rng = random.Random(SEED)
    items, used_theories = [], set()
    for depth in DEPTHS:
        # 35 per depth: 12 / 12 / 11, the short label rotating with the depth
        quota = {lab: 12 for lab in LABELS}
        quota[LABELS[depth % 3]] = 11
        for lab in LABELS:
            rows = d[(d["QDep"] == depth) & (d["answer"] == lab)].to_dict("records")
            rng.shuffle(rows)
            taken = 0
            for r in rows:
                if taken == quota[lab]:
                    break
                if r["id"] in used_theories:      # one question per theory: no item shares a theory
                    continue
                used_theories.add(r["id"])
                taken += 1
                n = len(items) + 1
                text = item_text(r["theory"], r["question"])
                items.append({
                    "item_key": f"proofwriter-d{depth}-{n:02d}", "dataset": "proofwriter", "all_models": True,
                    "item_id": r["id"], "stratum": f"depth-{depth}", "depth": depth, "label": lab,
                    "question": r["question"], "chars": len(text),
                    "record_line": json.dumps({"id": r["id"], "source": "ProofWriter (tasksource/proofwriter, "
                                               "depth-5 OWA, test)", "content": text}, ensure_ascii=False)})
    with SAMPLE.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    c = Counter((i["depth"], i["label"]) for i in items)
    log(f"pw: sampled {len(items)} items -> {SAMPLE.name}: {dict(sorted(c.items()))}")


def load_sample():
    return [json.loads(line) for line in SAMPLE.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------- phase 2: translate

def cmd_setup(args):
    import run_eval
    import sample as sample_mod
    src, dst = run_eval.run_dir(REF_RUN), run_eval.run_dir(RUN)
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("reference", "rag_index"):       # glossary g19: the release the training data was checked against
        if not (dst / name).exists():
            shutil.copytree(src / name, dst / name)
    if not (dst / "meta.json").exists():
        meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
        meta.update({"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sample": str(SAMPLE),
                     "items": len(load_sample()), "reference_from": f"runs/{REF_RUN}"})
        (dst / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    sample_mod.load = load_sample                 # run_eval's setup, on this experiment's items
    run_eval.cmd_setup(argparse.Namespace(run=RUN, items=""))


def cmd_translate(args):
    import run_eval
    run_eval.cmd_translate(argparse.Namespace(run=RUN, models=TRANSLATOR, items=args.items, runs=args.runs,
                                              max_usd=args.max_usd))


def translation_results():
    """{item_key: [result dicts of its translation runs]}"""
    import run_eval
    out = defaultdict(list)
    for p in sorted(run_eval.run_dir(RUN).glob(f"{TRANSLATOR}/*/r*/result.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        r["dir"] = p.parent
        out[r["item"]].append(r)
    return out


def successful_translation(results: list):
    """The first run (r1, r2, ...) with Status: success and a translation."""
    for r in sorted(results, key=lambda r: r["run"]):
        if r["status"] == "success" and (r["dir"] / "translation.md").exists():
            return r
    return None


def cmd_status(args):
    items = load_sample()
    tr = translation_results()
    by_depth = defaultdict(Counter)
    for it in items:
        rs = tr.get(it["item_key"], [])
        state = ("translated" if successful_translation(rs) else
                 "failed" if any(r["status"] == "failed" for r in rs) else
                 "error" if rs else "not run")
        by_depth[it["depth"]][state] += 1
    for depth, c in sorted(by_depth.items()):
        log(f"pw: depth {depth}: {dict(c)}")
    cost = sum(r.get("cost_usd") or 0 for rs in tr.values() for r in rs)
    log(f"pw: translation cost so far ${cost:.2f}")
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in RUNS.glob("*/*/r*/result.json")]
    for cond, c in sorted(Counter((r["condition"], state_of(r)) for r in runs).items()):
        log(f"pw: {cond[0]:16} {cond[1]:22} {c}")


def state_of(r: dict) -> str:
    if r.get("status") != "ok":
        return "error"
    if r.get("correct") is None:
        return "awaiting back-translation"
    return "correct" if r["correct"] else "wrong"


# ---------------------------------------------------------------------- package for Colab

def glossary_entries_for(code: str, retriever, kinds) -> str:
    import metrics
    from rag.server import render_entries
    symbols, _ = metrics.symbol_counts(code, kinds)
    keys = sorted({s.split("::", 1)[0] if "::" in s else s for s in symbols})
    return render_entries({k: retriever.entry(k) for k in keys}) if keys else "(no glossary symbols)"


def cmd_package(args):
    """finetune-qwen/benchmark/: sample.jsonl (English items, labels, and for each
    successfully translated item its BrainCode and the glossary entries it uses)
    plus the compact spec. Colab needs nothing else from the repository."""
    import metrics
    import run_eval
    from rag.retrieve import Retriever, braincode_code
    d = run_eval.run_dir(RUN)
    retriever = Retriever(dense=False, glossary_path=d / "reference" / "glossary.jsonl", index_dir=d / "rag_index",
                          use_llm=False)
    kinds = metrics.glossary_kinds(d / "reference" / "glossary.jsonl")
    tr = translation_results()
    PACKAGE.mkdir(parents=True, exist_ok=True)
    rows, n_ok = [], 0
    for it in load_sample():
        ok = successful_translation(tr.get(it["item_key"], []))
        row = {k: it[k] for k in ("item_key", "item_id", "depth", "label", "question")}
        row["input"] = json.loads(it["record_line"])["content"]
        row["braincode"] = None
        if ok:
            code = braincode_code((ok["dir"] / "translation.md").read_text(encoding="utf-8")).strip()
            row.update({"braincode": code, "translation_run": ok["run"],
                        "glossary_entries": glossary_entries_for(code, retriever, kinds)})
            n_ok += 1
        rows.append(row)
    with (PACKAGE / "sample.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.copy2(d / "reference" / "language-spec.compact.md", PACKAGE / "language-spec.compact.md")
    if any(not tr.get(i["item_key"]) for i in load_sample()):
        log("pw: warning: some items have no translation run yet (run `translate` first)")
    log(f"pw: packaged {len(rows)} items ({n_ok} with a successful translation) -> {PACKAGE}")


# ---------------------------------------------------------------------- phase 4b: back-translate the answers

DEF_RE = re.compile(r"->\s*(\w+)\s*:\s*[A-Z]")
HANDLE_RE = re.compile(r"\b[a-z_][a-z0-9_]*\b")


def verdict_slice(block: str) -> str:
    """The verdict of a block that re-encodes the whole problem (its theory must not
    reach the back-translator, which could then solve the problem itself): the
    trailing CLAIM lines, plus every TERM/LET/CLAIM line they reference, transitively."""
    lines = [ln.strip() for ln in block.splitlines()]
    defs = {m.group(1): ln for ln in lines if (m := DEF_RE.search(ln))}
    tail = []
    for ln in reversed(lines):
        if not ln or ln.startswith(("}", "UTTER", "#")):
            continue
        if not ln.startswith("CLAIM"):
            break
        tail.append(ln)
    keep, todo = set(), list(reversed(tail))
    while todo:
        ln = todo.pop()
        if ln in keep:
            continue
        keep.add(ln)
        own = (m.group(1) if (m := DEF_RE.search(ln)) else None)
        todo += [defs[h] for h in HANDLE_RE.findall(ln.split("->")[0]) if h in defs and h != own]
    return "\n".join(ln for ln in lines if ln in keep)


def answer_block(response: str):
    """The relevant part of a BrainCode-condition answer: its last ```braincode block,
    cut down to the verdict when the block re-encodes the problem."""
    blocks = BLOCK_RE.findall(response)
    if not blocks:
        return None
    block = blocks[-1].strip()
    if re.search(r"^\s*(MODE|TURN)\b", block, re.M) or block.count("CLAIM ") > 6:
        block = verdict_slice(block) or block
    return block


def backtranslate_answer(model: dict, run_path: Path, item: dict, image: str, retriever, kinds, sems) -> dict:
    """One Gemini back-translator (the evaluation's container setup) reads only
    the answer's BrainCode block and the question statement (never the theory,
    so it cannot solve the problem itself), writes the conclusion in English and
    says which verdict it states. The run's result.json is then scored."""
    import run_eval
    import translate_batch as tb
    import utils
    import loop_files as lf
    res_path = run_path / "result.json"
    record = json.loads(res_path.read_text(encoding="utf-8"))
    back = run_path / "back"
    done = back / "result.json"
    if done.exists() and json.loads(done.read_text(encoding="utf-8"))["status"] == "ok":
        return json.loads(done.read_text(encoding="utf-8"))
    response = (run_path / "response.md").read_text(encoding="utf-8")
    block = answer_block(response)
    if block is None:                             # no conclusion in BrainCode: scored wrong, nothing to translate
        record.update({"prediction": "none", "correct": False, "verdict_source": "no braincode block"})
        res_path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        return {"status": "no block"}
    if back.exists():
        shutil.rmtree(back)
    work = back / "work"
    for sub in ("session", "out", "attach"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    (work / "trajectory.txt").write_text(block, encoding="utf-8")
    d = run_eval.run_dir(RUN)
    shutil.copy2(d / "reference" / "language-spec.compact.md", work / "attach" / "1-language-spec.md")
    (work / "attach" / "2-glossary-entries.md").write_text(
        "# Glossary entries used by the conclusion\n\n" + glossary_entries_for(block, retriever, kinds),
        encoding="utf-8")
    shutil.copy2(FORMATS / "answer_reconstruction.md", work / "attach" / "3-format-answer-reconstruction.md")
    (work / "attach" / "4-question.md").write_text("# The statement in question\n\n" + item["question"] + "\n",
                                                  encoding="utf-8")
    (work / "prompt.md").write_text((TASKS / "answer_backtranslator.md").read_text(encoding="utf-8"),
                                    encoding="utf-8")
    tid = f"{item['item_key']}-{run_path.parent.parent.name}-{run_path.name}-back"
    container = f"{run_eval.CONTAINER_PREFIX}{model['name']}-{tid}-{random.randint(1000, 9999)}"
    env = {"RAG_PORT": run_eval.RAG_PORT, "PI_JSON": "1", "PI_SESSION_DIR": "/session", **tb.CONTEXT_LIMITS_ENV}
    cmd = utils.build_docker_cmd(
        None, None, model["pi_model"], d / "reference", work / "trajectory.txt", work / "prompt.md", work / "out",
        image, container_name=container, add_host=True,
        extra_mounts=[(lf.KIT_DIR, "/kit"), (work / "attach", "/attach")],
        writable_mounts=[(work / "session", "/session")], extra_env=env, passthrough_env=run_eval.route_keys(model))
    started = time.monotonic()
    with sems[model["route"]]:
        proc, _, used = tb.run_container_logged(cmd, container, run_eval.TIMEOUT_S, back / "attempt1.log")
    utils.scrub_secrets(back)   # an agent running `env` would log the keys
    used = {**used, **tb.collect_context_log(work / "session", used, back / "attempt1.context.jsonl")}
    recon = work / "out" / "reconstruction.md"
    text = recon.read_text(encoding="utf-8") if recon.exists() else ""
    m = VERDICT_RE.findall(text)
    status = "ok" if m and proc.returncode == 0 else "error"
    if recon.exists():
        shutil.copy2(recon, back / "reconstruction.md")
    tb.keep_session(work / "session", back / "session.jsonl")
    shutil.rmtree(work, ignore_errors=True)
    verdict = m[-1].capitalize() if m else None
    res = {"model": model["name"], "status": status, "verdict": verdict,
           "duration_s": round(time.monotonic() - started, 1), "usage": used,
           "cost_usd": run_eval.price_of(model, used),
           "problem": None if status == "ok" else (utils.last_error_line(proc.stderr or "") or "no verdict")}
    done.write_text(json.dumps(res, indent=1), encoding="utf-8")
    if status == "ok":
        record.update({"prediction": verdict.lower(), "correct": verdict.lower() == item["label"].lower(),
                       "verdict_source": "back-translation"})
        res_path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def cmd_backtranslate(args):
    import metrics
    import run_eval
    available = run_eval.prepare_keys()
    model = next(m for m in run_eval.select_models(BACKTRANSLATOR, available))
    items = {i["item_key"]: i for i in load_sample()}
    jobs = sorted(p.parent for c in BRAINCODE_CONDITIONS for p in RUNS.glob(f"{c}/*/r*/result.json")
                  if json.loads(p.read_text(encoding="utf-8"))["status"] == "ok")
    if not jobs:
        sys.exit(f"pw: no BrainCode-condition runs under {RUNS}; copy Colab's runs there first")
    image = run_eval._image()
    sems = run_eval._route_semaphores()
    d = run_eval.run_dir(RUN)
    kinds = metrics.glossary_kinds(d / "reference" / "glossary.jsonl")
    log(f"pw: {len(jobs)} answers to back-translate ({BACKTRANSLATOR})")
    with run_eval.Services(RUN, models=[model]) as svc:
        with ThreadPoolExecutor(max_workers=run_eval.ROUTE_CONCURRENCY["proxy"]) as pool:
            futs = {pool.submit(backtranslate_answer, model, p, items[p.parent.name], image, svc.retriever, kinds,
                                sems): p for p in jobs}
            for f in as_completed(futs):
                p = futs[f]
                try:
                    r = f.result()
                except Exception as e:  # noqa: BLE001 - one broken run must not stop the others
                    r = {"status": "error", "problem": str(e)}
                log(f"pw: back {p.parent.parent.name} {p.parent.name} {p.name} -> {r['status']}"
                    + (f" verdict {r.get('verdict')}" if r.get("verdict") else "")
                    + (f" [{r.get('problem')}]" if r["status"] == "error" else ""))


# ---------------------------------------------------------------------- solvability check

SOLVER = "gemini-flash"     # the proxy's Gemini 3.7 Flash


def runner_module():
    """finetune-qwen/run_proofwriter.py: the exact prompts Qwen gets in Colab."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_proofwriter", PACKAGE.parent / "run_proofwriter.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def proxy_chat(messages: list) -> dict:
    import os
    import urllib.error
    import urllib.request
    body = json.dumps({"model": SOLVER, "messages": messages, "max_tokens": 16000}).encode()
    for attempt in range(4):
        req = urllib.request.Request(os.environ["PROXY_BASE_URL"].rstrip("/") + "/chat/completions", data=body,
                                     method="POST", headers={"Content-Type": "application/json",
                                                             "Authorization": f"Bearer {os.environ['PROXY_API_KEY']}"})
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                r = json.loads(resp.read())
            u = r.get("usage") or {}
            return {"text": r["choices"][0]["message"].get("content") or "",
                    "prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0)}
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(15 * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace')}")


def cmd_solvability(args):
    """Can a strong model still solve the problems from their BrainCode? Gemini 3.7 Flash answers the
    translated items twice, from the English and from the BrainCode, with the same prompts as Qwen
    (call 1 of the BrainCode runs: reason, then `The answer is: ...`). Conditions gemini_english and
    gemini_braincode; the gap between them is what the translations alone cost."""
    import utils
    utils.load_dotenv(REPO / "swarm" / ".env")
    import os
    os.environ.setdefault("PROXY_BASE_URL", "https://vertex-proxy-v26q.onrender.com/v1")
    rp = runner_module()
    items = [json.loads(line) for line in (PACKAGE / "sample.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()]
    items = [i for i in items if i["braincode"]]

    def one(item, kind):
        out = RUNS / f"gemini_{kind}" / item["item_key"] / "r1"
        res = out / "result.json"
        if res.exists() and json.loads(res.read_text(encoding="utf-8"))["status"] == "ok":
            return None
        out.mkdir(parents=True, exist_ok=True)
        msgs = rp.messages_for(item, kind, True)
        rec = {"item": item["item_key"], "depth": item["depth"], "label": item["label"],
               "condition": f"gemini_{kind}", "input": kind, "run": 1, "model": SOLVER}
        try:
            r = proxy_chat(msgs)
            (out / "response.md").write_text(r["text"], encoding="utf-8")
            m = rp.ANSWER_RE.findall(r["text"])
            pred = m[-1].lower() if m else "none"
            rec.update({"status": "ok", "prediction": pred, "correct": pred == item["label"].lower(),
                        "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"]})
        except Exception as e:  # noqa: BLE001 - recorded; redone on the next run
            rec.update({"status": "error", "problem": str(e)[:300], "correct": None})
        res.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        return rec

    jobs = [(it, kind) for it in items for kind in ("english", "braincode")]
    log(f"pw: solvability: {len(jobs)} calls ({len(items)} items x English/BrainCode) on {SOLVER}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(one, it, kind): (it, kind) for it, kind in jobs}
        for f in as_completed(futs):
            it, kind = futs[f]
            r = f.result()
            if r:
                log(f"pw: gemini_{kind:9} {it['item_key']} -> "
                    + (("correct" if r["correct"] else f"wrong (pred {r['prediction']}, gold {it['label']})")
                       if r["status"] == "ok" else "ERROR " + r["problem"][:120]))
    for kind in ("english", "braincode"):
        rs = [json.loads(p.read_text(encoding="utf-8")) for p in RUNS.glob(f"gemini_{kind}/*/r1/result.json")]
        ok = [r for r in rs if r["status"] == "ok"]
        log(f"pw: gemini_{kind}: {sum(r['correct'] for r in ok)}/{len(ok)} correct"
            + (f" ({sum(r['correct'] for r in ok) / len(ok):.0%})" if ok else ""))


# ---------------------------------------------------------------------- analysis

COND_LABELS = {"baseline": "Qwen 2.5, English (baseline)",
               "braincode_ft": "Fine-tuned Qwen 2.5, BrainCode → BrainCode verdict",
               "braincode_ft_english": "Fine-tuned Qwen 2.5, BrainCode → English answer",
               "braincode_base": "Qwen 2.5, BrainCode → BrainCode verdict (no fine-tuning)",
               "braincode_base_english": "Qwen 2.5, BrainCode → English answer (no fine-tuning)",
               "baseline_ft": "Fine-tuned Qwen 2.5, English",
               "gemini_english": "Gemini 3.7 Flash, English (solvability reference)",
               "gemini_braincode": "Gemini 3.7 Flash, BrainCode → English answer (solvability)"}


def english_decision_rows(rows: list) -> list:
    """For each BrainCode-condition run, its English decision (call 1, `direct_*`) as a run of the
    derived condition `<condition>_english`: the same reasoning, scored before any BrainCode verdict."""
    return [{**r, "condition": r["condition"] + "_english", "prediction": r["direct_prediction"],
             "correct": r["direct_correct"]}
            for r in rows if r["condition"] in BRAINCODE_CONDITIONS and r.get("status") == "ok"
            and r.get("direct_prediction") is not None]


def cmd_analyze(args):
    import pandas as pd
    from statsmodels.stats.contingency_tables import mcnemar
    sys.path.insert(0, str(EVAL_DIR / "improvement"))
    from improve import wilson
    RESULTS.mkdir(exist_ok=True)
    items = {i["item_key"]: i for i in load_sample()}
    tr = translation_results()
    translated = {k for k in items if successful_translation(tr.get(k, []))}
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in RUNS.glob("*/*/r*/result.json")]
    rows = [r for r in rows if not r["condition"].startswith("smoke") and not r["condition"].endswith("_onecall")]
    verdict_runs = [r for r in rows if r["condition"] in BRAINCODE_CONDITIONS and r["status"] == "ok"]
    rows += english_decision_rows(rows)
    pending = Counter(r["condition"] for r in rows if r["status"] == "ok" and r.get("correct") is None)
    errors = Counter(r["condition"] for r in rows if r["status"] != "ok")
    df = pd.DataFrame([r for r in rows if r["status"] == "ok" and r.get("correct") is not None])
    if df.empty:
        sys.exit("pw: no scored runs yet")
    df["correct"] = df["correct"].astype(int)
    df["depth"] = df["item"].map(lambda k: items[k]["depth"])
    df["label"] = df["item"].map(lambda k: items[k]["label"])
    conds = [c for c in COND_LABELS if c in set(df["condition"])]
    n_items = len(items)
    L = [f"# ProofWriter (OWA, depth-5 test, proof depth {'/'.join(map(str, DEPTHS))}): Qwen 2.5 vs "
         f"fine-tuned Qwen 2.5 on BrainCode", "",
         f"{n_items} items ({PER_DEPTH} per depth, labels balanced). Translator: {TRANSLATOR}; "
         f"back-translator: {BACKTRANSLATOR}. Errored runs left out: {dict(errors) or 'none'}; "
         f"awaiting back-translation: {dict(pending) or 'none'}.", "",
         "## Phase 2: translation", "", "| depth | items | successful translations | rate |", "|---|---:|---:|---:|"]
    for depth in DEPTHS:
        keys = [k for k, i in items.items() if i["depth"] == depth]
        ok = sum(k in translated for k in keys)
        L.append(f"| {depth} | {len(keys)} | {ok} | {ok / len(keys):.0%} |")
    L.append(f"| all | {n_items} | {len(translated)} | {len(translated) / n_items:.0%} |")

    def acc_table(sub_df, title, note):
        out = ["", f"## {title}", "", note, "", "| condition | depth 3 | depth 4 | all | 95% CI (Wilson) |",
               "|---|---:|---:|---:|---|"]
        for c in conds:
            s = sub_df[sub_df.condition == c]
            if s.empty:
                continue
            per = [s[s.depth == dp] for dp in DEPTHS]
            p_, lo, hi = wilson(int(s.correct.sum()), len(s))
            out.append(f"| {COND_LABELS[c]} | " + " | ".join(f"{x.correct.mean():.1%} ({len(x)})" if len(x) else "—"
                                                        for x in per)
                       + f" | {p_:.1%} ({len(s)}) | {lo:.1%}–{hi:.1%} |")
        return out

    paired = df[df["item"].isin(translated)]
    L += acc_table(paired, "Accuracy on the successfully translated items (the paired comparison)",
                   f"{len(translated)} items: every condition answers the same items. Cells: accuracy (runs).")
    # Intention to treat: every item; a BrainCode condition counts an untranslated item as wrong in every run
    itt = df.copy()
    for c in [c for c in conds if c.startswith("braincode")]:      # the verdict and English-answer conditions
        runs_per_item = max(1, int(df[df.condition == c].groupby("item").size().max()))
        missing = [{"condition": c, "item": k, "run": r, "correct": 0, "depth": items[k]["depth"],
                    "label": items[k]["label"]} for k in items if k not in translated
                   for r in range(1, runs_per_item + 1)]
        itt = pd.concat([itt, pd.DataFrame(missing)], ignore_index=True)
    L += acc_table(itt, "Accuracy on all items (intention to treat)",
                   "An item without a successful translation counts as wrong for the BrainCode conditions: "
                   "the whole pipeline is measured, translation included.")

    for c in conds:
        if c == "baseline" or "baseline" not in conds:
            continue
        maj = (paired.groupby(["item", "condition"])["correct"].mean().unstack() >= 0.5)
        maj = maj.dropna(subset=["baseline", c]).astype(int) if c in maj else None
        if maj is None or maj.empty:
            continue
        a = int(((maj.baseline == 1) & (maj[c] == 1)).sum())
        b = int(((maj.baseline == 1) & (maj[c] == 0)).sum())
        c_ = int(((maj.baseline == 0) & (maj[c] == 1)).sum())
        d_ = int(((maj.baseline == 0) & (maj[c] == 0)).sum())
        p = mcnemar([[a, b], [c_, d_]], exact=True).pvalue
        diff = paired[paired.condition == c].correct.mean() - paired[paired.condition == "baseline"].correct.mean()
        L += ["", f"## {COND_LABELS[c]} vs baseline (paired, {len(maj)} items)", "",
              f"Accuracy difference {diff:+.1%} (percentage points, all runs). McNemar, exact, on the per-item "
              "majority vote over runs:", "",
              f"| | {c} correct | wrong |", "|---|---:|---:|", f"| baseline correct | {a} | {b} |",
              f"| baseline wrong | {c_} | {d_} |", "",
              f"Discordant pairs: {b} baseline-only vs {c_} {c}-only; exact p = {p:.4f}."]
        try:
            import statsmodels.api as sm
            pair = paired[paired.condition.isin(["baseline", c])].copy()
            pair["treated"] = (pair.condition == c).astype(int)
            gee = sm.GEE.from_formula("correct ~ treated", groups="item", data=pair, family=sm.families.Binomial(),
                                      cov_struct=sm.cov_struct.Exchangeable()).fit()
            L.append(f"GEE (item-clustered logistic): log-odds {gee.params['treated']:+.3f}, "
                     f"p = {gee.pvalues['treated']:.4f}.")
        except Exception as e:  # noqa: BLE001 - e.g. no variation
            L.append(f"GEE not estimable: {e}")

    if verdict_runs:
        L += ["", "## BrainCode conditions: English answer (call 1) vs BrainCode verdict (call 2)", "",
              "The model reasons in English from the BrainCode problem and ends with an answer (call 1), then writes "
              "that decision as a BrainCode verdict block, which a Gemini back-translator reads (call 2). *Usable* = "
              "the back-translator found a True/False/Unknown verdict; *agrees* = the verdict says what the English "
              "answer said.", "",
              "| condition | runs | English answer accuracy | BrainCode verdict accuracy | verdict block written | "
              "verdict usable | verdict agrees with English answer |", "|---|---:|---:|---:|---:|---:|---:|"]
        for c in BRAINCODE_CONDITIONS:
            rs = [r for r in verdict_runs if r["condition"] == c and r["item"] in translated]
            if not rs:
                continue
            scored = [r for r in rs if r.get("correct") is not None]
            usable = [r for r in scored if (r.get("prediction") or "none") in ("true", "false", "unknown")]

            def share(xs, n):
                return f"{len(xs) / n:.0%}" if n else "—"
            L.append(f"| {c} | {len(rs)} | {share([r for r in rs if r.get('direct_correct')], len(rs))} | "
                     f"{share([r for r in scored if r['correct']], len(scored))} | "
                     f"{share([r for r in rs if r.get('has_answer_block')], len(rs))} | "
                     f"{share(usable, len(scored))} | "
                     f"{share([r for r in usable if r['prediction'] == r.get('direct_prediction')], len(usable))} |")

    L += ["", "## Accuracy by gold label (translated items)", "", "| condition | " + " | ".join(LABELS) + " |",
          "|---|" + "---:|" * len(LABELS)]
    for c in conds:
        s = paired[paired.condition == c]
        L.append(f"| {COND_LABELS[c]} | " + " | ".join(
            f"{s[s.label == lab].correct.mean():.0%}" if len(s[s.label == lab]) else "—" for lab in LABELS) + " |")
    L += ["", "## Predictions (translated items)", "", "| condition | " + " | ".join(
        ["true", "false", "unknown", "other"]) + " |", "|---|---:|---:|---:|---:|"]
    for c in conds:
        pred = paired[paired.condition == c]["prediction"].fillna("none").str.lower()
        n = Counter(p if p in ("true", "false", "unknown") else "other" for p in pred)
        L.append(f"| {COND_LABELS[c]} | {n['true']} | {n['false']} | {n['unknown']} | {n['other']} |")
    (RESULTS / "proofwriter.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    df.to_csv(RESULTS / "proofwriter_runs.csv", index=False)
    log(f"pw: wrote {RESULTS / 'proofwriter.md'}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["sample", "setup", "translate", "status", "package", "backtranslate",
                                       "solvability", "analyze"])
    p.add_argument("--runs", type=int, default=1, help="translation runs per item (translate)")
    p.add_argument("--items", default="", help="comma-separated item keys (translate)")
    p.add_argument("--max-usd", type=float, default=20.0, help="translation cost cap (translate)")
    p.add_argument("--workers", type=int, default=8, help="parallel calls (solvability)")
    args = p.parse_args(argv)
    {"sample": cmd_sample, "setup": cmd_setup, "translate": cmd_translate, "status": cmd_status,
     "package": cmd_package, "backtranslate": cmd_backtranslate, "solvability": cmd_solvability,
     "analyze": cmd_analyze}[args.command](args)


if __name__ == "__main__":
    main()
