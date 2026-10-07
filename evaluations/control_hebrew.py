#!/usr/bin/env python3
"""Hebrew control for expressivity: the same round trip as the BrainCode
expressivity evaluation, through an ordinary natural language instead.

    English item -> Hebrew (model A) -> English (model A, never sees the original)

Same 48 items, same Gemini models, same pi container harness and proxy, same
metrics (metrics.bleu / rouge_l / word_levenshtein_similarity / corpus_bleu).
The containers get no BrainCode: /reference is an empty directory, no /kit,
no RAG server, and /attach holds only the output format.

    python control_hebrew.py translate      [--models a,b] [--items k1,k2]   # English -> Hebrew
    python control_hebrew.py backtranslate  [--models a,b] [--items k1,k2]   # Hebrew -> English
    python control_hebrew.py status
    python control_hebrew.py analyze        # -> results/main/expressivity_hebrew_control.{md,csv,png}

Runs are saved in runs/main-hebrew/<model>/<item>/r1/ (translation.md,
result.json, session.jsonl, attempt<N>.log, back/). Resumable: finished steps
are skipped, errored ones redone.
"""
import argparse
import json
import random
import re
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent / "swarm"))
sys.path.insert(0, str(EVAL_DIR))

import metrics  # noqa: E402

SOURCE_RUN = EVAL_DIR / "runs" / "main"          # items and originals (and the BrainCode round trips)
OUT = EVAL_DIR / "runs" / "main-hebrew"
RESULTS = EVAL_DIR / "results" / "main"
MODELS = "gemini-flash-3.7,gemini-flash-high-3.7"
MARKER_RE = re.compile(r"<\|(?:user|assistant)\|>")
HEBREW_RE = re.compile(r"[א-ת]")
LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
LATIN_RE = re.compile(r"[A-Za-z]")
CODE_RE = re.compile(r"```.*?```|`[^`\n]*`|https?://\S+|\S*[/\\]\S+|\S+\.\w{1,5}\b", re.S)
# Of the letters outside code, URLs and paths: below this the text wasn't translated. Low on purpose: UI labels
# stay verbatim by design, and a Mind2Web action list is mostly labels (a correct translation can be ~15% Hebrew).
MIN_HEBREW_SHARE = 0.05
STEPS = {
    "forward": {"task": "translator_hebrew.md", "format": "translation_hebrew.md", "attach": "1-format-translation.md",
                "output": "translation.md"},
    "back": {"task": "backtranslator_hebrew.md", "format": "reconstruction.md",
             "attach": "1-format-reconstruction.md", "output": "reconstruction.md"},
}


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------- checks

def markers(text: str) -> list:
    return MARKER_RE.findall(text or "")


def hebrew_share(text: str) -> float:
    """Hebrew letters / all letters, outside code, URLs and paths (kept verbatim by design)."""
    prose = CODE_RE.sub(" ", MARKER_RE.sub(" ", text or ""))
    letters = LETTER_RE.findall(prose)
    return len(HEBREW_RE.findall(prose)) / len(letters) if letters else 0.0


def latin_word_share(text: str) -> float:
    """Words with Latin letters / all words of a Hebrew translation: text left in English
    (code, names, labels), the counterpart of BrainCode's quoted share."""
    words = metrics.WORD_RE.findall(MARKER_RE.sub(" ", text or ""))
    return sum(bool(LATIN_RE.search(w)) for w in words) / len(words) if words else 0.0


def output_problems(step: str, original: str, text: str) -> list:
    problems = []
    if not text.strip():
        return ["empty output"]
    if markers(text) != markers(original):
        problems.append(f"turn markers {len(markers(text))} (expected {len(markers(original))}, same order)")
    if step == "forward" and hebrew_share(text) < MIN_HEBREW_SHARE:
        problems.append(f"only {hebrew_share(text):.0%} of the prose letters are Hebrew")
    return problems


# ---------------------------------------------------------------------- container steps

def items() -> list:
    return [json.loads(line) for line in (SOURCE_RUN / "items.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]


def original_of(item_key: str) -> str:
    return (SOURCE_RUN / "items" / item_key / "item_raw.txt").read_text(encoding="utf-8")


def run_step(step: str, model: dict, item: dict, image: str, sems: dict) -> dict:
    import run_eval
    import translate_batch as tb
    import utils
    cfg = STEPS[step]
    run_dir = OUT / model["name"] / item["item_key"] / "r1"
    out_dir = run_dir if step == "forward" else run_dir / "back"
    res_path = out_dir / "result.json"
    original = original_of(item["item_key"])
    if res_path.exists():
        prev = json.loads(res_path.read_text(encoding="utf-8"))
        if prev["status"] == "ok":
            return {"status": "done"}
        saved = out_dir / cfg["output"]
        if saved.exists() and not output_problems(step, original, saved.read_text(encoding="utf-8")):
            # rejected by an earlier, stricter check: the saved output passes today's, so keep it (no new call)
            prev.update({"status": "ok", "problem": None, "rechecked": prev["problem"]})
            res_path.write_text(json.dumps(prev, ensure_ascii=False, indent=1), encoding="utf-8")
            return {**prev, "status": "ok"}
    if step == "forward":
        source = original
    else:
        fwd = run_dir / "result.json"
        if not fwd.exists() or json.loads(fwd.read_text(encoding="utf-8"))["status"] != "ok":
            return {"status": "skipped", "problem": "no usable Hebrew translation"}
        source = (run_dir / "translation.md").read_text(encoding="utf-8")
    work = out_dir / "work"
    if work.exists():
        shutil.rmtree(work)
    for sub in ("session", "out", "attach", "empty_reference"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    (work / "trajectory.txt").write_text(source, encoding="utf-8")
    shutil.copy2(EVAL_DIR / "doc_formats" / cfg["format"], work / "attach" / cfg["attach"])
    prompt = tb.fill_template((EVAL_DIR / "tasks" / cfg["task"]).read_text(encoding="utf-8"),
                              {"DATASET": item["dataset"]})
    usage, problems, text, started = {}, [], "", time.monotonic()
    for attempt in (1, 2):
        feedback = ("" if attempt == 1 else "\n\n## Your previous attempt was rejected\n\n" + "; ".join(problems)
                    + ". Fix this and write the whole output again.")
        (work / "prompt.md").write_text(prompt + feedback, encoding="utf-8")
        for f in (work / "out").iterdir():
            f.unlink()
        container = (f"{run_eval.CONTAINER_PREFIX}{model['name']}-he-{step}-{item['item_key']}-{attempt}-"
                     f"{random.randint(1000, 9999)}")
        # No tool-result cap: its truncation notice points the model at the BrainCode kit (/kit/rag.mjs),
        # and the items are at most 6,000 characters anyway. Compaction stays as in the swarm.
        env = {"PI_JSON": "1", "PI_SESSION_DIR": "/session", **tb.CONTEXT_LIMITS_ENV, "PI_TOOL_RESULT_MAX_CHARS": "0"}
        cmd = utils.build_docker_cmd(
            None, None, model["pi_model"], work / "empty_reference", work / "trajectory.txt", work / "prompt.md",
            work / "out", image, container_name=container, add_host=True,
            extra_mounts=[(work / "attach", "/attach")], writable_mounts=[(work / "session", "/session")],
            extra_env=env, passthrough_env=run_eval.route_keys(model))
        with sems[model["route"]]:
            proc, _, used = tb.run_container_logged(cmd, container, run_eval.TIMEOUT_S,
                                                    out_dir / f"attempt{attempt}.log")
        utils.scrub_secrets(out_dir)   # an agent running `env` would log the keys
        used = {**used, **tb.collect_context_log(work / "session", used, out_dir / f"attempt{attempt}.context.jsonl")}
        for k, v in used.items():
            usage[k] = usage.get(k, 0) + v
        produced = work / "out" / cfg["output"]
        text = produced.read_text(encoding="utf-8", errors="replace") if produced.exists() else ""
        problems = ([f"harness exited {proc.returncode}"] if proc.returncode != 0 else []) + (
            output_problems(step, original, text) if produced.exists() else [f"no /output/{cfg['output']} written"])
        if not problems:
            break
    if text:
        (out_dir / cfg["output"]).write_text(text, encoding="utf-8")
    tb.keep_session(work / "session", out_dir / "session.jsonl")
    shutil.rmtree(work, ignore_errors=True)
    res = {"model": model["name"], "item": item["item_key"], "dataset": item["dataset"], "step": step,
           "status": "error" if problems else "ok", "problem": "; ".join(problems) or None, "attempts": attempt,
           "duration_s": round(time.monotonic() - started, 1), "usage": usage,
           "cost_usd": run_eval.price_of(model, usage)}
    if step == "forward" and text:
        res.update({"hebrew_share": round(hebrew_share(text), 3), "latin_word_share": round(latin_word_share(text), 3)})
    res_path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def run_all(step: str, args):
    import loop
    import run_eval
    available = run_eval.prepare_keys()
    models = run_eval.select_models(args.models, available)
    its = items()
    if args.items:
        wanted = set(args.items.split(","))
        its = [i for i in its if i["item_key"] in wanted]
    jobs = [(m, it) for it in its for m in models]
    log(f"hebrew: {len(jobs)} {step} runs ({len(its)} items x {[m['name'] for m in models]})")
    image = run_eval._image()
    sems = run_eval._route_semaphores()
    throttle = loop.start_throttle(loop.load_loop_config())     # the proxy throttle only: no RAG server
    try:
        with ThreadPoolExecutor(max_workers=run_eval.ROUTE_CONCURRENCY["proxy"]) as pool:
            futs = {pool.submit(run_step, step, m, it, image, sems): (m, it) for m, it in jobs}
            for f in as_completed(futs):
                m, it = futs[f]
                try:
                    r = f.result()
                except Exception as e:  # noqa: BLE001 - one broken run must not stop the others
                    r = {"status": "error", "problem": str(e)}
                if r["status"] == "done":
                    continue
                log(f"hebrew: {step} {m['name']} {it['item_key']} -> {r['status']}"
                    + (f" (${r['cost_usd']:.3f}, {r['duration_s']:.0f}s)" if r.get("cost_usd") else "")
                    + (f" [{r.get('problem')}]" if r["status"] != "ok" else ""))
    finally:
        throttle.stop()
        run_eval.kill_model_containers([m["name"] for m in models])


def cmd_status(args):
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in OUT.glob("*/*/r1/**/result.json")]
    by = Counter((r["model"], r["step"], r["status"]) for r in rows)
    for key in sorted(by):
        log(f"hebrew: {key[0]:22} {key[1]:8} {key[2]:6} {by[key]}")
    log(f"hebrew: cost ${sum(r.get('cost_usd') or 0 for r in rows):.2f}")


# ---------------------------------------------------------------------- analysis

METRICS = (("bleu", "BLEU"), ("rouge_l", "ROUGE-L"), ("word_lev_sim", "word-Levenshtein similarity"))


def scores(original: str, recon: str) -> dict:
    return {"bleu": metrics.bleu(original, recon), "rouge_l": metrics.rouge_l(original, recon),
            "word_lev_sim": metrics.word_levenshtein_similarity(original, recon)}


def hebrew_rows(model_names) -> list:
    rows = []
    for p in sorted(OUT.glob("*/*/r1/back/result.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r["status"] != "ok" or r["model"] not in model_names:
            continue
        original = original_of(r["item"])
        recon = (p.parent / "reconstruction.md").read_text(encoding="utf-8")
        hebrew = (p.parent.parent / "translation.md").read_text(encoding="utf-8")
        rows.append({"method": "hebrew", "model": r["model"], "item": r["item"], "dataset": r["dataset"],
                     **scores(original, recon), "copied": latin_word_share(hebrew),
                     "original": original, "reconstruction": recon})
    return rows


def braincode_rows(model_names) -> list:
    import analyze
    data = analyze.load("main", exclude=[m["name"] for m in analyze_models() if m["name"] not in model_names])
    rows, seen = [], set()
    for r in sorted(data["runs"], key=lambda r: (r["model"], r["item"], r["run"])):
        if not r.get("back") or r["status"] not in ("success", "failed") or (r["model"], r["item"]) in seen:
            continue     # one round trip per model and item: its first usable run, as in expressivity.md
        seen.add((r["model"], r["item"]))
        rows.append({"method": "braincode", "model": r["model"], "item": r["item"], "dataset": r["dataset"],
                     "bleu": r["back"]["bleu"], "rouge_l": r["back"]["rouge_l"],
                     "word_lev_sim": r["back"]["word_lev_sim"], "copied": analyze.quoted_share(r),
                     "original": r["back"]["original"], "reconstruction": r["back"]["reconstruction"]})
    return rows


def analyze_models() -> list:
    return json.loads((EVAL_DIR / "models.json").read_text(encoding="utf-8"))["models"]


def paired(rows: list) -> list:
    """Rows of items that have both a BrainCode and a Hebrew round trip, per model."""
    have = Counter((r["model"], r["item"]) for r in rows)
    return [r for r in rows if have[(r["model"], r["item"])] == 2]


def comparison_table(rows: list, labels: dict, ds_labels: dict) -> tuple:
    """(markdown lines, csv rows) of the per-model, per-dataset means, BrainCode vs Hebrew."""
    import numpy as np
    lines = ["| model | dataset | pairs | " + " | ".join(f"{name} BC | {name} HE" for _, name in METRICS)
             + " | copied BC | copied HE |", "|---|---|---:|" + "---:|" * (2 * len(METRICS) + 2)]
    out = []
    for model in sorted({r["model"] for r in rows}):
        for ds in [None] + sorted({r["dataset"] for r in rows}):
            sub = [r for r in rows if r["model"] == model and (ds is None or r["dataset"] == ds)]
            if not sub:
                continue
            bc = [r for r in sub if r["method"] == "braincode"]
            he = [r for r in sub if r["method"] == "hebrew"]
            row = {"model": labels.get(model, model), "dataset": ds_labels.get(ds, ds) if ds else "all",
                   "pairs": len(bc)}
            for key, _ in METRICS + (("copied", "copied"),):
                row[f"{key}_braincode"] = float(np.mean([r[key] for r in bc]))
                row[f"{key}_hebrew"] = float(np.mean([r[key] for r in he]))
            out.append(row)
            lines.append(f"| {row['model']} | {row['dataset']} | {row['pairs']} | " + " | ".join(
                f"{row[k + '_braincode']:.3f} | **{row[k + '_hebrew']:.3f}**" for k, _ in METRICS)
                + f" | {row['copied_braincode']:.2f} | {row['copied_hebrew']:.2f} |")
    return lines, out


def tests_table(rows: list, labels: dict) -> list:
    import numpy as np
    from scipy.stats import wilcoxon
    lines = ["| model | metric | pairs | BrainCode mean | Hebrew mean | difference (BC − HE) | median per-item "
             "difference | items where BrainCode is higher | Wilcoxon p |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model in sorted({r["model"] for r in rows}):
        bc = {r["item"]: r for r in rows if r["model"] == model and r["method"] == "braincode"}
        he = {r["item"]: r for r in rows if r["model"] == model and r["method"] == "hebrew"}
        keys = sorted(bc)
        for key, name in METRICS:
            d = np.array([bc[k][key] - he[k][key] for k in keys])
            try:
                p = wilcoxon(d).pvalue if np.any(d != 0) else float("nan")
            except ValueError:
                p = float("nan")
            lines.append(f"| {labels.get(model, model)} | {name} | {len(keys)} | "
                         f"{np.mean([bc[k][key] for k in keys]):.3f} | {np.mean([he[k][key] for k in keys]):.3f} | "
                         f"{d.mean():+.3f} | {np.median(d):+.3f} | {int((d > 0).sum())} of {len(keys)} | "
                         f"{p:.2g} |")
    return lines


def corpus_lines(rows: list, labels: dict) -> list:
    lines = ["| model | pairs | corpus BLEU, BrainCode | corpus BLEU, Hebrew |", "|---|---:|---:|---:|"]
    for model in sorted({r["model"] for r in rows}):
        bc = [r for r in rows if r["model"] == model and r["method"] == "braincode"]
        he = [r for r in rows if r["model"] == model and r["method"] == "hebrew"]
        cb = [metrics.corpus_bleu([r["original"] for r in x], [r["reconstruction"] for r in x]) for x in (bc, he)]
        lines.append(f"| {labels.get(model, model)} | {len(bc)} | {cb[0]:.3f} | {cb[1]:.3f} |")
    return lines


def cmd_analyze(args):
    import csv
    import analyze
    names = set(MODELS.split(","))
    labels = {m["name"]: m["label"] for m in analyze_models()}
    he = hebrew_rows(names)
    if not he:
        sys.exit("hebrew: no finished Hebrew round trips yet")
    bc = braincode_rows(names)
    rows = paired(bc + he)
    n_he = Counter(r["model"] for r in he)
    n_bc = Counter(r["model"] for r in bc)
    RESULTS.mkdir(parents=True, exist_ok=True)
    comp, comp_csv = comparison_table(rows, labels, analyze.DS_LABELS)
    L = ["# Expressivity control: BrainCode vs Hebrew round trips", "",
         "**Every score is in [0, 1]; higher = the reconstruction is closer to the original = better.** Both round "
         "trips start from the same English item and end in English, written by the same model, which never sees "
         "the original on the way back. They are scored with the same functions (`metrics.py`): sentence BLEU, "
         "ROUGE-L F1 and word-Levenshtein similarity on lower-cased words, turn markers removed.", "",
         "- **BrainCode (BC):** English → BrainCode → English, with the swarm translator's setup (spec, glossary g19, "
         "RAG, kit). From `results/main/expressivity.md`.",
         "- **Hebrew (HE):** English → Hebrew → English, with the same container harness and proxy, but no BrainCode, "
         "no glossary and no RAG. Code, paths, URLs, identifiers, UI labels and names stay verbatim in the Hebrew.",
         "- **copied:** text carried over unchanged, which comes back almost for free. BC: share of the BrainCode's "
         "characters inside quoted literals. HE: share of the Hebrew's words still in Latin script.", "",
         "Hebrew is the reference for an ordinary, fluent translation: the gap BC − HE is how much more a BrainCode "
         "round trip loses than a translation into another natural language.", "",
         "Round trips available: " + ", ".join(f"{labels.get(m, m)}: BrainCode {n_bc[m]}, Hebrew {n_he[m]}"
                                                for m in sorted(names)) + ". Compared: items with both, per model.", "",
         "## Means per model and dataset (Hebrew in bold)", "", *comp, "",
         "## Paired comparison per item", "",
         "Difference = BrainCode − Hebrew on the same item (negative = BrainCode loses more). Wilcoxon signed-rank, "
         "two-sided, on the per-item differences.", "", *tests_table(rows, labels), "",
         "## Corpus BLEU", "", *corpus_lines(rows, labels)]
    (RESULTS / "expressivity_hebrew_control.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    fields = ["method", "model", "item", "dataset", "bleu", "rouge_l", "word_lev_sim", "copied"]
    with (RESULTS / "expressivity_hebrew_control.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    figure(rows, labels, analyze)
    log(f"hebrew: wrote {RESULTS / 'expressivity_hebrew_control.md'}")


def figure(rows: list, labels: dict, analyze):
    import matplotlib.pyplot as plt
    import numpy as np
    analyze.style()
    datasets = [d for d in analyze.DATASETS if any(r["dataset"] == d for r in rows)]
    models = sorted({r["model"] for r in rows})
    series = [(m, meth) for m in models for meth in ("braincode", "hebrew")]
    colors = {"braincode": "#2a78d6", "hebrew": "#eb6834"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    fig.suptitle("Round-trip similarity: BrainCode vs Hebrew (higher = more similar = better)", x=0.02, ha="left",
                 fontsize=12, fontweight="bold", color=analyze.INK)
    width = 0.8 / len(series)
    strong = {m["name"] for m in analyze_models() if m["tier"] == "strong"}
    for ax, (key, title) in zip(axes, METRICS):
        for i, (m, meth) in enumerate(series):
            vals = [np.mean([r[key] for r in rows if r["model"] == m and r["method"] == meth and r["dataset"] == d]
                            or [np.nan]) for d in datasets]
            c = colors[meth]
            ax.bar(np.arange(len(datasets)) + (i - (len(series) - 1) / 2) * width, vals, width=width * 0.92,
                   color=c if m in strong else analyze.SURFACE, edgecolor=c, linewidth=1.5,
                   hatch=None if m in strong else "///",
                   label=f"{'BrainCode' if meth == 'braincode' else 'Hebrew'}, {labels.get(m, m)}")
        ax.set_xticks(range(len(datasets)), [analyze.DS_LABELS[d] for d in datasets], rotation=30, ha="right")
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Score (0-1, higher = better)")
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.22))
    analyze.save(fig, RESULTS / "expressivity_hebrew_control.png")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["translate", "backtranslate", "status", "analyze"])
    p.add_argument("--models", default=MODELS)
    p.add_argument("--items", default="")
    args = p.parse_args(argv)
    if args.command == "translate":
        run_all("forward", args)
    elif args.command == "backtranslate":
        run_all("back", args)
    else:
        {"status": cmd_status, "analyze": cmd_analyze}[args.command](args)


if __name__ == "__main__":
    main()
