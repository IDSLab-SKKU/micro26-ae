#!/usr/bin/env python3
"""Compare the Table 6 cross-architecture results.

Both runs — `rtx_pro6000/emulate_hopper` and `h100/native` under
exp1_table6_cross_arch/ — produce a benchmark score and, with
log_samples, per-sample logprobs. `python3 scripts/compare.py` shows both at once for
the cross-architecture pair:

  rtx_pro6000/emulate_hopper  vs  h100/native
  (emulating Hopper's config on the Blackwell machine should match a real H100)

Produce the results first (see exp1_table6_cross_arch/README.md), copying
the H100's results into h100/native/results/ on this machine. No GPU is needed
here — this only reads JSON. Everything is checked for bit-exact equality: the
two runs should produce identical numbers, not merely close ones.

    python3 scripts/compare.py
    python3 scripts/compare.py --out cmp.md   # also write the table as markdown
"""
import argparse
import json
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parent.parent
BASE = "exp1_table6_cross_arch"

# The cross-architecture pair: emulating Hopper's config on the Blackwell
# machine should match a real H100. Each side is
# (heading, experiment path under artifact/, expected arch tag).
PAIR = {
    "left": ("emulate_hopper", f"{BASE}/rtx_pro6000/emulate_hopper", "blackwell"),
    "right": ("h100 native", f"{BASE}/h100/native", "hopper"),
}


def load_result(exp_relpath: str) -> dict | None:
    """Load an experiment's result JSON, newest if several were written."""
    results_dir = EXPERIMENTS / exp_relpath / "results"
    files = [p for p in results_dir.glob("*.json")
             if not p.stem.endswith("_samples")]
    if not files:
        return None
    best, best_ts = None, ""
    for p in files:
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        ts = data.get("timestamp", "")
        if best is None or ts >= best_ts:
            best, best_ts = data, ts
    return best


def load_samples(exp_relpath: str) -> dict | None:
    """Load an experiment's per-sample log (results/<run>_samples.json)."""
    results_dir = EXPERIMENTS / exp_relpath / "results"
    files = list(results_dir.glob("*_samples.json"))
    if not files:
        return None
    newest = max(files, key=lambda p: p.stat().st_mtime)
    try:
        return json.loads(newest.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def collect_metrics(result: dict) -> dict:
    """Flatten a result into {(task, metric): value} for numeric metrics."""
    out = {}
    for task in result.get("tasks", {}).values():
        name = task.get("task_name", "?")
        for metric, value in task.get("metrics", {}).items():
            if isinstance(value, (int, float)):
                out[(name, metric)] = float(value)
    return out


def sample_logprobs(sample: dict) -> list[float]:
    """The per-sample logprobs from an lm_eval sample record.

    filtered_resps holds one entry per request. Multiple-choice tasks store
    [logprob, is_greedy] per answer choice; perplexity tasks (rolling
    loglikelihood, e.g. wikitext) store the logprob as a bare float. Handle both.
    """
    out = []
    for fr in sample.get("filtered_resps", []):
        if isinstance(fr, (list, tuple)) and fr:
            out.append(float(fr[0]))
        elif isinstance(fr, (int, float)):
            out.append(float(fr))
    return out


def logprob_task_stats(left_list: list, right_list: list) -> tuple:
    """(total, identical, max_abs_diff, prompt_hash_mismatches) for one task."""
    left = {s.get("doc_id"): s for s in left_list}
    right = {s.get("doc_id"): s for s in right_list}
    ids = sorted(set(left) & set(right))
    total = identical = hash_mm = 0
    maxd = 0.0
    for i in ids:
        if left[i].get("prompt_hash") != right[i].get("prompt_hash"):
            hash_mm += 1
        for a, b in zip(sample_logprobs(left[i]), sample_logprobs(right[i])):
            total += 1
            if a == b:
                identical += 1
            else:
                maxd = max(maxd, abs(a - b))
    return total, identical, maxd, hash_mm


def fmt(value: float) -> str:
    """Number formatting with enough digits to show bit-exact agreement."""
    return f"{value:.6f}" if abs(value) < 100 else f"{value:.4f}"


def arch_of(result: dict) -> str:
    return (result.get("device") or {}).get("arch") or "unknown"


def compare(pair: dict) -> list[str]:
    """Print scores and per-sample logprobs side by side, task by task."""
    (lh, lp, lexp), (rh, rp, rexp) = pair["left"], pair["right"]
    md = ["# Table 6 — cross-architecture reproduction\n"]
    title = "Cross-architecture reproduction: emulate Hopper (F=13)  vs  native H100"
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")

    lres, rres = load_result(lp), load_result(rp)
    for side, res, exp, path in ((lh, lres, lexp, lp), (rh, rres, rexp, rp)):
        if res is None:
            print(f"  [missing] {side}: no result under {path}/results/")
            md.append(f"- **{side}**: missing — run `{path}`")
        else:
            got = arch_of(res)
            warn = "" if got == exp else f"  !! expected {exp}, got {got}"
            dev = (res.get("device") or {}).get("name", "?")
            print(f"  {side:16} {got:10} {dev}{warn}")
            md.append(f"- **{side}**: {got} — {dev}"
                      f"{' — ARCH MISMATCH' if warn else ''}")

    if lres is None or rres is None:
        print("\n  -> both results are needed to compare.")
        md.append("\n_Both results are needed to compare._\n")
        return md

    lsamp, rsamp = load_samples(lp), load_samples(rp)
    have_lp = lsamp is not None and rsamp is not None
    if not have_lp:
        which = lh if lsamp is None else rh
        print(f"\n  (no per-sample log for {which}; showing scores only)")

    lm, rm = collect_metrics(lres), collect_metrics(rres)
    tasks = sorted({t for (t, _) in lm} | {t for (t, _) in rm})

    header = (f"  {'task':15}{'metric':11}{'score':>10}   "
              f"{'logprobs (identical/total)':>26}   match")
    print(f"\n{header}\n  {'-' * (len(header) - 2)}")
    md.append("\n| task | metric | score | logprobs (identical/total) | match |")
    md.append("| --- | --- | ---: | ---: | :---: |")

    n_tasks = tasks_ok = 0
    lp_total = lp_ident = 0
    details = []
    for task in tasks:
        metrics = sorted({m for (t, m) in lm if t == task}
                         | {m for (t, m) in rm if t == task})
        score_ok = all(lm.get((task, m)) == rm.get((task, m)) for m in metrics)
        m0 = metrics[0] if metrics else "?"
        a0 = lm.get((task, m0))

        if have_lp and task in lsamp and task in rsamp:
            tot, ident, maxd, hmm = logprob_task_stats(lsamp[task], rsamp[task])
            lp_total += tot
            lp_ident += ident
            lp_ok = (ident == tot) and (hmm == 0)
            lp_cell = f"{ident:,} / {tot:,}"
        else:
            tot, lp_ok, lp_cell, maxd, hmm = None, True, "—", 0.0, 0

        task_ok = score_ok and lp_ok
        n_tasks += 1
        tasks_ok += task_ok
        mark = "✓" if task_ok else "✗"
        score_str = fmt(a0) if a0 is not None else "—"
        print(f"  {task:15}{m0:11}{score_str:>10}   {lp_cell:>26}   {mark}")
        md.append(f"| {task} | {m0} | {score_str} | {lp_cell} | {mark} |")

        if not score_ok:
            for m in metrics:
                a, b = lm.get((task, m)), rm.get((task, m))
                if a == b:
                    continue
                if a is None or b is None:
                    details.append(f"  ! {task}/{m}: present on only one side")
                else:
                    details.append(f"  ! {task}/{m}: {fmt(a)} vs {fmt(b)} "
                                   f"(Δ={a - b:+.3g})")
        if have_lp and tot is not None and not lp_ok:
            msg = (f"  ! {task}: {tot - ident:,}/{tot:,} logprobs differ, "
                   f"max|Δ|={maxd:.3g}")
            if hmm:
                msg += f", {hmm} prompt-hash mismatch"
            details.append(msg)

    print(f"  {'-' * (len(header) - 2)}")
    if have_lp:
        summ = (f"{tasks_ok}/{n_tasks} tasks identical  ·  "
                f"{lp_ident:,}/{lp_total:,} logprobs bit-identical")
    else:
        summ = (f"{tasks_ok}/{n_tasks} tasks with identical scores "
                f"(logprobs not checked)")
    print(f"  {summ}")
    md.append(f"\n**{summ}**\n")
    for d in details:
        print(d)
        md.append(f"- {d.strip().lstrip('! ')}")
    return md


def main():
    ap = argparse.ArgumentParser(
        description="Compare Table 6 cross-architecture results.")
    ap.add_argument("--out", type=Path, default=None,
                    help="also write the comparison as markdown to this file")
    args = ap.parse_args()

    md = compare(PAIR)
    if args.out:
        args.out.write_text("\n".join(md) + "\n")
        print(f"\nMarkdown written to: {args.out}")


if __name__ == "__main__":
    main()
