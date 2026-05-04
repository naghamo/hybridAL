"""
monitor.py — Live progress monitor for run_experiments.sh running in background.

Usage:
    python monitor.py --log run.log
    python monitor.py --log run.log --total 450
    python monitor.py --log run.log --interval 5

Press Ctrl+C to exit.
"""

import argparse
import glob
import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────── #

STEPS = {
    "baselines":       "Step 1 — Baselines",
    "hybrid":          "Step 2 — HybridAL (Optuna)",
    "backbones":       "Step 3 — Backbones",
    "samplers":        "Step 4 — Samplers",
    "fixed_switch":    "Step 5 — Fixed Switch",
    "signal_ablation": "Step 6 — Signal Ablation",
    "quick":           "Quick test",
}


def count_json_files(base_dir):
    """Count completed experiment JSONs, broken down by step."""
    counts = {}
    for step in STEPS:
        step_dir = os.path.join(base_dir, step)
        n = len(glob.glob(os.path.join(step_dir, "**", "*.json"), recursive=True))
        if n > 0:
            counts[step] = n
    total = sum(counts.values())
    return total, counts


def get_last_result(base_dir):
    """Return (path, mtime, parsed_info) of the most recently written JSON."""
    latest_path, latest_mtime = None, 0
    for p in glob.glob(os.path.join(base_dir, "**", "*.json"), recursive=True):
        mt = os.path.getmtime(p)
        if mt > latest_mtime:
            latest_mtime = mt
            latest_path = p
    if not latest_path:
        return None, 0, {}
    info = _parse_result_json(latest_path)
    return latest_path, latest_mtime, info


def _parse_result_json(path):
    """Extract key info from a saved experiment JSON."""
    try:
        with open(path) as f:
            d = json.load(f)
        cfg = d.get("cfg", {})
        rounds = d.get("round_val_stats", [])
        last_val = rounds[-1] if rounds else {}
        test = d.get("final_test_stats", {})
        meta = d.get("strategy_metadata", {})
        return {
            "dataset":   cfg.get("data", "?"),
            "strategy":  cfg.get("strategy_class", "?"),
            "model":     cfg.get("model_name_or_path", "?").split("/")[-1],
            "sampler":   cfg.get("sampler_class", "?"),
            "seed":      cfg.get("seed", "?"),
            "signal":    cfg.get("strategy_kwargs", {}).get("signal", ""),
            "total_rounds": d.get("total_rounds", "?"),
            "val_f1":    last_val.get("f1_score"),
            "val_loss":  last_val.get("loss"),
            "val_acc":   last_val.get("accuracy"),
            "test_f1":   test.get("f1_score"),
            "test_acc":  test.get("accuracy"),
            "switched":  meta.get("switched"),
            "switch_round": meta.get("switch_round"),
            "labeled":   last_val.get("pool_stats", {}).get("labeled_count"),
        }
    except Exception:
        return {}


def parse_current_from_log(log_file):
    """
    Scan the log file (backwards) and extract what's currently happening:
    - Current experiment name
    - Current round / total rounds
    - Last val F1 seen
    - Current step
    - Any error lines
    """
    result = {
        "current_exp": None,
        "current_round": None,
        "total_rounds": None,
        "last_f1": None,
        "last_loss": None,
        "current_step": None,
        "current_epoch": None,
        "errors_recent": 0,
        "switched": None,
    }
    try:
        with open(log_file, "r", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return result

    # Read backwards for efficiency
    for line in reversed(lines[-500:]):
        line = line.strip()
        if not line:
            continue

        # Step header
        if "STEP" in line and "—" in line and result["current_step"] is None:
            result["current_step"] = line.strip().lstrip("=- ").rstrip("=- ")

        # Experiment header: [N/M] StrategyName_...
        m = re.search(r'\[(\d+)/(\d+)\]\s+(\S+)', line)
        if m and result["current_exp"] is None:
            result["current_exp"] = m.group(3)[:60]

        # Round info from tqdm: Round N | ...
        m = re.search(r'Round\s+(\d+)\s*\|.*?(\d+)\s*labeled', line)
        if m and result["current_round"] is None:
            result["current_round"] = int(m.group(1))

        # Val F1 from round completion line
        m = re.search(r'Val F1:\s*([\d.]+)', line)
        if m and result["last_f1"] is None:
            result["last_f1"] = float(m.group(1))

        m = re.search(r'Loss:\s*([\d.]+)', line)
        if m and result["last_loss"] is None:
            result["last_loss"] = float(m.group(1))

        # Rounds bar: Rounds: X%|... N/M
        m = re.search(r'Rounds.*?(\d+)/(\d+)', line)
        if m and result["total_rounds"] is None:
            result["current_round"] = int(m.group(1))
            result["total_rounds"]  = int(m.group(2))

        # Epoch info
        m = re.search(r'Epoch\s+(\d+)/(\d+)', line)
        if m and result["current_epoch"] is None:
            result["current_epoch"] = f"{m.group(1)}/{m.group(2)}"

        # Switch event
        if "SWITCHING to FineTune" in line and result["switched"] is None:
            m2 = re.search(r'round\s+(\d+)', line)
            result["switched"] = int(m2.group(1)) if m2 else True

        if result["current_exp"] and result["last_f1"] is not None:
            break

    # Count recent errors (last 200 lines)
    result["errors_recent"] = sum(1 for l in lines[-200:] if "ERROR" in l or "FAILED" in l)
    return result


def bar(done, total, width=36):
    if total <= 0:
        return "[" + "?" * width + "]"
    frac = min(done / total, 1.0)
    filled = int(width * frac)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def format_td(seconds):
    if seconds is None or seconds < 0 or seconds > 86400 * 14:
        return "unknown"
    return str(timedelta(seconds=int(seconds)))


def fmt_f1(v):
    return f"{v:.4f}" if v is not None else "—"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", default=["experiments"],
                        help="Base experiment directory.")
    parser.add_argument("--total", type=int, default=0,
                        help="Expected total experiments (enables ETA). "
                             "Step 1=90, Steps 1+2=180, all=~600")
    parser.add_argument("--log", default="run.log",
                        help="nohup log file (default: run.log).")
    parser.add_argument("--interval", type=float, default=8.0,
                        help="Refresh interval in seconds (default 8).")
    args = parser.parse_args()

    base_dir    = args.dirs[0]
    start_time  = time.time()
    start_total, _ = count_json_files(base_dir)
    prev_total  = start_total
    rate_samples = []   # (time, count) for rolling rate

    print(f"  Watching: {base_dir}  |  Log: {args.log}")
    print("  Press Ctrl+C to stop.\n")

    try:
        while True:
            now   = time.time()
            total_done, per_step = count_json_files(base_dir)
            elapsed = now - start_time
            new_since = total_done - start_total

            # Rolling rate (last 5 samples)
            rate_samples.append((now, total_done))
            if len(rate_samples) > 5:
                rate_samples.pop(0)
            if len(rate_samples) >= 2:
                dt = rate_samples[-1][0] - rate_samples[0][0]
                dn = rate_samples[-1][1] - rate_samples[0][1]
                rate = dn / dt if dt > 0 else 0  # exp/sec
            else:
                rate = new_since / elapsed if elapsed > 0 else 0

            # Last completed experiment
            last_path, last_mtime, last_info = get_last_result(base_dir)
            last_ago = f"{int(now - last_mtime)}s ago" if last_mtime else "—"

            # Current activity from log
            cur = parse_current_from_log(args.log)

            # ── render ────────────────────────────────────────────────────── #
            os.system("clear")
            ts = datetime.now().strftime("%H:%M:%S")

            print(f"\n  ╔{'═'*62}╗")
            print(f"  ║  HybridAL Experiment Monitor          {ts:>22}  ║")
            print(f"  ╚{'═'*62}╝")

            # Overall progress
            print()
            if args.total:
                remaining = args.total - total_done
                eta_sec   = (remaining / rate) if rate > 0 else None
                pct       = 100 * total_done / args.total
                print(f"  Overall  {bar(total_done, args.total)}  "
                      f"{total_done}/{args.total}  ({pct:.1f}%)")
                print(f"  ETA      {format_td(eta_sec)}   "
                      f"Rate: {rate*3600:.1f} exp/hr   Elapsed: {format_td(elapsed)}")
            else:
                print(f"  Completed: {total_done} experiments  (+{new_since} this session)")
                if rate > 0:
                    print(f"  Rate: {rate*3600:.1f} exp/hr   Elapsed: {format_td(elapsed)}")

            # Per-step breakdown
            print(f"\n  {'Step':<35} {'Done':>6}")
            print(f"  {'─'*42}")
            for key, label in STEPS.items():
                n = per_step.get(key, 0)
                if n > 0:
                    print(f"  {label:<35} {n:>6}")
            if not per_step:
                print("  (no results yet)")

            # Current activity
            print(f"\n  ── Currently running ──────────────────────────────")
            if cur["current_step"]:
                print(f"  Step    : {cur['current_step']}")
            if cur["current_exp"]:
                # Parse experiment name into readable parts
                exp = cur["current_exp"]
                print(f"  Exp     : {exp}")
            if cur["current_round"] is not None:
                rnd_str = str(cur["current_round"])
                if cur["total_rounds"]:
                    rnd_str += f"/{cur['total_rounds']}"
                    pct_r = 100 * cur["current_round"] / cur["total_rounds"]
                    rnd_str += f"  ({pct_r:.0f}%)"
                print(f"  Round   : {rnd_str}")
            if cur["current_epoch"]:
                print(f"  Epoch   : {cur['current_epoch']}")
            if cur["last_f1"] is not None:
                f1_str = f"{cur['last_f1']:.4f}"
                loss_str = f"{cur['last_loss']:.4f}" if cur["last_loss"] else "—"
                print(f"  Val F1  : {f1_str}   Loss: {loss_str}")
            if cur["switched"]:
                print(f"  Switched: FineTune at round {cur['switched']}")
            if cur["errors_recent"] > 0:
                print(f"  ⚠ Errors: {cur['errors_recent']} in last 200 log lines")

            # Last completed experiment
            if last_info:
                print(f"\n  ── Last completed ({last_ago}) ─────────────────────")
                print(f"  Dataset  : {last_info.get('dataset','?'):<15}  "
                      f"Strategy: {last_info.get('strategy','?')}")
                print(f"  Model    : {last_info.get('model','?'):<15}  "
                      f"Seed: {last_info.get('seed','?')}")
                sig = last_info.get("signal")
                if sig:
                    print(f"  Signal   : {sig}")
                rounds_done = last_info.get("total_rounds", "?")
                labeled_n   = last_info.get("labeled", "?")
                print(f"  Rounds   : {rounds_done}   Labeled: {labeled_n}")
                val_f1  = last_info.get("val_f1")
                test_f1 = last_info.get("test_f1")
                test_acc = last_info.get("test_acc")
                print(f"  Val F1   : {fmt_f1(val_f1)}   "
                      f"Test F1: {fmt_f1(test_f1)}   Test Acc: {fmt_f1(test_acc)}")
                sw = last_info.get("switch_round")
                if sw is not None:
                    print(f"  Switched : round {sw}")

            # Delta since last check
            delta = total_done - prev_count if (prev_count := prev_total) else 0
            prev_total = total_done
            if delta > 0:
                print(f"\n  +{delta} experiment(s) completed since last check")

            print()
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n  Monitor stopped.")


if __name__ == "__main__":
    main()
