"""Ad-hoc cohort monitor: polls the live coord and prints one compact line
per tick. Exits when the M5 lands `--m5-target` rounds or after `--max-polls`.

Run:
    python scripts/monitor_cohort.py --interval 60 --m5-target 2 --max-polls 45
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

COORD = "https://dllm.planetbass.de"


def _get(path: str):
    with urllib.request.urlopen(f"{COORD}{path}", timeout=20) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--m5-target", type=int, default=2)
    ap.add_argument("--max-polls", type=int, default=45)
    args = ap.parse_args()

    polls = 0
    m5_rounds = 0
    last_round = None
    while m5_rounds < args.m5_target and polls < args.max_polls:
        try:
            st = _get("/status")
            ws = _get("/workers")["workers"]
        except Exception as e:  # noqa: BLE001 - best-effort monitor
            print(f"[{time.strftime('%H:%M:%S')}] poll error: {e}", flush=True)
            polls += 1
            time.sleep(args.interval)
            continue

        now = time.time()
        parts = []
        for w in ws:
            gpu = "M5" if "Apple" in (w.get("gpu") or "") else "3060"
            last = w.get("last_seen_ts")
            seen = f"{int(now - last)}s" if last else "never"
            tps = int(w["last_tokens_per_sec"]) if w.get("last_tokens_per_sec") else 0
            parts.append(
                f"{gpu}#{w['worker_id']}:r={w['rounds_contributed']},"
                f"inner={w.get('inner_steps')},tok/s={tps},seen={seen}"
            )
            if gpu == "M5":
                m5_rounds = max(m5_rounds, w["rounds_contributed"])

        rnd = st.get("current_round")
        advanced = "  <-- ROUND ADVANCED" if rnd != last_round and last_round is not None else ""
        last_round = rnd
        nsub = st.get("n_submitted")
        wsz = st.get("world_size")
        ros = int(st.get("round_open_seconds") or 0)
        vloss = st.get("last_val_loss")
        vloss_s = f"{vloss:.3f}" if isinstance(vloss, (int, float)) else "?"
        print(
            f"[{time.strftime('%H:%M:%S')}] round={rnd} ws={wsz} sub={nsub} "
            f"open={ros}s val={vloss_s} | " + " | ".join(parts) + advanced,
            flush=True,
        )
        polls += 1
        if m5_rounds >= args.m5_target:
            break
        time.sleep(args.interval)

    print(f"=== DONE: M5 contributed {m5_rounds} round(s) over {polls} poll(s) ===", flush=True)


if __name__ == "__main__":
    main()
