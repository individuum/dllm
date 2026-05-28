"""History-persistence mixin for the coordinator.

The dashboard chart at GET / is backed by a ring buffer that is mirrored to an
append-only NDJSON log (`<checkpoint_dir>/history.jsonl`) so it survives coord
restarts, plus a backfill path that reconstructs sparse history from each
checkpoint's `meta.json`. Method bodies moved verbatim from `server.py`.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .server import CoordinatorState

log = logging.getLogger("dllm.coord")


class HistoryMixin:
    """Append / load / backfill of the dashboard history ring buffer."""

    def _append_history(self: CoordinatorState, entry: dict) -> None:
        """Append to the in-memory deque AND the on-disk NDJSON log."""
        self.history.append(entry)
        if self._history_path is not None:
            try:
                self._history_path.parent.mkdir(parents=True, exist_ok=True)
                with self._history_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            except OSError as e:
                log.warning("could not persist history entry: %s", e)

    def _load_history_from_disk(self: CoordinatorState) -> None:
        """Load history.jsonl into the deque on startup (caps at maxlen)."""
        if self._history_path is None or not self._history_path.exists():
            return
        entries: list[dict] = []
        try:
            with self._history_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # tolerate partial last write
        except OSError as e:
            log.warning("could not load history.jsonl: %s", e)
            return
        # Deque maxlen drops oldest automatically
        for entry in entries[-self.history.maxlen :]:
            self.history.append(entry)
        log.info("loaded %d history entries from %s", len(self.history), self._history_path)

    def _backfill_history_from_checkpoint_metas(self: CoordinatorState) -> None:
        """Reconstruct sparse history from each ckpt_*/meta.json (one per
        checkpoint-every rounds). First-time enable: fills in everything that
        happened before history persistence existed.
        """
        if self.checkpoint_dir is None or not self.checkpoint_dir.exists():
            return
        existing_rounds = {h.get("round") for h in self.history}
        added = 0
        for ckpt_dir in sorted(self.checkpoint_dir.iterdir()):
            if not ckpt_dir.is_dir() or not ckpt_dir.name.startswith("ckpt_"):
                continue
            meta_file = ckpt_dir / "meta.json"
            if not meta_file.exists():
                continue
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            round_no = meta.get("round")
            if round_no in existing_rounds:
                continue
            self.history.append(
                {
                    "round": round_no,
                    "val_loss": meta.get("last_val_loss"),
                    "flops_total": float(meta.get("flops_total", 0.0)),
                    "ts": float(meta.get("ts", 0.0)),
                }
            )
            existing_rounds.add(round_no)
            added += 1
        if added:
            # Re-sort by round so the chart draws cleanly
            sorted_hist = sorted(self.history, key=lambda h: (h.get("round") or 0))
            self.history.clear()
            self.history.extend(sorted_hist)
            log.info(
                "backfilled %d history entries from %d checkpoint meta files",
                added,
                added,
            )
