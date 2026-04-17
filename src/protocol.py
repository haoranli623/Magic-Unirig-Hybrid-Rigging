from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def group_ids_by_character(sample_ids: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for sid in sample_ids:
        ch = sid.split("/")[0]
        out[ch].append(sid)
    return {k: sorted(v) for k, v in out.items()}


def build_train_recon_tasks(
    train_ids: list[str],
    limit: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    ids = list(train_ids)
    rng.shuffle(ids)
    if limit is not None and limit >= 0:
        ids = ids[:limit]

    return [
        {
            "task_type": "train_recon",
            "source_sample_id": sid,
            "target_sample_id": sid,
        }
        for sid in ids
    ]


def build_cross_motion_tasks(
    train_ids: list[str],
    val_ids: list[str],
    limit: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    train_map = group_ids_by_character(train_ids)
    val_map = group_ids_by_character(val_ids)

    chars = sorted([c for c in train_map.keys() if c in val_map and len(train_map[c]) > 0 and len(val_map[c]) > 0])
    rng.shuffle(chars)

    tasks: list[dict[str, Any]] = []
    for ch in chars:
        s = rng.choice(train_map[ch])
        t = rng.choice(val_map[ch])
        tasks.append(
            {
                "task_type": "cross_motion",
                "character_id": ch,
                "source_sample_id": str(s),
                "target_sample_id": str(t),
            }
        )

    if limit is not None and limit >= 0:
        tasks = tasks[:limit]
    return tasks


def build_cross_motion_tasks_all_val_targets(
    train_ids: list[str],
    val_ids: list[str],
    limit: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    train_map = group_ids_by_character(train_ids)

    tasks: list[dict[str, Any]] = []
    for tgt in sorted(val_ids):
        ch = tgt.split("/")[0]
        sources = train_map.get(ch, [])
        if not sources:
            continue
        src = str(rng.choice(sources))
        tasks.append(
            {
                "task_type": "cross_motion",
                "character_id": ch,
                "source_sample_id": src,
                "target_sample_id": str(tgt),
            }
        )

    if limit is not None and limit >= 0:
        tasks = tasks[:limit]
    return tasks


def protocol_note_text() -> str:
    return (
        "RigMo protocol text indicates sampling train/test motion pairs from the same object, "
        "using training sequences for baseline optimization and testing sequences for cross-motion evaluation. "
        "For this reproduction, cross-motion tasks are built as (source=train motion, target=val motion) per character."
    )
