"""Compatibility boundary for the upgraded :mod:`stable_worldmodel` API.

LeFlow's algorithmic code intentionally remains independent of package API
renames. The current stable-worldmodel release no longer treats
``history_size`` or ``frame_skip`` as World/environment constructor options and
folds ``evaluate_from_dataset`` into ``World.evaluate``. In this project the
former two options were always one, so removing them preserves their intended
no-op semantics while the upgraded package supplies the environment implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import stable_worldmodel as swm


_REMOVED_WORLD_KEYS = ("history_size", "frame_skip")


def world_kwargs(config: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Convert a legacy world config into constructor kwargs for the new API.

    ``history_size=1`` and ``frame_skip=1`` were no-op fields in the previous
    configuration. Rejecting other values prevents silently changing a run
    whose old package behaviour cannot be reproduced by the new World API.
    """
    kwargs = dict(config)
    for key in _REMOVED_WORLD_KEYS:
        value = kwargs.pop(key, 1)
        if int(value) != 1:
            raise ValueError(
                f"stable-worldmodel no longer supports world.{key}={value}. "
                "LeFlow only supports the legacy no-op value 1 with this "
                "package version."
            )
    return kwargs


def create_world(config: Mapping[str, Any] | Any, *, image_shape: tuple[int, int]):
    """Construct a new-API World while preserving legacy no-op config fields."""
    return swm.World(**world_kwargs(config), image_shape=tuple(image_shape))


def evaluate_from_dataset(
    world,
    dataset,
    *,
    start_steps,
    goal_offset_steps: int,
    eval_budget: int,
    episodes_idx,
    callables=None,
    video_path: str | Path | None = None,
):
    """Run the legacy dataset-driven evaluation through ``World.evaluate``.

    Argument names changed upstream only; this preserves sampling, simulator
    restoration, goal construction, rollout budget, and artifact behaviour.
    """
    return world.evaluate(
        dataset=dataset,
        start_steps=list(start_steps),
        goal_offset=int(goal_offset_steps),
        eval_budget=int(eval_budget),
        episodes_idx=list(episodes_idx),
        callables=callables,
        video=video_path,
    )
