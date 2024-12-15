"""Training-schedule configuration.

A schedule JSON describes the data a classifier cycles through during augmentation-supported
fine-tuning: a fixed ``test_set`` / ``valid_set``, a set of named ``datasets`` (real or
generated, each with an ``augment_num`` count of augmented copies and a ``proportion`` of the
data to use), optional ``combined_datasets`` built from those, and an ordered ``schedule`` of
stages (which dataset to train on, for how many epochs, whether early-stopping may skip it).

The schedule is expressed as dataclasses with explicit validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EvalSet:
    data: str
    split: str
    segment: str
    augment_num: int = 0


@dataclass
class DatasetSpec:
    name: str
    path: object            # str, or list[str] for combined datasets
    split: object
    segment: object
    gen_data: object        # bool, or list[bool] for combined
    augment_num: int
    proportion: float = 1.0
    combined: bool = False
    base_sets: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Stage:
    key: str
    epochs: int
    letskip: bool = False


@dataclass
class Schedule:
    test_set: EvalSet
    valid_set: EvalSet
    datasets: dict[str, DatasetSpec]
    stages: list[Stage]

    def resolved_stages(self) -> list[tuple[DatasetSpec, int, bool]]:
        return [(self.datasets[s.key], s.epochs, s.letskip) for s in self.stages]

    @property
    def data_paths(self) -> list[str]:
        return _flatten([self.test_set.data, self.valid_set.data,
                         *[d.path for d in self.datasets.values() if not d.combined]])

    @property
    def split_paths(self) -> list[str]:
        return _flatten([self.test_set.split, self.valid_set.split,
                         *[d.split for d in self.datasets.values() if not d.combined]])

    @property
    def segment_paths(self) -> list[str]:
        return _flatten([self.test_set.segment, self.valid_set.segment,
                         *[d.segment for d in self.datasets.values() if not d.combined]])


def _flatten(items) -> list[str]:
    out: list[str] = []
    for item in items:
        out.extend(item if isinstance(item, list) else [item])
    return out


def _eval_set(raw: dict) -> EvalSet:
    return EvalSet(data=raw["data"], split=raw["split"], segment=raw["segment"],
                   augment_num=int(raw.get("augment_num", 0)))


def from_dict(raw: dict) -> Schedule:
    """Validate and build a :class:`Schedule` from a parsed JSON dict."""
    try:
        datasets: dict[str, DatasetSpec] = {}
        for name, d in raw["datasets"].items():
            augment_num = int(d["augment_num"])
            if augment_num < 0:
                raise ValueError("augment_num must be non-negative")
            prop = float(d.get("proportion", 1.0))
            if not 0.0 <= prop <= 1.0:
                raise ValueError("proportion must be in [0, 1]")
            datasets[name] = DatasetSpec(
                name=name, path=d["path"], split=d["split"], segment=d["segment"],
                gen_data=bool(d["gen_data"]), augment_num=augment_num, proportion=prop,
            )

        for name, c in raw.get("combined_datasets", {}).items():
            base_sets = c["base_sets"]
            for b in base_sets:
                if b not in datasets:
                    raise ValueError(f"combined dataset '{name}' references unknown base set '{b}'")
            proportions = c["proportion"]
            for p in proportions:
                if not 0.0 <= p <= 1.0:
                    raise ValueError("proportion must be in [0, 1]")
            augment_num = int(c.get("augment_num", min(datasets[b].augment_num for b in base_sets)))
            if augment_num < 0:
                raise ValueError("augment_num must be non-negative")
            datasets[name] = DatasetSpec(
                name=name,
                path=[datasets[b].path for b in base_sets],
                split=[datasets[b].split for b in base_sets],
                segment=[datasets[b].segment for b in base_sets],
                gen_data=[datasets[b].gen_data for b in base_sets],
                augment_num=augment_num, proportion=proportions,
                combined=True, base_sets=list(base_sets),
            )

        stages = [Stage(key=s["key"], epochs=int(s["epochs"]), letskip=bool(s.get("letskip", False)))
                  for s in raw["schedule"]]
        for s in stages:
            if s.key not in datasets:
                raise ValueError(f"schedule references unknown dataset '{s.key}'")

        return Schedule(
            test_set=_eval_set(raw["test_set"]),
            valid_set=_eval_set(raw["valid_set"]),
            datasets=datasets, stages=stages,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid schedule: {exc}") from exc


def load_schedule(path: str | Path) -> Schedule:
    return from_dict(json.loads(Path(path).read_text()))
