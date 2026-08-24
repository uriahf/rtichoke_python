"""Internal semantic metadata for model/population evaluations.

This module does not change public inputs or rendered grouping. It records the
semantic information that can be known from the existing input shapes while
preserving ``reference_group`` as the compatibility grouping key.
"""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Union

import numpy as np

_SHARED_POPULATION = "__shared_population__"


@dataclass(frozen=True)
class _EvaluationMetadata:
    """Semantic identity available for one compatibility reference group."""

    reference_group: str
    evaluation: str
    model: Optional[str]
    population: str


def _build_evaluation_metadata(
    probs: Mapping[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    times: Union[np.ndarray, Dict[str, np.ndarray]],
) -> dict[str, _EvaluationMetadata]:
    """Describe evaluations without changing existing grouping behavior.

    With shared outcome/time arrays, probability keys identify models evaluated
    in one shared population. With keyed outcome/time dictionaries, keys identify
    distinct evaluation populations, but the current API does not separately
    encode model identity; that field is therefore left unknown rather than
    inferred from a generic group label.
    """
    keyed_population = isinstance(reals, dict) or isinstance(times, dict)

    if keyed_population:
        return {
            group: _EvaluationMetadata(
                reference_group=group,
                evaluation=group,
                model=None,
                population=group,
            )
            for group in probs
        }

    return {
        group: _EvaluationMetadata(
            reference_group=group,
            evaluation=group,
            model=group,
            population=_SHARED_POPULATION,
        )
        for group in probs
    }


def _build_calibration_evaluation_metadata(
    probs: Mapping[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    times: Union[np.ndarray, Dict[str, np.ndarray]],
) -> dict[str, _EvaluationMetadata]:
    """Describe the semantic evaluations used by calibration production data.

    Calibration has one existing input shape that carries more information than
    the generic compatibility grouping: a single named model can provide one
    concatenated prediction vector while keyed outcome/time arrays identify
    multiple populations. Production calibration splits that vector by those
    population keys, so the model identity is explicitly known for every
    population and is retained here. Other input shapes keep the established
    generic model-known/model-unknown semantics.
    """
    population_keys: list[str] | None = None
    if isinstance(reals, dict):
        population_keys = list(reals)
    elif isinstance(times, dict):
        population_keys = list(times)

    if population_keys is not None and len(probs) == 1 and set(probs) != set(population_keys):
        model = next(iter(probs))
        return {
            population: _EvaluationMetadata(
                reference_group=population,
                evaluation=population,
                model=model,
                population=population,
            )
            for population in population_keys
        }

    return _build_evaluation_metadata(probs, reals, times)
