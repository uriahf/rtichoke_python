import numpy as np

from rtichoke.processing.evaluation_semantics import (
    _SHARED_POPULATION,
    _build_evaluation_metadata,
)


def test_shared_outcomes_identify_models_in_one_population():
    probs = {
        "Model A": np.array([0.1, 0.9]),
        "Model B": np.array([0.2, 0.8]),
    }
    reals = np.array([0, 1])
    times = np.array([5.0, 10.0])

    metadata = _build_evaluation_metadata(probs, reals, times)

    assert set(metadata) == set(probs)
    assert metadata["Model A"].reference_group == "Model A"
    assert metadata["Model A"].evaluation == "Model A"
    assert metadata["Model A"].model == "Model A"
    assert metadata["Model A"].population == _SHARED_POPULATION
    assert metadata["Model B"].model == "Model B"
    assert metadata["Model B"].population == _SHARED_POPULATION


def test_keyed_outcomes_identify_populations_without_guessing_model_identity():
    probs = {
        "Population A": np.array([0.1, 0.9]),
        "Population B": np.array([0.2, 0.8]),
    }
    reals = {
        "Population A": np.array([0, 1]),
        "Population B": np.array([1, 0]),
    }
    times = {
        "Population A": np.array([5.0, 10.0]),
        "Population B": np.array([4.0, 9.0]),
    }

    metadata = _build_evaluation_metadata(probs, reals, times)

    assert metadata["Population A"].reference_group == "Population A"
    assert metadata["Population A"].evaluation == "Population A"
    assert metadata["Population A"].model is None
    assert metadata["Population A"].population == "Population A"
    assert metadata["Population B"].model is None
    assert metadata["Population B"].population == "Population B"


def test_paired_labels_remain_compatibility_evaluation_labels():
    pair = "Model A @ Population A"
    probs = {pair: np.array([0.1, 0.9])}
    reals = {pair: np.array([0, 1])}
    times = {pair: np.array([5.0, 10.0])}

    metadata = _build_evaluation_metadata(probs, reals, times)[pair]

    assert metadata.reference_group == pair
    assert metadata.evaluation == pair
    assert metadata.population == pair
    assert metadata.model is None
