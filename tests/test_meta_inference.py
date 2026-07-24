import numpy as np
import pytest

from active_inference_meta import (
    MetaInferenceController,
    MetaLikelihood,
    MetaLikelihoodParameters,
    MetaObservation,
    load_trace,
    run_meta_trace,
)


def test_packaged_likelihood_parameters_have_expected_shapes():
    parameters = MetaLikelihoodParameters.from_json()

    assert parameters.mu_err.shape == (4, 4)
    assert parameters.kappa_err.shape == (4, 4)
    assert parameters.alpha_err.shape == (4, 4)
    assert parameters.beta_err.shape == (4, 4)
    assert parameters.mu_lat.shape == (4, 3)
    assert parameters.sigma_lat.shape == (4, 3)


def test_meta_likelihood_modalities_are_finite_and_nonnegative():
    likelihood = MetaLikelihood()
    observations = (0.05, 2.4, 80.0, 87.5)
    expected_shapes = ((4,), (4, 4), (4, 3), (3,))

    for modality, (observation, shape) in enumerate(
        zip(observations, expected_shapes, strict=True)
    ):
        values = likelihood.likelihoods(observation, modality)
        assert values.shape == shape
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)


def test_meta_preferences_match_configured_utility_equations():
    likelihood = MetaLikelihood()
    size = likelihood.grid_size
    context, prediction_error = np.meshgrid(
        np.arange(size, dtype=float),
        np.arange(size, dtype=float),
        indexing="ij",
    )
    context /= size - 1
    prediction_error /= size - 1
    gate = 1.0 / (1.0 + np.exp(-6.0 * (context - 0.45)))
    joint_utility = -(20.0 + 15.0 * gate) * prediction_error
    joint_probability = np.exp(joint_utility - joint_utility.max())
    joint_probability /= joint_probability.sum()

    latency = likelihood.get_o_grid(2)
    latency_utility = -(
        latency / 800.0
        + 2.0 * np.clip((latency - 600.0) / 200.0, 0.0, None) ** 2
    )
    latency_probability = np.exp(latency_utility - latency_utility.max())
    latency_probability /= latency_probability.sum()

    assert np.allclose(np.exp(likelihood.log_preferences[(0, 1)]), joint_probability)
    assert np.allclose(np.exp(likelihood.log_preferences[2]), latency_probability)


def test_meta_online_learning_updates_error_and_latency_parameters():
    likelihood = MetaLikelihood()
    old_mu_error = likelihood.parameters.mu_err.copy()
    old_mu_latency = likelihood.parameters.mu_lat.copy()
    beliefs = (
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )

    likelihood.update_from_observation(
        np.array([0.05, 3.0, 100.0, 87.5]),
        beliefs,
    )

    assert likelihood.parameters.mu_err[0, 0] > old_mu_error[0, 0]
    assert likelihood.parameters.mu_lat[0, 2] != old_mu_latency[0, 2]
    assert np.array_equal(
        likelihood.parameters.mu_lat[:1, :2],
        old_mu_latency[:1, :2],
    )


def test_meta_controller_returns_a_normalized_policy_posterior():
    controller = MetaInferenceController()
    decision = controller.infer(
        2,
        MetaObservation(0.05, 2.4, 80.0, 87.5),
    )

    assert decision.action_index in range(5)
    assert decision.selected_resolution in (2, 5, 10, 20)
    assert decision.policy_posterior.shape == (5,)
    assert np.isclose(decision.policy_posterior.sum(), 1.0)
    assert np.all(np.isfinite(decision.expected_free_energy))
    assert np.all(np.isfinite(decision.risk))
    assert np.all(np.isfinite(decision.ambiguity))


def test_packaged_trace_runs_sequential_meta_inference():
    observations = load_trace()
    decisions = run_meta_trace(observations, initial_resolution=2)

    assert len(decisions) == len(observations)
    assert all(decision.selected_resolution in (2, 5, 10, 20) for decision in decisions)
    assert all(np.isclose(decision.policy_posterior.sum(), 1.0) for decision in decisions)
    assert any(decision.switched for decision in decisions)


def test_invalid_resolution_and_observation_are_rejected():
    controller = MetaInferenceController()
    observation = MetaObservation(0.05, 2.4, 80.0, 87.5)

    with pytest.raises(ValueError, match="current_resolution"):
        controller.infer(3, observation)
    with pytest.raises(ValueError, match="finite"):
        MetaObservation(np.nan, 2.4, 80.0, 87.5).as_array()
