"""Active-inference controller for selecting task-model representations."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from PyAIF import (
    ActiveInfAgent,
    ContinuousLikelihood,
    GenerativeModel,
    ShallowInference,
    utils,
)

from .likelihood import MetaLikelihood, MetaLikelihoodParameters
from .observations import MetaDecision, MetaObservation


def _object_array(*arrays) -> np.ndarray:
    result = np.empty(len(arrays), dtype=object)
    for index, array in enumerate(arrays):
        result[index] = np.asarray(array, dtype=float)
    return result


def _meta_transitions() -> np.ndarray:
    resolution = np.zeros((4, 4, 5), dtype=float)
    for action in range(4):
        resolution[action, :, action] = 1.0
    resolution[:, :, 4] = np.eye(4)
    context = np.eye(4, dtype=float)[:, :, None]
    compute = np.eye(3, dtype=float)[:, :, None]
    return _object_array(resolution, context, compute)


@dataclass(frozen=True)
class MetaInferenceConfig:
    """Numerical configuration for the meta-inference agent."""

    resolutions: tuple[int, int, int, int] = (2, 5, 10, 20)
    message_passing_iterations: int = 30
    policy_samples: int = 500
    exact_state_limit: int = 4096
    random_seed: int = 0
    policy_workers: int = 1

    def __post_init__(self) -> None:
        if len(self.resolutions) != 4 or len(set(self.resolutions)) != 4:
            raise ValueError("Exactly four distinct resolutions are required.")
        if any(resolution < 1 for resolution in self.resolutions):
            raise ValueError("Resolutions must be positive.")


class MetaInferenceController:
    """Infer context and select a computational representation.

    Actions 0-3 select the corresponding configured representation. Action 4
    preserves the current representation.
    """

    KEEP_ACTION = 4

    def __init__(
        self,
        config: MetaInferenceConfig | None = None,
        *,
        parameters: MetaLikelihoodParameters | None = None,
    ) -> None:
        self.config = config or MetaInferenceConfig()
        self.likelihood_model = MetaLikelihood(parameters)
        controls_dim = (5, 1, 1)
        states_dim = self.likelihood_model.states_dim
        policies = utils.construct_policies(
            states_dim,
            controls_dim,
            policy_len=1,
            control_fac_idx=[0],
        )
        model = GenerativeModel(
            B=_meta_transitions(),
            D=_object_array(np.ones(4), np.ones(4), np.ones(3)),
            controls_dim=controls_dim,
            controllable_factors=[0],
            policies=policies,
        )
        likelihood = ContinuousLikelihood.from_model(
            self.likelihood_model,
            modality_dependencies=self.likelihood_model.modality_dependencies,
            grid_size=self.likelihood_model.grid_size,
            policy_samples=self.config.policy_samples,
            exact_state_limit=self.config.exact_state_limit,
            random_seed=self.config.random_seed,
        )
        self.agent = ActiveInfAgent(
            model=model,
            likelihood=likelihood,
            inference=ShallowInference(
                message_passing_iterations=self.config.message_passing_iterations,
                policy_workers=self.config.policy_workers,
            ),
            action_selection="deterministic",
        )
        self._context_prior = np.full(4, 0.25, dtype=float)
        self._compute_prior = np.full(3, 1.0 / 3.0, dtype=float)

    def reset(self) -> None:
        """Reset accumulated meta-level context and compute beliefs."""

        self._context_prior = np.full(4, 0.25, dtype=float)
        self._compute_prior = np.full(3, 1.0 / 3.0, dtype=float)
        self.agent.pD[1] = self._context_prior.copy()
        self.agent.pD[2] = self._compute_prior.copy()
        self.agent.reset()

    def infer(
        self,
        current_resolution: int,
        observation: MetaObservation,
    ) -> MetaDecision:
        """Infer latent context and select a representation action."""

        try:
            current_index = self.config.resolutions.index(int(current_resolution))
        except ValueError as error:
            raise ValueError(
                f"current_resolution must be one of {self.config.resolutions}."
            ) from error

        resolution_prior = np.zeros(4, dtype=float)
        resolution_prior[current_index] = 1.0
        self.agent.pD[0] = resolution_prior.copy()
        self.agent.pD[1] = self._context_prior.copy()
        self.agent.pD[2] = self._compute_prior.copy()
        self.agent.reset()
        self.agent.observe(observation.as_array())
        self.agent.infer_states()
        expected_free_energy, _ = self.agent.infer_policies()
        action = self.agent.select_action()
        if action is None:
            raise RuntimeError("The meta-inference agent did not select an action.")

        self._context_prior = np.asarray(self.agent.posteriors[1], dtype=float).copy()
        self._compute_prior = np.asarray(self.agent.posteriors[2], dtype=float).copy()
        action_index = int(action[0])
        selected_resolution = (
            current_resolution
            if action_index == self.KEEP_ACTION
            else self.config.resolutions[action_index]
        )
        return MetaDecision(
            action_index=action_index,
            selected_resolution=int(selected_resolution),
            switched=int(selected_resolution) != int(current_resolution),
            policy_posterior=np.asarray(self.agent.posterior_pi, dtype=float).copy(),
            expected_free_energy=np.asarray(expected_free_energy, dtype=float).copy(),
            risk=np.asarray(self.agent.risk, dtype=float).copy(),
            ambiguity=np.asarray(self.agent.ambiguity, dtype=float).copy(),
            state_posteriors=tuple(
                np.asarray(posterior, dtype=float).copy()
                for posterior in self.agent.posteriors
            ),
        )

    def infer_sequence(
        self,
        initial_resolution: int,
        observations: Iterable[MetaObservation],
    ) -> list[MetaDecision]:
        """Apply meta-inference sequentially, carrying the selected model."""

        current_resolution = int(initial_resolution)
        decisions = []
        for observation in observations:
            decision = self.infer(current_resolution, observation)
            decisions.append(decision)
            current_resolution = decision.selected_resolution
        return decisions

    def clone(self) -> MetaInferenceController:
        """Return an independent controller with the same learned parameters."""

        return MetaInferenceController(
            config=self.config,
            parameters=copy.deepcopy(self.likelihood_model.parameters),
        )
