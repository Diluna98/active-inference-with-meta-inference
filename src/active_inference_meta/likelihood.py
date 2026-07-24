"""Continuous meta-likelihood used for adaptive representation selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
from scipy.special import digamma, gammaln
from scipy.stats import t as student_t


def _softmax(values: np.ndarray, axis: int = 0, gamma: float = 1.0) -> np.ndarray:
    scaled = gamma * np.asarray(values, dtype=float)
    exponentials = np.exp(scaled - np.max(scaled))
    return exponentials / exponentials.sum(axis=axis, keepdims=True)


@dataclass(frozen=True)
class MetaLikelihoodParameters:
    """Learned conditional distributions for the meta-generative model."""

    mu_err: np.ndarray
    kappa_err: np.ndarray
    alpha_err: np.ndarray
    beta_err: np.ndarray
    mu_lat: np.ndarray
    sigma_lat: np.ndarray

    @classmethod
    def from_json(cls, path: str | Path | None = None) -> MetaLikelihoodParameters:
        if path is None:
            resource = files("active_inference_meta").joinpath(
                "data/meta_likelihood_parameters.json"
            )
            data = json.loads(resource.read_text(encoding="utf-8"))
        else:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        parameters = cls(
            **{name: np.asarray(data[name], dtype=float) for name in cls.__annotations__}
        )
        parameters.validate()
        return parameters

    def validate(self) -> None:
        expected_context_shape = (4, 4)
        expected_latency_shape = (4, 3)
        for name in ("mu_err", "kappa_err", "alpha_err", "beta_err"):
            if getattr(self, name).shape != expected_context_shape:
                raise ValueError(f"{name} must have shape {expected_context_shape}.")
        for name in ("mu_lat", "sigma_lat"):
            if getattr(self, name).shape != expected_latency_shape:
                raise ValueError(f"{name} must have shape {expected_latency_shape}.")
        if np.any(self.kappa_err <= 0) or np.any(self.alpha_err <= 0):
            raise ValueError("Student-t precision parameters must be positive.")
        if np.any(self.beta_err <= 0) or np.any(self.sigma_lat <= 0):
            raise ValueError("Likelihood scale parameters must be positive.")


class MetaLikelihood:
    """Likelihood over resolution, context, and compute-state factors.

    Hidden factors:
        0. representation resolution: 2x2, 5x5, 10x10, or 20x20
        1. environmental/task context: four learned complexity regimes
        2. compute availability: low, medium, or high

    Observation modalities:
        0. information-gain proxy
        1. prediction error
        2. task inference latency in milliseconds
        3. available CPU percentage
    """

    states_dim = (4, 4, 3)
    modality_dependencies = ((1,), (0, 1), (0, 2), (2,))
    epsilon = 1e-16

    def __init__(
        self,
        parameters: MetaLikelihoodParameters | None = None,
        *,
        grid_size: int = 100,
    ) -> None:
        self.parameters = parameters or MetaLikelihoodParameters.from_json()
        self.grid_size = int(grid_size)
        if self.grid_size < 2:
            raise ValueError("grid_size must be at least two.")

        self.mu_information = np.array([0.02, 0.1, 0.5, 3.0], dtype=float)
        self.sigma_information = np.array([0.02, 0.015, 0.08, 0.4], dtype=float)
        self.mu_cpu = np.array([20.0, 57.5, 87.5], dtype=float)
        self.sigma_cpu = np.array([8.0, 10.0, 8.0], dtype=float)
        self.log_preferences = self._build_preferences()

    def get_o_grid(self, modality: int, N_grid: int | None = None) -> np.ndarray:
        size = self.grid_size if N_grid is None else int(N_grid)
        limits = {
            0: (0.0, 2.0),
            1: (2.0, 10.0),
            2: (50.0, 9000.0),
            3: (0.0, 100.0),
        }
        if modality not in limits:
            raise ValueError(f"Unknown meta-observation modality: {modality}")
        lower, upper = limits[modality]
        return np.linspace(lower, upper, size)

    def _build_preferences(self) -> dict:
        size = self.grid_size
        complexity, prediction_error = np.meshgrid(
            np.arange(size, dtype=float),
            np.arange(size, dtype=float),
            indexing="ij",
        )
        normalized_complexity = complexity / max(size - 1, 1)
        normalized_error = prediction_error / max(size - 1, 1)
        context_gate = 1.0 / (
            1.0 + np.exp(-6.0 * (normalized_complexity - 0.45))
        )
        joint_utility = -(
            20.0 + 15.0 * context_gate
        ) * normalized_error
        joint_probability = _softmax(joint_utility.ravel()).reshape(size, size)

        latency_grid = self.get_o_grid(2)
        comfort_ms = 600.0
        deadline_ms = 800.0
        latency_linear = latency_grid / deadline_ms
        latency_excess = np.clip(
            (latency_grid - comfort_ms) / (deadline_ms - comfort_ms),
            0.0,
            None,
        )
        latency_utility = -(latency_linear + 2.0 * latency_excess**2)
        latency_probability = _softmax(latency_utility)

        return {
            0: np.log(self.epsilon),
            (0, 1): np.log(joint_probability + self.epsilon),
            2: np.log(latency_probability + self.epsilon),
            3: np.log(self.epsilon),
        }

    def parameter_information_gain(
        self,
        state_beliefs,
    ) -> float:
        """Return the Normal-Gamma information gain used by the paper agent."""

        resolution = np.asarray(state_beliefs[0], dtype=float)
        context = np.asarray(state_beliefs[1], dtype=float)
        responsibility = np.outer(resolution, context)
        parameters = self.parameters

        kappa_new = parameters.kappa_err + responsibility
        alpha_new = parameters.alpha_err + 0.5 * responsibility
        mu_new = (
            parameters.kappa_err * parameters.mu_err
            + responsibility * parameters.mu_err
        ) / kappa_new
        beta_new = parameters.beta_err.copy()

        epsilon = 1e-12
        term_mu = (
            0.5
            * (alpha_new / np.maximum(beta_new, epsilon))
            * parameters.kappa_err
            * (mu_new - parameters.mu_err) ** 2
        )
        term_kappa = 0.5 * (
            np.log(parameters.kappa_err / kappa_new)
            - parameters.kappa_err / kappa_new
            + 1.0
        )
        term_gamma = (
            parameters.alpha_err
            * np.log(beta_new / parameters.beta_err)
            - (gammaln(alpha_new) - gammaln(parameters.alpha_err))
            + (alpha_new - parameters.alpha_err) * digamma(alpha_new)
            - (beta_new - parameters.beta_err) * (alpha_new / beta_new)
        )
        divergence = np.clip(term_mu + term_kappa + term_gamma, 0.0, None)
        mean_information = 0.5 * np.log(
            (kappa_new + self.epsilon)
            / (parameters.kappa_err + self.epsilon)
        )
        return float(np.sum(responsibility * (mean_information + divergence)))

    def update_from_observation(
        self,
        observation,
        state_beliefs,
        learning_rate: float = 0.1,
    ) -> None:
        """Apply the paper's online prediction-error and latency updates."""

        values = np.asarray(observation, dtype=float)
        resolution = np.asarray(state_beliefs[0], dtype=float)
        context = np.asarray(state_beliefs[1], dtype=float)
        compute = np.asarray(state_beliefs[2], dtype=float)
        parameters = self.parameters

        error_responsibility = np.outer(resolution, context)
        kappa_old = parameters.kappa_err.copy()
        mu_old = parameters.mu_err.copy()
        kappa_new = kappa_old + error_responsibility
        mu_new = (
            kappa_old * mu_old
            + error_responsibility * values[1]
        ) / kappa_new
        alpha_new = parameters.alpha_err + 0.5 * error_responsibility
        beta_new = parameters.beta_err + (
            0.5
            * (kappa_old * error_responsibility / kappa_new)
            * (values[1] - mu_old) ** 2
        )
        parameters.kappa_err[...] = kappa_new
        parameters.mu_err[...] = mu_new
        parameters.alpha_err[...] = alpha_new
        parameters.beta_err[...] = beta_new

        latency_responsibility = np.outer(resolution, compute)
        latency_error = values[2] - parameters.mu_lat
        parameters.mu_lat[...] = np.clip(
            parameters.mu_lat
            + learning_rate * latency_responsibility * latency_error,
            50.0,
            9000.0,
        )
        variance = parameters.sigma_lat**2
        variance += (
            learning_rate
            * latency_responsibility
            * (latency_error**2 - variance)
        )
        parameters.sigma_lat[...] = np.sqrt(np.clip(variance, 1e-6, None))

    @staticmethod
    def _gaussian(observation, mean, sigma):
        standardized = (np.asarray(observation) - mean) / sigma
        return np.exp(-0.5 * standardized**2) / (sigma * np.pi)

    def likelihoods(self, observation: float, modality: int) -> np.ndarray:
        observation = float(observation)
        if modality == 0:
            return self._gaussian(
                observation,
                self.mu_information,
                self.sigma_information,
            )
        if modality == 1:
            nu = 2.0 * self.parameters.alpha_err
            scale = np.sqrt(
                self.parameters.beta_err
                * (self.parameters.kappa_err + 1.0)
                / (self.parameters.alpha_err * self.parameters.kappa_err)
            )
            return student_t.pdf(
                observation,
                df=nu,
                loc=self.parameters.mu_err,
                scale=scale,
            )
        if modality == 2:
            return self._gaussian(
                observation,
                self.parameters.mu_lat,
                self.parameters.sigma_lat,
            )
        if modality == 3:
            return self._gaussian(observation, self.mu_cpu, self.sigma_cpu)
        raise ValueError(f"Unknown meta-observation modality: {modality}")

    def likelihoods_grid_vec(
        self,
        observation_grid: np.ndarray,
        modality: int,
        state_samples,
    ) -> np.ndarray:
        grid = np.asarray(observation_grid, dtype=float)
        if modality == 0:
            context = np.asarray(state_samples, dtype=int)
            mean = self.mu_information[context]
            sigma = self.sigma_information[context]
            return self._gaussian(grid[None, :], mean[:, None], sigma[:, None])

        if modality == 1:
            resolution, context = (
                np.asarray(values, dtype=int) for values in state_samples
            )
            mean = self.parameters.mu_err[resolution, context]
            kappa = self.parameters.kappa_err[resolution, context]
            alpha = self.parameters.alpha_err[resolution, context]
            beta = self.parameters.beta_err[resolution, context]
            nu = 2.0 * alpha
            scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
            return student_t.pdf(
                grid[None, :],
                df=nu[:, None],
                loc=mean[:, None],
                scale=scale[:, None],
            )

        if modality == 2:
            resolution, compute = (
                np.asarray(values, dtype=int) for values in state_samples
            )
            mean = self.parameters.mu_lat[resolution, compute]
            sigma = self.parameters.sigma_lat[resolution, compute]
            return self._gaussian(grid[None, :], mean[:, None], sigma[:, None])

        if modality == 3:
            compute = np.asarray(state_samples, dtype=int)
            mean = self.mu_cpu[compute]
            sigma = self.sigma_cpu[compute]
            return self._gaussian(grid[None, :], mean[:, None], sigma[:, None])

        raise ValueError(f"Unknown meta-observation modality: {modality}")
