"""Continuous meta-likelihood used for adaptive representation selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
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
        joint_utility = -3.0 * normalized_complexity * normalized_error
        joint_probability = _softmax(joint_utility.ravel()).reshape(size, size)

        index = np.arange(size, dtype=float)
        latency_utility = -0.02 * index
        delay_index = max(1, round(0.05 * size))
        latency_utility[delay_index:] += -0.1 * (
            index[delay_index:] - delay_index
        )
        latency_probability = _softmax(latency_utility, gamma=0.1)

        return {
            0: np.log(self.epsilon),
            (0, 1): np.log(joint_probability + self.epsilon),
            2: np.log(latency_probability + self.epsilon),
            3: np.log(self.epsilon),
        }

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
