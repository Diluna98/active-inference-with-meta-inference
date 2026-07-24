"""Animation of live model reconstruction during RSSI navigation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from active_inference_navigation import GridNavigationEnvironment
from matplotlib.animation import FuncAnimation, PillowWriter

from .adaptive_navigation import (
    AdaptiveNavigationConfig,
    AdaptiveNavigationResult,
    run_adaptive_navigation_episode,
)


def save_adaptive_navigation_gif(
    output: str | Path,
    *,
    config: AdaptiveNavigationConfig | None = None,
    fps: int = 3,
) -> tuple[Path, AdaptiveNavigationResult]:
    """Run the adaptive episode and render its full diagnostic trace."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = config or AdaptiveNavigationConfig()
    environment = GridNavigationEnvironment(
        model_size=config.navigation.model_size,
        start=(487.5, 487.5),
        goal=(212.5, 312.5),
        random_seed=config.navigation.random_seed,
    )
    result = run_adaptive_navigation_episode(
        config=config,
        environment=environment,
    )

    figure = plt.figure(figsize=(10.4, 7.6), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 0.82))
    path_axis = figure.add_subplot(grid[0, 0])
    belief_axis = figure.add_subplot(grid[0, 1])
    metrics_axis = figure.add_subplot(grid[1, 0])
    latency_axis = metrics_axis.twinx()
    policy_axis = figure.add_subplot(grid[1, 1])
    figure.suptitle("Active inference with live meta-inference", fontsize=15)

    def update(frame: int):
        step = result.steps[frame]
        history = result.steps[: frame + 1]
        positions = np.asarray([item.position for item in history])

        path_axis.clear()
        path_axis.set_xlim(0.0, environment.workspace_size)
        path_axis.set_ylim(0.0, environment.workspace_size)
        path_axis.set_aspect("equal")
        ticks = np.linspace(0.0, environment.workspace_size, step.active_resolution + 1)
        path_axis.set_xticks(ticks, minor=True)
        path_axis.set_yticks(ticks, minor=True)
        path_axis.grid(which="minor", color="#c9d1db", linewidth=0.65)
        path_axis.plot(positions[:, 0], positions[:, 1], color="#1f77b4", linewidth=2.2)
        path_axis.scatter(
            *result.initial_position,
            marker="D",
            s=42,
            color="#2ca02c",
            edgecolor="white",
            zorder=3,
        )
        path_axis.scatter(
            *result.source_position,
            marker="*",
            s=155,
            color="#d62728",
            edgecolor="white",
            zorder=4,
        )
        path_axis.scatter(
            *positions[-1],
            s=65,
            color="#1f77b4",
            edgecolor="white",
            zorder=5,
        )
        switch_text = (
            "model rebuilt"
            if step.meta_decision is not None and step.meta_decision.switched
            else ("meta update" if step.meta_decision is not None else "task update")
        )
        path_axis.set_title(
            f"Navigation | active source model {step.active_resolution}×"
            f"{step.active_resolution} ({step.active_resolution**2} states)"
        )
        path_axis.set_xlabel(
            f"step {step.step} | distance {step.distance:.1f} | RSSI {step.rssi:.2f} | "
            f"{switch_text}"
        )

        belief_axis.clear()
        belief = step.source_belief.reshape(
            step.active_resolution,
            step.active_resolution,
        )
        belief_axis.imshow(
            belief,
            origin="lower",
            extent=(0.0, 500.0, 0.0, 500.0),
            cmap="magma",
            interpolation="nearest",
            vmin=0.0,
            vmax=max(float(belief.max()), 1e-8),
        )
        belief_axis.scatter(
            *result.source_position,
            marker="*",
            s=125,
            facecolor="none",
            edgecolor="#00ffff",
            linewidth=1.5,
        )
        belief_axis.set_title(
            f"Source posterior q(s) | shape {step.active_resolution}×"
            f"{step.active_resolution}"
        )
        belief_axis.set_xlim(0.0, 500.0)
        belief_axis.set_ylim(0.0, 500.0)
        belief_axis.set_aspect("equal")

        metrics_axis.clear()
        latency_axis.clear()
        indices = np.asarray([item.step for item in history])
        information = np.asarray([item.information_gain_proxy for item in history])
        surprise = np.asarray([item.prediction_error for item in history])
        latency = np.asarray([item.reference_latency_ms for item in history])
        metrics_axis.plot(indices, information, "o-", label="source information gain")
        metrics_axis.plot(indices, surprise, "o-", label="predictive surprise")
        metrics_axis.set_xlabel("task step")
        metrics_axis.set_ylabel("nats")
        metrics_axis.grid(color="#e1e5eb", linewidth=0.6)
        latency_axis.plot(
            indices,
            latency,
            "s--",
            color="#d62728",
            label="reference latency",
        )
        latency_axis.tick_params(axis="y", colors="#d62728")
        lines = metrics_axis.lines + latency_axis.lines
        metrics_axis.legend(
            lines,
            [line.get_label() for line in lines],
            loc="upper left",
            fontsize=8,
        )
        metrics_axis.set_title(
            f"Live task metrics | measured inference {step.measured_latency_ms:.2f} ms"
        )

        policy_axis.clear()
        labels = ("2×2", "5×5", "10×10", "20×20", "keep")
        latest_meta_step = next(
            (
                item
                for item in reversed(history)
                if item.meta_decision is not None
            ),
            None,
        )
        if latest_meta_step is None:
            posterior = np.zeros(5)
            title = "Meta policy posterior | first update pending"
        else:
            posterior = latest_meta_step.meta_decision.policy_posterior
            title = (
                f"Meta posterior at step {latest_meta_step.step} | selected "
                f"{latest_meta_step.meta_decision.selected_resolution}×"
                f"{latest_meta_step.meta_decision.selected_resolution}"
            )
        colors = [
            "#1f77b4" if index == int(np.argmax(posterior)) else "#b7c4d3"
            for index in range(5)
        ]
        policy_axis.bar(labels, posterior, color=colors)
        policy_axis.set_ylim(0.0, 1.05)
        policy_axis.set_ylabel("probability")
        policy_axis.set_title(title)
        policy_axis.grid(axis="y", color="#e1e5eb", linewidth=0.6)
        return []

    animation = FuncAnimation(
        figure,
        update,
        frames=len(result.steps),
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )
    animation.save(output, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(figure)
    return output, result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the live adaptive navigation and meta-inference GIF."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/results/adaptive_meta_navigation.gif"),
    )
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--fps", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = AdaptiveNavigationConfig(maximum_steps=args.steps)
    output, result = save_adaptive_navigation_gif(
        args.output,
        config=config,
        fps=args.fps,
    )
    print(f"saved animation: {output}")
    print(f"model resolutions: {result.resolutions.tolist()}")
    print(f"switch steps: {list(result.switch_steps)}")


if __name__ == "__main__":
    main()
