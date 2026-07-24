# Active Inference with Meta-Inference

[![CI](https://github.com/Diluna98/active-inference-with-meta-inference/actions/workflows/ci.yml/badge.svg)](https://github.com/Diluna98/active-inference-with-meta-inference/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A hierarchical active-inference implementation that selects the computational
representation used by a task-level agent. The meta-level agent balances task
uncertainty and prediction quality against inference latency and available
compute.

![Live navigation with meta-inference rebuilding the source model](docs/results/adaptive_meta_navigation.gif)

The animation runs the continuous RSSI task from
[Active-Inference Navigation Agent](https://github.com/Diluna98/active-inference-navigation-agent)
using the task and meta-inference procedure used for the paper experiments.
The task agent begins with a `2×2` source representation. Meta-inference
selects resolutions from live task evidence. Every accepted switch rebuilds the
PyAIF navigation model, changes the source-state dimension, and remaps all deep
temporal source beliefs into the new grid. The trajectory therefore continues
without discarding accumulated source belief.

The implementation uses
[PyAIF](https://github.com/Diluna98/python_active_inference) for state and
policy inference.

## Architecture

```text
task observations
      │
      ▼
task-level active-inference agent
      │
      ├── information-gain proxy
      ├── prediction error
      ├── inference latency
      └── CPU availability
                │
                ▼
      meta-inference agent
                │
                ▼
  select 2×2, 5×5, 10×10, 20×20,
          or keep the current model
```

The meta-generative model contains three hidden-state factors:

```text
representation ∈ {2×2, 5×5, 10×10, 20×20}
task context   ∈ {context 0, context 1, context 2, context 3}
compute state  ∈ {low, medium, high}
```

Its four continuous observation modalities are:

```text
[information-gain proxy, prediction error, latency in ms, CPU availability]
```

Actions `0–3` select one of the four representations. Action `4` keeps the
current representation. Policy inference evaluates the expected free energy of
these alternatives using learned prediction-error and latency likelihoods.

## Installation

```bash
git clone https://github.com/Diluna98/active-inference-with-meta-inference.git
cd active-inference-with-meta-inference
python -m venv .venv
python -m pip install -e .
```

Install development tools with:

```bash
python -m pip install -e ".[dev]"
```

## Run the example

Run meta-inference over the packaged task trace:

```bash
active-inference-meta
```

Save the complete posterior, expected-free-energy, risk, and ambiguity traces:

```bash
active-inference-meta --output meta_decisions.json
```

Use a custom JSON observation trace:

```bash
active-inference-meta \
  --trace observations.json \
  --initial-resolution 5
```

Each trace record has this structure:

```json
{
  "information_gain_proxy": 0.05,
  "prediction_error": 2.4,
  "inference_latency_ms": 80.0,
  "cpu_availability": 87.5
}
```

## Run adaptive RSSI navigation

Run the real task/meta-agent loop:

```bash
active-inference-meta-navigation
```

Recreate the README animation:

```bash
active-inference-meta-navigation-gif
```

The integrated example preserves the paper setup:

- Three-step deep temporal task inference with 10 message-passing iterations
- 25 cardinal movement policies and 500 policy samples
- The master-grid RSSI Fisher-information proxy
- The unweighted, policy-averaged binned RSSI surprise
- Raw wall-clock latency measured around task state inference only
- CPU availability computed from the paper's resolution-specific latency baselines
- Meta-inference every three task updates
- Online learning of the meta-level prediction-error and latency parameters

There is no artificial latency multiplier. The CPU observation is
`clip(100 * baseline_latency[resolution] / measured_latency, 0, 100)`, as in
the experiment code. Absolute latency and therefore the inferred compute state
depend on the machine and software version. A real deployment should measure
new baselines and retrain or adapt the latency likelihood for its target
computer.

If meta-inference selects a different representation, that task action is not
executed. The model is rebuilt first, all policy- and time-dependent source
beliefs are remapped, and task inference resumes on the next update.

## Real-sensor deployment boundary

The task observation and likelihood model are the domain-specific boundary.
The simulation currently supplies `[x, y, RSSI]` and uses the corresponding
continuous Gaussian task likelihood. A robot can instead supply task
observations from its localization and radio sensors and replace or calibrate
that likelihood model.

The following components should remain unchanged when reproducing the paper
method in a real setting:

- Deep temporal task inference and policy evaluation
- Information and prediction-error extraction from the task model
- The four-observation meta-generative model
- The three-step meta-inference schedule
- Belief-preserving representation switching

Sensor preprocessing, observation noise, likelihood parameters, and
hardware-specific latency baselines should be calibrated from real data.

## Python API

```python
from active_inference_meta import (
    MetaInferenceController,
    MetaObservation,
)

controller = MetaInferenceController()
decision = controller.infer(
    current_resolution=2,
    observation=MetaObservation(
        information_gain_proxy=0.05,
        prediction_error=2.4,
        inference_latency_ms=80.0,
        cpu_availability=87.5,
    ),
)

print(decision.selected_resolution)
print(decision.policy_posterior)
print(decision.expected_free_energy)
```

For sequential operation, call `infer()` repeatedly or use
`infer_sequence()`. The controller performs online updates of its learned
prediction-error and latency likelihood parameters.

The integrated navigation loop is also available as a Python API:

```python
from active_inference_meta import (
    AdaptiveNavigationConfig,
    run_adaptive_navigation_episode,
)

result = run_adaptive_navigation_episode(
    config=AdaptiveNavigationConfig(maximum_steps=18)
)

print(result.resolutions)
print(result.switch_steps)
```

## Learned meta-likelihood

The packaged parameters define:

- A Gaussian likelihood for the information-gain proxy conditioned on context
- A Student-t predictive likelihood for error conditioned on representation and context
- A Gaussian latency likelihood conditioned on representation and compute state
- A Gaussian CPU likelihood conditioned on compute state

The learned parameter tables are stored in
`src/active_inference_meta/data/meta_likelihood_parameters.json`.

## Development

```bash
ruff check .
pytest -q
python -m build
```

GitHub Actions runs linting, tests, and package builds on Python 3.10 and 3.11.

## License

This project is available under the [MIT License](LICENSE).

Copyright © 2026 Diluna A. Warnakulasuriya.
