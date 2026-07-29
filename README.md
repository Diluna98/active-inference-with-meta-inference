# Active Inference with Meta-Inference

[![CI](https://github.com/Diluna98/active-inference-with-meta-inference/actions/workflows/ci.yml/badge.svg)](https://github.com/Diluna98/active-inference-with-meta-inference/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A hierarchical active-inference implementation for adaptive computational
model selection. A task-level agent performs continuous-observation RSSI
navigation, while a meta-level agent selects the spatial resolution of the
task agent's source-location representation.

![Live navigation with meta-inference rebuilding the source model](docs/results/adaptive_meta_navigation.gif)

The animation runs the simulated continuous RSSI task from
[Active-Inference Navigation Agent](https://github.com/Diluna98/active-inference-navigation-agent).
The task agent begins with a `2×2` source representation. Meta-inference
selects among `2×2`, `5×5`, `10×10`, and `20×20` representations from
task-level evidence and computational measurements. Every accepted switch
rebuilds the PyAIF navigation model, changes the source-state dimension, and
remaps the full deep temporal source belief into the new grid. Navigation
therefore continues without resetting the accumulated source belief.

The implementation uses
[PyAIF](https://github.com/Diluna98/python_active_inference) for state and
policy inference.

## Architecture

```text
continuous task observations [x, y, RSSI]
      │
      ▼
task-level active-inference agent
      │
      ├── RSSI Fisher-information proxy
      ├── policy-averaged RSSI surprise
      ├── state-inference latency
      └── baseline-ratio compute availability
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
[RSSI information proxy, predictive surprise, latency in ms, compute availability]
```

Actions `0–3` select one of the four representations. Action `4` keeps the
current representation. Policy inference evaluates the expected free energy of
these alternatives using likelihoods over task context, representation,
latency, and compute state.

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

In this implementation, `information_gain_proxy` is the RSSI
Fisher-information proxy,
`prediction_error` is policy-averaged predictive surprise, and
`cpu_availability` is the baseline-ratio compute-availability proxy.

## Run the adaptive RSSI simulation

Run the integrated task/meta-agent simulation:

```bash
active-inference-meta-navigation
```

Recreate the README animation:

```bash
active-inference-meta-navigation-gif
```

## Adaptive navigation configuration

The integrated simulation uses:

- Three-step deep temporal task inference with 10 message-passing iterations
- 25 cardinal movement policies and 500 policy samples
- The master-grid RSSI Fisher-information proxy
- The unweighted, policy-averaged binned RSSI surprise
- Raw wall-clock latency measured around task state inference only
- Compute availability derived from bundled resolution-specific latency baselines
- Meta-inference every three task updates
- Online learning of the meta-level prediction-error and latency parameters

The RSSI information proxy is the likelihood-weighted spatial Fisher
sensitivity of the fixed `20×20` reference signal model. Predictive surprise is
the mean negative log probability of the observed RSSI bin across task
policies; it is not weighted by the task policy posterior.

There is no artificial latency multiplier. The compute-availability
observation is:

```text
clip(100 × baseline_latency[resolution] / measured_latency, 0, 100)
```

This is a baseline-ratio proxy, not a direct operating-system CPU-utilization
measurement. Absolute latency and therefore the inferred compute state depend
on the processor, operating-system load, Python runtime, and PyAIF version.
Runs on different systems may therefore produce different timing observations
and model-selection trajectories.

If meta-inference selects a different representation, that task action is not
executed. The model is rebuilt first, all policy- and time-dependent source
beliefs and policy predictions are preserved, and task inference resumes on
the next update.

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

## Run on the TurtleBot

The real-world command composes the hardware-independent adaptive policy with
the ROS observation and TurtleBot actuator adapters supplied by
`active-inference-navigation-agent`:

```text
/tb4_08/odom + /tb4_08/rssi
              |
              v
      adaptive navigation policy <--- processor availability (psutil)
              |
              v
       NavigationAction
              |
              v
      TurtleBotActionExecutor ---> /tb4_08/cmd_vel
```

ROS, TurtleBot, and `Twist` types are imported only by the composition module.
The task and meta-inference controllers remain usable without ROS.

On the Ubuntu computer that will run inference:

```bash
source /opt/ros/humble/setup.bash
cd ~/active-inference-with-meta-inference
source .venv/bin/activate
python -m pip install -e .

active-inference-meta-ros --planning-windows 20
```

The packaged configuration uses:

- a 7 m by 7 m arena and 20 by 20 movement grid;
- a 2 by 2 initial source representation;
- candidate representations 2 by 2, 5 by 5, 10 by 10, and 20 by 20;
- arena positive x from odometry negative y and arena positive y from
  odometry positive x;
- `/tb4_08/odom`, `/tb4_08/rssi`, and `/tb4_08/cmd_vel`;
- a five-sample RSSI median and five-sample CPU-availability median;
- a three-step temporal horizon for task-level Active Inference;
- meta-likelihood learning enabled with `learning_A: true`;
- the calibrated dBm likelihood and persistent `-54 dBm` termination rule.

External CPU availability is sampled continuously on the processor running the
command. The sampler subtracts CPU time used by this program and its live child
processes from system-wide busy time, so task inference, meta-inference, ROS
callbacks, and visualization are not treated as environmental CPU load. The
remaining external load is reported as availability (`100 - external CPU
utilization`) and remains separate from the task observation `[x, y, RSSI]`.
Replace `ComputeResourceSource` if availability later arrives from a different
computer or over a ROS topic.

The support of each continuous meta-observation modality is configured once:

```yaml
meta_observation_bounds:
  information_gain_proxy: [0.0, 0.2]
  prediction_error: [20.0, 35.0]
  inference_latency_ms: [50.0, 9000.0]
  cpu_availability: [0.0, 100.0]
```

These bounds define the likelihood observation grids. Every real
meta-observation is clipped to them immediately before state inference and
online likelihood learning, so an outlier cannot update the model outside its
configured support. Profiling still records the raw task and CPU measurements.

Learned prediction-error and latency likelihood parameters can persist across
robot trials:

```yaml
meta_learning:
  checkpoint: learned_meta_likelihood.yaml
  load_if_available: true
  save_on_exit: true
```

For an explicit main configuration file, a relative checkpoint path is
resolved relative to that configuration file. At startup, an existing
checkpoint replaces the `meta_likelihood` priors from the main YAML. If the
checkpoint does not exist, the main-YAML priors are used. Updated parameters
are atomically saved on runtime exit when meta-inference and likelihood
learning are enabled. Delete or rename the checkpoint to deliberately restart
from the priors.

During a ROS run, every scheduled meta-level action prints the current and
selected representation, including actions that keep the existing model:

```text
META model: step=3, current=2x2, selected=5x5, switched=True
```

Before each physical trial:

1. Clear the arena and ensure the emergency stop is accessible.
2. Place the robot at the centre of the configured start cell.
3. Establish repeatable odometry; with the packaged frame configuration,
   odometry `(0, 0)` at the start-cell centre maps to arena `(0.175, 0.175)`.
4. Verify that odometry and RSSI topics are publishing current values.
5. First test `stay`, positive x, and positive y using the navigation-agent
   actuator-test command.
6. Start adaptive navigation only after confirming the axis mapping and arena
   boundaries.

The executor performs closed-loop odometry motion without Nav2 or SLAM and
publishes a stop command on completion, errors, or shutdown.

## Collect fixed-resolution profiling data

Use this repository—not the navigation-agent repository—to collect the four
signals needed to fit the meta likelihood. Copy the packaged YAML to a separate
file such as `profile_2x2.yaml`, then change these sections:

```yaml
meta_inference:
  enabled: false
  initial_resolution: 10
  fixed_resolution: 2
  candidate_resolutions: [2, 5, 10, 20]
  meta_interval: 3

profiling:
  enabled: true
  output: meta_profile_2x2.csv
```

Run the physical experiment normally:

```bash
active-inference-meta-ros \
  --config profile_2x2.yaml \
  --planning-windows 20
```

With meta-inference disabled, the task agent stays at `fixed_resolution`;
resolution selection and meta-likelihood learning are not executed. Every task
inference step is immediately appended and flushed to CSV with:

```text
step,resolution,information_gain_proxy,prediction_error,inference_latency_ms,cpu_availability
```

Repeat with fixed resolutions `2`, `5`, `10`, and `20`, using a different
output filename for each run. Existing non-empty output files are appended, so
use a new filename when trials must remain separate.

## View the goal belief on the metric arena

Enable the asynchronous map renderer in the runtime YAML:

```yaml
visualization:
  enabled: true
  mode: map
  refresh_steps: 1
  output: goal_belief.png
```

The PNG is replaced atomically after each accepted frame. It shows the arena
dimensions from YAML (7 m by 7 m in the packaged configuration), with arena x
horizontal, arena y vertical, and `(0, 0)` at the lower-left. The heatmap is
the task-level source/goal posterior, the blue circle is the current robot
position, and the green star is the maximum-posterior goal cell.

The actual grid uses the active resolution, so it automatically changes when
meta-inference switches between 2x2, 5x5, 10x10, and 20x20 models.

Open `goal_belief.png` through an SSH-aware editor such as VS Code Remote SSH,
or copy it to the local computer while the run is active. A compact SSH
terminal view remains available with:

```yaml
visualization:
  enabled: true
  mode: terminal
  refresh_steps: 1
  clear_terminal: true
```

Belief snapshots are copied only after `infer_states()` has finished and are
rendered by a background worker. Queue submission never waits; if rendering
falls behind, an old frame is discarded. Rendering time is therefore not
included in `inference_latency_ms`. Rendering still consumes a small amount of
processor time and can consequently influence the separately measured CPU
availability. For calibration-quality CPU experiments, either disable the
visualization or use a larger `refresh_steps`, such as `5`.

## Learned meta-likelihood

The packaged parameters define:

- A Gaussian likelihood for the RSSI Fisher-information proxy conditioned on context
- A Student-t likelihood for predictive surprise conditioned on representation and context
- A Gaussian latency likelihood conditioned on representation and compute state
- A Gaussian compute-availability likelihood conditioned on compute state

All initial meta-likelihood priors are configurable under `meta_likelihood` in
`src/active_inference_meta/resources/meta_navigation.yaml`. This includes the
information, prediction-error, latency, and CPU means/scales. Meta-level
learning is configured separately under `meta_agent`:

```yaml
meta_agent:
  learning_A: true
  learning_rate: 0.1
  forgetting_rate: 0.95
```

Set `learning_A: false` for a fixed likelihood. The legacy packaged JSON
parameter file remains supported by `MetaLikelihoodParameters.from_json()` for
backward compatibility, but the TurtleBot composition uses the YAML priors.

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
