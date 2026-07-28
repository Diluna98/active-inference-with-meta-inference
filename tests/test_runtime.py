from active_inference_meta import run_adaptive_simulation_runtime


def test_adaptive_policy_runs_through_generic_simulation_runtime():
    result = run_adaptive_simulation_runtime(planning_windows=1)

    assert len(result.observations) >= 1
    assert len(result.actions) <= 1
