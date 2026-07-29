import io

import numpy as np

from active_inference_meta.visualization import (
    AsyncMapBeliefVisualizer,
    AsyncTerminalBeliefVisualizer,
    BeliefSnapshot,
    render_belief_heatmap,
)


def test_terminal_heatmap_reports_map_goal_cell():
    belief = np.asarray([0.1, 0.2, 0.6, 0.1])

    rendered = render_belief_heatmap(BeliefSnapshot(3, 2, belief, 0.5, 1.5))

    assert "step=3" in rendered
    assert "resolution=2x2" in rendered
    assert "MAP cell=(0, 1)" in rendered
    assert "p=0.6000" in rendered


def test_async_visualizer_writes_outside_calling_thread():
    output = io.StringIO()
    visualizer = AsyncTerminalBeliefVisualizer(
        clear_terminal=False,
        output=output,
    )

    visualizer.submit(0, 2, np.full(4, 0.25), 0.5, 0.5)
    visualizer.close()

    assert "GOAL BELIEF" in output.getvalue()


def test_map_visualizer_writes_metric_arena_png(tmp_path):
    output = tmp_path / "goal_belief.png"
    visualizer = AsyncMapBeliefVisualizer(
        output,
        arena_width=7.0,
        arena_height=7.0,
    )

    visualizer.submit(0, 2, np.asarray([0.1, 0.2, 0.6, 0.1]), 0.175, 0.175)
    visualizer.close()

    assert output.exists()
    assert output.stat().st_size > 0
