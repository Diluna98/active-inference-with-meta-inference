import io

import numpy as np

from active_inference_meta.visualization import (
    AsyncTerminalBeliefVisualizer,
    BeliefSnapshot,
    render_belief_heatmap,
)


def test_terminal_heatmap_reports_map_goal_cell():
    belief = np.asarray([0.1, 0.2, 0.6, 0.1])

    rendered = render_belief_heatmap(BeliefSnapshot(3, 2, belief))

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

    visualizer.submit(0, 2, np.full(4, 0.25))
    visualizer.close()

    assert "GOAL BELIEF" in output.getvalue()
