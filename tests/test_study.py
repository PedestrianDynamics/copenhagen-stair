import pandas as pd
import pytest

from kopenhagen.geometry import SCENARIOS, Layout, gaps, walkable_area
from kopenhagen.metrics import per_agent
from kopenhagen.simulate import RunConfig, run


def test_walkable_area_is_valid_and_has_hole_only_with_column():
    assert walkable_area(Layout()).is_valid
    assert len(walkable_area(Layout()).interiors) == 1
    assert len(walkable_area(Layout(has_column=False)).interiors) == 0


def test_gap_column_to_right_glass():
    lay = Layout(stair_width=2.0, column_diameter=0.5, column_offset=0.1)
    assert gaps(lay)["column_to_right_glass"] == pytest.approx(0.65)


def test_clearance_of_a_point_next_to_left_glass_end():
    lay = Layout()
    traj = pd.DataFrame({"t": [0.0, 1.0], "id": [1, 1], "x": [-0.7, -0.7], "y": [0.0, -0.5],
                         "vx": [0.0, 0.0], "vy": [0.0, 0.0]})
    ag = per_agent(lay, traj, radius=0.25)
    assert ag["min_clr_left"].iloc[0] == pytest.approx(0.05)


@pytest.mark.parametrize("model", ["sfm", "cfsm"])
def test_smoke_run_everyone_leaves(model):
    traj = run(SCENARIOS["no column"], RunConfig(model=model, n_agents=5, seed=1))
    assert traj["id"].nunique() == 5
    assert traj.attrs["stuck"] == 0
    assert traj["y"].min() < -3.0


def test_step_entry_angle_of_diagonal_walker():
    from kopenhagen.metrics import step_entry

    traj = pd.DataFrame({"t": [0.0, 0.1], "id": [1, 1], "x": [0.0, 0.1], "y": [0.05, -0.05],
                         "vx": [1.0, 1.0], "vy": [-1.0, -1.0]})
    entry = step_entry(traj)
    assert entry.loc[1, "entry_angle"] == pytest.approx(45.0)


def test_keep_right_lanes_separate_up_and_down_on_the_stair():
    cfg = RunConfig(model="cfsm", n_agents=10, inflow=0.8, up_share=0.5, seed=0, bypass_radius=0.8,
                    wide_spawn=True, keep_right=True, max_time=120)
    traj = run(SCENARIOS["as built"], cfg)
    assert traj.attrs["stuck"] == 0
    mid = traj[(traj["y"] < -1.0) & (traj["y"] > -3.0)]
    assert (mid[mid["group"] == "main"]["x"] < 0).mean() > 0.9
    assert (mid[mid["group"] == "up"]["x"] > 0).mean() > 0.9
