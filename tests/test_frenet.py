import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import numpy as np
from core_engine.navigation.frenet import FrenetPath


def test_frenet_wrap_around():
    # 10x10 square loop track (Total length ~ 40m)
    waypoints = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 10.0],
        [0.0, 10.0]
    ])
    frenet = FrenetPath(waypoints)
    L = frenet.track_length
    assert L > 39.0 and L < 41.0

    # Test forward movement along regular segment
    s_prev = 5.0
    s_curr = 6.5
    ds = frenet.calculate_progress(s_curr, s_prev)
    assert np.isclose(ds, 1.5)

    # Test forward movement across start/finish line (s_prev = 39.5 -> s_curr = 0.5)
    s_prev_sf = L - 0.5
    s_curr_sf = 0.5
    ds_sf = frenet.calculate_progress(s_curr_sf, s_prev_sf)
    assert np.isclose(ds_sf, 1.0)

    # Test backward movement across start/finish line (s_prev = 0.5 -> s_curr = 39.5)
    s_prev_back = 0.5
    s_curr_back = L - 0.5
    ds_back = frenet.calculate_progress(s_curr_back, s_prev_back)
    assert np.isclose(ds_back, -1.0)

    print("ALL FRENET WRAP-AROUND TESTS PASSED!")


if __name__ == "__main__":
    test_frenet_wrap_around()
