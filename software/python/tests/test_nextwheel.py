"""Unit tests for NextWheel python module."""

import datetime
import os

import numpy as np

import nextwheel

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_read_dat():
    """Test the read_dat function."""
    data = nextwheel.read_dat(
        root_dir + "/nextwheel/sample.dat",
        root_dir + "/nextwheel/sample_calibration.json",
    )
    # No regression test
    assert np.isclose(data["Analog"].data["Force"].mean(), -1.9105986194916111)
    assert np.isclose(
        data["Analog"].data["Moment"].mean(), 0.46255237817195616
    )
    assert np.isclose(data["IMU"].data["Acc"].mean(), -2.9063238045895927)
    assert np.isclose(data["IMU"].data["Gyro"].mean(), -26.455109319914804)
    assert np.isclose(data["IMU"].data["Mag"].mean(), 29.033203171222343)
    assert np.isclose(data["Encoder"].data["Angle"].mean(), 51772.59336173234)


def test_time_as_unix_time():
    """Test that time is not relative to start, but as a complete unix time."""
    data = nextwheel.read_dat(
        root_dir + "/nextwheel/sample.dat",
        root_dir + "/nextwheel/sample_calibration.json",
    )
    assert datetime.datetime.fromtimestamp(
        data["Analog"].time[0]
    ) == datetime.datetime(2000, 12, 31, 19, 3, 21, 432000)
    assert datetime.datetime.fromtimestamp(
        data["Analog"].time[1]
    ) == datetime.datetime(2000, 12, 31, 19, 3, 21, 440142)
    assert datetime.datetime.fromtimestamp(
        data["IMU"].time[0]
    ) == datetime.datetime(2000, 12, 31, 19, 3, 21, 432196)
    assert datetime.datetime.fromtimestamp(
        data["Encoder"].time[0]
    ) == datetime.datetime(2000, 12, 31, 19, 3, 21, 434985)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
