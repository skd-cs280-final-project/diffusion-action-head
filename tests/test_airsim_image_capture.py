"""AirSim image transport recovery without a simulator, GPU or control commands."""

import copy
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture(scope="module")
def capture():
    # The production logger imports the evaluation CLI at module load. Isolate
    # that unrelated global parsing while executing the real capture code.
    path = Path(__file__).resolve().parents[1] / "airsim_plugin" / "AirVLNSimulatorClientTool_AeroVLA.py"
    spec = importlib.util.spec_from_file_location("airsim_capture_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"utils.logger": SimpleNamespace(logger=logging.getLogger("image-capture-test"))}):
        spec.loader.exec_module(module)
    return module


CAMERAS = ["FrontCamera", "LeftCamera", "RightCamera", "RearCamera", "DownCamera"]


def responses(value=27):
    batch = []
    for index in range(len(CAMERAS)):
        batch.append(SimpleNamespace(height=2, width=3,
                                     image_data_uint8=np.full((2, 3, 3), value + index, dtype=np.uint8).tobytes()))
        batch.append(SimpleNamespace(height=2, width=3, image_data_float=[0., 10., 50., 100., 120., -5.]))
    return batch


class FakeClient:
    """Only image retrieval exists: control or scene mutation would fail the test."""
    def __init__(self, attempts):
        self.attempts = attempts
        self.requests = []

    def simGetImages(self, *, requests):
        self.requests.append(requests)
        response = self.attempts[min(len(self.requests) - 1, len(self.attempts) - 1)]
        if isinstance(response, Exception):
            raise response
        return response


def test_valid_capture_preserves_camera_order_rgb_and_depth_conversion(capture):
    batch = responses()
    images, depths = capture._decode_image_responses(batch, CAMERAS)
    assert len(images) == len(depths) == 5
    expected_depth = np.array([[0, 25, 127], [255, 255, 0]], dtype=np.uint8)
    for index in range(5):
        np.testing.assert_array_equal(images[index], np.full((2, 3, 3), 27 + index, dtype=np.uint8))
        np.testing.assert_array_equal(depths[index], expected_depth)


@pytest.mark.parametrize("kind", ["zero_rgb", "empty_rgb", "short_rgb", "zero_depth", "short_depth",
                                 "missing_response", "extra_response", "no_response"])
def test_invalid_camera_batch_is_rejected(capture, kind):
    batch = responses()
    if kind == "zero_rgb":
        batch[0].height = batch[0].width = 0
        batch[0].image_data_uint8 = b""
    elif kind == "empty_rgb":
        batch[0].image_data_uint8 = b""
    elif kind == "short_rgb":
        batch[8].image_data_uint8 = b"abc"
    elif kind == "zero_depth":
        batch[1].height = 0
        batch[1].image_data_float = []
    elif kind == "short_depth":
        batch[9].image_data_float = [1.0]
    elif kind == "missing_response":
        batch.pop()
    elif kind == "extra_response":
        batch.append(copy.deepcopy(batch[-1]))
    elif kind == "no_response":
        batch = None
    with pytest.raises(ValueError):
        capture._decode_image_responses(batch, CAMERAS)


def test_zero_sized_down_camera_retries_entire_batch_without_mixing_frames(capture):
    bad = responses(1)
    bad[8].height = bad[8].width = 0
    bad[8].image_data_uint8 = b""
    client = FakeClient([bad, responses(80)])
    with patch.object(capture.time, "sleep") as sleep:
        images, _ = capture._request_image_responses(client, CAMERAS)
    assert len(client.requests) == 2
    sleep.assert_called_once_with(1)
    for index, image in enumerate(images):
        np.testing.assert_array_equal(image, np.full((2, 3, 3), 80 + index, dtype=np.uint8))
    for request in client.requests:
        assert [item.camera_name for item in request] == [name for name in CAMERAS for _ in range(2)]
        assert [item.image_type for item in request] == [capture.airsim.ImageType.Scene,
                                                       capture.airsim.ImageType.DepthPerspective] * 5
        assert not any(item.compress for item in request)


def test_rpc_failure_retries_and_a_real_black_image_is_valid(capture):
    good = responses()
    good[0].image_data_uint8 = bytes(18)
    client = FakeClient([RuntimeError("temporary RPC error"), good])
    with patch.object(capture.time, "sleep") as sleep:
        images, _ = capture._request_image_responses(client, CAMERAS)
    assert len(client.requests) == 2
    sleep.assert_called_once_with(1)
    assert images[0].shape == (2, 3, 3) and not images[0].any()


def test_permanent_empty_frame_stops_after_eleven_attempts(capture):
    bad = responses()
    bad[0].height = bad[0].width = 0
    bad[0].image_data_uint8 = b""
    client = FakeClient([bad])
    with patch.object(capture.time, "sleep") as sleep:
        with pytest.raises(RuntimeError, match="after 11 attempts.*FrontCamera RGB") as error:
            capture._request_image_responses(client, CAMERAS)
    assert len(client.requests) == 11
    assert sleep.call_count == 10
    assert isinstance(error.value.__cause__, ValueError)


def test_public_capture_uses_validated_retries_for_each_client(capture):
    first = FakeClient([[], responses(30)])
    second = FakeClient([responses(90)])
    tool = capture.AirVLNSimulatorClientTool.__new__(capture.AirVLNSimulatorClientTool)
    tool.airsim_clients = [[first], [second]]
    with patch.object(capture.time, "sleep") as sleep:
        result = tool.getImageResponses()
    assert len(first.requests) == 2 and len(second.requests) == 1
    sleep.assert_called_once_with(1)
    assert [row[0][0][0][0, 0, 0] for row in result] == [30, 90]
