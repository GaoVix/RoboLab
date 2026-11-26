"""CPU-only checks that raw capture preserves the original packing call."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture_first_eval_query import attach_raw_capture


class RawCaptureTests(unittest.TestCase):
    def test_forwards_once_and_snapshots_before_preprocessing(self):
        raw = {"right_image": np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
               "wrist_image": np.zeros((3, 4, 3), dtype=np.uint8),
               "joint_position": np.arange(7, dtype=np.float32),
               "gripper_position": np.zeros(1, dtype=np.float32)}
        expected_raw = {key: value.copy() for key, value in raw.items()}
        calls, payload = [], {"original_payload": object()}

        class Client:
            def _pack_request(self, extracted, instruction):
                calls.append((extracted, instruction))
                # A preprocessor may mutate its input; the snapshot must still
                # represent the frame as it was before the original call.
                extracted["right_image"][:] = 255
                return payload

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            client = Client()
            state = attach_raw_capture(client, output)
            result = client._pack_request(raw, "original instruction")
            self.assertIs(result, payload)
            self.assertEqual(len(calls), 1)
            self.assertIs(calls[0][0], raw)
            self.assertEqual(calls[0][1], "original instruction")
            self.assertEqual(state["original_pack_calls"], 1)
            with np.load(output / "raw_observation.npz", allow_pickle=False) as saved:
                for key in expected_raw:
                    self.assertEqual(saved[key].tobytes(), expected_raw[key].tobytes())

    def test_second_frame_does_not_overwrite_the_first_capture(self):
        raw = {"right_image": np.zeros((2, 2, 3), dtype=np.uint8),
               "wrist_image": np.zeros((2, 2, 3), dtype=np.uint8),
               "joint_position": np.zeros(7, dtype=np.float32),
               "gripper_position": np.zeros(1, dtype=np.float32)}
        calls = []

        class Client:
            def _pack_request(self, extracted, instruction):
                calls.append(instruction)
                return {}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            client = Client()
            attach_raw_capture(client, output)
            client._pack_request(raw, "first")
            with self.assertRaisesRegex(RuntimeError, "only the first request"):
                client._pack_request(raw, "second")
            self.assertEqual(calls, ["first"])


if __name__ == "__main__":
    unittest.main()
