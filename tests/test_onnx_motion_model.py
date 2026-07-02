import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_onnx_motion_model.py"
GODOT_RUNNER = ROOT / "addons" / "motion_matching" / "onnx" / "mm_onnx_motion_model.gd"


def load_module():
    spec = importlib.util.spec_from_file_location("run_onnx_motion_model", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_query_matches_learned_motion_matching_contract():
    module = load_module()

    query = module.make_query_features()

    assert query.shape == (1, 1, 24)
    assert query.dtype == np.float32
    assert np.isfinite(query).all()


def test_query_json_requires_exactly_24_features(tmp_path):
    module = load_module()
    query_path = tmp_path / "query.json"
    query_path.write_text(json.dumps({"query": [0.0] * 23}), encoding="utf-8")

    with pytest.raises(ValueError, match="24"):
        module.load_query_features(query_path)


def test_godot_runner_exposes_onnx_motion_model_api():
    source = GODOT_RUNNER.read_text(encoding="utf-8")

    assert "class_name MMOnnxMotionModel" in source
    assert "func run_query(query: PackedFloat32Array) -> Dictionary:" in source
    assert "--model-dir" in source
    assert "--query-json" in source


def test_rollout_never_feeds_compressed_transition_into_stepper(monkeypatch):
    module = load_module()
    stepper_shapes = []

    class FakeIo:
        name = "x"

    class FakeSession:
        def __init__(self, kind):
            self.kind = kind

        def get_inputs(self):
            return [FakeIo()]

        def get_outputs(self):
            return [FakeIo()]

        def run(self, _output_names, feed):
            value = next(iter(feed.values()))
            if self.kind == "projector":
                assert value.shape == (1, 1, 24)
                return [np.zeros((1, 1, 56), dtype=np.float32)]
            if self.kind == "stepper":
                stepper_shapes.append(value.shape)
                assert value.shape == (1, 1, 56)
                return [np.ones((1, 1, 56), dtype=np.float32)]
            if self.kind == "decompressor":
                assert value.shape == (1, 1, 56)
                return [np.ones((1, 1, 1155), dtype=np.float32)]
            if self.kind == "compressor":
                assert value.shape == (1, 1, 2310)
                return [np.ones((1, 1, 32), dtype=np.float32)]
            raise AssertionError(self.kind)

    monkeypatch.setattr(module, "_session", lambda path: FakeSession(path.stem))

    result = module.run_models(Path("models"), module.make_query_features(), rollout_steps=3)

    assert stepper_shapes == [(1, 1, 56), (1, 1, 56), (1, 1, 56)]
    assert result["final_latent"].shape == (1, 1, 32)
