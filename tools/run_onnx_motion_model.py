#!/usr/bin/env python
"""Run a learned motion matching ONNX stack from a 24-float Godot query."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


MODEL_FILES = {
    "projector": "projector.onnx",
    "stepper": "stepper.onnx",
    "decompressor": "decompressor.onnx",
    "compressor": "compressor.onnx",
}


def make_query_features() -> np.ndarray:
    values = []
    for index in range(24):
        wave = np.sin(index * 0.37)
        counter = np.cos(index * 0.19)
        values.append(0.22 * wave + 0.08 * counter + 0.04)
    return np.asarray(values, dtype=np.float32).reshape(1, 1, 24)


def load_query_features(query_path: Path) -> np.ndarray:
    data = json.loads(query_path.read_text(encoding="utf-8"))
    values: Any = data.get("query", data) if isinstance(data, dict) else data
    array = np.asarray(values, dtype=np.float32)
    if array.shape == (24,):
        array = array.reshape(1, 1, 24)
    if array.shape != (1, 1, 24):
        raise ValueError(f"Expected query JSON to contain exactly 24 features, got shape {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("Query contains NaN or infinite values.")
    return array


def _session(model_path: Path):
    import onnxruntime as ort

    if not model_path.exists():
        raise FileNotFoundError(f"Missing ONNX model: {model_path}")
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def _run(session: Any, input_array: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session.run([output_name], {input_name: input_array.astype(np.float32)})[0]


def run_models(model_dir: Path, query: np.ndarray, rollout_steps: int) -> dict[str, Any]:
    sessions = {name: _session(model_dir / file_name) for name, file_name in MODEL_FILES.items()}

    stepper_latent = _run(sessions["projector"], query)
    first_latent = stepper_latent.copy()
    first_pose = _run(sessions["decompressor"], stepper_latent)
    pose = first_pose
    transition_latent = np.zeros((1, 1, 32), dtype=np.float32)

    for _step in range(rollout_steps):
        next_stepper_latent = _run(sessions["stepper"], stepper_latent)
        next_pose = _run(sessions["decompressor"], next_stepper_latent)
        transition = np.concatenate([pose, next_pose], axis=-1)
        transition_latent = _run(sessions["compressor"], transition)
        stepper_latent = next_stepper_latent
        pose = next_pose

    return {
        "query": query,
        "first_latent": first_latent,
        "final_stepper_latent": stepper_latent,
        "final_latent": transition_latent,
        "first_pose": first_pose,
        "final_pose": pose,
    }


def summarize(result: dict[str, np.ndarray], rollout_steps: int) -> dict[str, Any]:
    shapes = {name: list(value.shape) for name, value in result.items()}
    finite = all(bool(np.isfinite(value).all()) for value in result.values())
    return {
        "model": "learned_motion_matching_onnx",
        "rollout_steps": rollout_steps,
        "shapes": shapes,
        "query_l2": float(np.linalg.norm(result["query"])),
        "latent_l2": float(np.linalg.norm(result["final_latent"])),
        "pose_l2": float(np.linalg.norm(result["final_pose"])),
        "checks": {
            "shape_contracts_match": shapes["query"] == [1, 1, 24]
            and shapes["first_latent"] == [1, 1, 56]
            and shapes["final_stepper_latent"] == [1, 1, 56]
            and shapes["final_latent"] == [1, 1, 32]
            and shapes["first_pose"] == [1, 1, 1155]
            and shapes["final_pose"] == [1, 1, 1155],
            "all_outputs_finite": finite,
            "rollout_steps": rollout_steps,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path, help="Directory containing projector/stepper/decompressor/compressor ONNX files.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory where lmm_onnx_summary.json will be written.")
    parser.add_argument("--query-json", type=Path, help="Optional JSON file containing a 24-float query array.")
    parser.add_argument("--rollout-steps", type=int, default=20)
    args = parser.parse_args()

    query = load_query_features(args.query_json) if args.query_json else make_query_features()
    result = run_models(args.model_dir, query, args.rollout_steps)
    summary = summarize(result, args.rollout_steps)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "lmm_onnx_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"ok": summary["checks"]["shape_contracts_match"] and summary["checks"]["all_outputs_finite"], "summary_path": str(summary_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
