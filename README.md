# Motion Matching for Godot 4.4
[![Build GDExtension](https://github.com/GuilhermeGSousa/godot-motion-matching/actions/workflows/builds.yml/badge.svg?branch=master)](https://github.com/GuilhermeGSousa/godot-motion-matching/actions/workflows/builds.yml) [![Demo](https://img.shields.io/badge/Extension-Demo-blue)](https://github.com/GuilhermeGSousa/godot-motion-matching-demo)

![](https://github.com/GuilhermeGSousa/godot-motion-matching/blob/master/motion_matching_demo.gif)

Motion Matching is an animation technique that allows you to easily setup character movement animations from large amounts of unlabeled animation data, without requiring any blend trees or state machines.

This extension is fully integrated into Godot's `AnimationTree` system, and can be used in tandem with more traditional animation techniques.

### :gear: How it Works
Motion Matching uses a set of animations contained in an animation library to build a **pose database**, which contains **features** that describe different animation frames in different ways. At runtime, these **features** are periodically compared against what the character is doing, and the animation that best matches those **features** is played.

The only requirement for all this to work is to have animations with both root motion, and a root bone at the foot level.

You can find more on how to set all this up on the wiki [here!](https://github.com/GuilhermeGSousa/godot-motion-matching/wiki)

### :brain: ONNX Learned Motion Model

This fork adds an optional ONNX bridge for learned motion matching models. The bridge is intentionally external to the native GDExtension build: Godot calls a Python runner, and the runner executes a model stack with `onnxruntime`. This keeps the base extension build unchanged while still letting a Godot project query learned ONNX locomotion models at runtime.

Expected model files:

- `projector.onnx`: 24 query features to 56 latent features
- `stepper.onnx`: 56 latent features to the next 56 latent features
- `decompressor.onnx`: 56 latent features to 1155 pose features
- `compressor.onnx`: 2310 transition pose features to a 32 feature transition embedding

Install the Python dependencies:

```bash
python -m pip install -r requirements-onnx.txt
```

Run the model stack directly:

```bash
python tools/run_onnx_motion_model.py \
  --model-dir path/to/Learned-Motion-Matching/onnx \
  --output-dir out/onnx_motion
```

Use it from Godot:

```gdscript
var model := MMOnnxMotionModel.new(
	ProjectSettings.globalize_path("res://"),
	"path/to/Learned-Motion-Matching/onnx",
	ProjectSettings.globalize_path("user://onnx_motion")
)

var query := PackedFloat32Array()
for i in range(24):
	query.append(0.0)

var result := model.run_query(query)
if result.ok:
	print(result.summary)
```

`MMOnnxMotionModel` lives at `addons/motion_matching/onnx/mm_onnx_motion_model.gd`. The wrapper validates the 24-float query contract, writes a query JSON file, runs `tools/run_onnx_motion_model.py`, and returns the parsed ONNX summary to Godot.

Run the source tests:

```bash
python -m pytest tests/test_onnx_motion_model.py -q
```

Run the Godot smoke test by opening `tests/godot_onnx_smoke/project.godot`. The smoke scene reads these optional environment variables:

- `MM_ONNX_REPO_ROOT`
- `MM_ONNX_MODEL_DIR`
- `MM_ONNX_OUTPUT_DIR`

If `MM_ONNX_OUTPUT_DIR` is not set, the scene writes to `tmp/godot_onnx_smoke_output` inside this repo. The scene prints `MM_ONNX_SMOKE_RESULT ...` and exits with a non-zero code if the ONNX query fails.


### :raised_hands: Credits
I want to thank all the contributors that made this project possible!

[Fire](https://github.com/fire)
[GeorgeS](https://github.com/GeorgeS2019)
[Remi](https://github.com/Remi123)
[Roberts Kalnins](https://github.com/rkalnins)

### Sources

- [Road to Next Gen Animation - GDC Talk](https://www.gdcvault.com/play/1023280/Motion-Matching-and-The-Road)
- [Simon Clavet's implementation video](https://www.youtube.com/watch?v=jcpIrw38E-s&ab_channel=SimonClavet)
- [Orange Duck's Blog](https://theorangeduck.com/)
- [Remi's Motion Matching implementation](https://github.com/Remi123/MotionMatching)
- Demo data taken from [O3DE Motion Matching Implementation](https://github.com/o3de/o3de/tree/development/Gems/MotionMatching)

A motion matching implementation in Godot 4.4, implemented following [Dan Holden's article](https://www.theorangeduck.com/page/code-vs-data-driven-displacement).
