extends Node

const DEFAULT_OUTPUT_DIR := "tmp/godot_onnx_smoke_output"


func _ready() -> void:
	var repo_root := OS.get_environment("MM_ONNX_REPO_ROOT")
	if repo_root.is_empty():
		repo_root = ProjectSettings.globalize_path("res://../..")

	var model_dir := OS.get_environment("MM_ONNX_MODEL_DIR")
	var output_dir := OS.get_environment("MM_ONNX_OUTPUT_DIR")
	if output_dir.is_empty():
		output_dir = repo_root.path_join(DEFAULT_OUTPUT_DIR)

	var runner_script: Script = load(repo_root.path_join("addons/motion_matching/onnx/mm_onnx_motion_model.gd"))
	var runner: RefCounted = runner_script.new(repo_root, model_dir, output_dir)
	var query := PackedFloat32Array()
	for i in range(24):
		query.append(0.05 + sin(float(i) * 0.31) * 0.2)

	var result: Dictionary = runner.run_query(query)
	var summary: Dictionary = result.get("summary", {})
	print("MM_ONNX_SMOKE_RESULT " + JSON.stringify({
		"ok": result.get("ok", false),
		"exit_code": result.get("exit_code", -1),
		"summary_path": result.get("summary_path", ""),
		"pose_l2": summary.get("pose_l2", 0.0),
		"checks": summary.get("checks", {}),
	}))
	get_tree().quit(0 if result.get("ok", false) else 1)
