@tool
extends RefCounted
class_name MMOnnxMotionModel

const SCRIPT_PATH := "tools/run_onnx_motion_model.py"

var python_executable := "python"
var repo_root := ""
var model_dir := ""
var output_dir := ""
var rollout_steps := 20


func _init(p_repo_root := "", p_model_dir := "", p_output_dir := "") -> void:
	repo_root = p_repo_root
	model_dir = p_model_dir
	output_dir = p_output_dir


func run_query(query: PackedFloat32Array) -> Dictionary:
	if query.size() != 24:
		return {
			"ok": false,
			"exit_code": -1,
			"error": "ONNX motion matching query must contain exactly 24 floats.",
		}

	var resolved_output_dir := _resolved_output_dir()
	DirAccess.make_dir_recursive_absolute(resolved_output_dir)

	var query_path := resolved_output_dir.path_join("lmm_onnx_query.json")
	var query_file := FileAccess.open(query_path, FileAccess.WRITE)
	if query_file == null:
		return {
			"ok": false,
			"exit_code": -1,
			"error": "Could not write query JSON: %s" % query_path,
		}
	query_file.store_string(JSON.stringify({"query": Array(query)}))
	query_file.close()

	return run_validation(query_path)


func run_validation(query_json_path := "") -> Dictionary:
	var resolved_repo_root := _resolved_repo_root()
	var resolved_output_dir := _resolved_output_dir()
	var args: Array[String] = [
		resolved_repo_root.path_join(SCRIPT_PATH),
		"--model-dir",
		_resolved_model_dir(),
		"--output-dir",
		resolved_output_dir,
		"--rollout-steps",
		str(rollout_steps),
	]
	if not query_json_path.is_empty():
		args.append("--query-json")
		args.append(query_json_path)

	var output: Array = []
	var exit_code := OS.execute(python_executable, args, output, true, false)
	if exit_code != 0:
		return {
			"ok": false,
			"exit_code": exit_code,
			"output": output,
		}

	var summary_path := resolved_output_dir.path_join("lmm_onnx_summary.json")
	var summary_text := FileAccess.get_file_as_string(summary_path)
	var summary: Variant = JSON.parse_string(summary_text)
	if typeof(summary) != TYPE_DICTIONARY:
		return {
			"ok": false,
			"exit_code": exit_code,
			"output": output,
			"error": "Missing or invalid ONNX summary JSON.",
		}

	return {
		"ok": _summary_is_valid(summary),
		"exit_code": exit_code,
		"output": output,
		"summary_path": summary_path,
		"summary": summary,
	}


func _summary_is_valid(summary: Dictionary) -> bool:
	var checks: Dictionary = summary.get("checks", {})
	return checks.get("shape_contracts_match") == true and checks.get("all_outputs_finite") == true


func _resolved_repo_root() -> String:
	if not repo_root.is_empty():
		return repo_root.simplify_path()
	return ProjectSettings.globalize_path("res://").simplify_path()


func _resolved_model_dir() -> String:
	if not model_dir.is_empty():
		return model_dir.simplify_path()
	return _resolved_repo_root().path_join("../Learned-Motion-Matching/onnx").simplify_path()


func _resolved_output_dir() -> String:
	if not output_dir.is_empty():
		return output_dir.simplify_path()
	return ProjectSettings.globalize_path("user://onnx_motion_matching").simplify_path()
