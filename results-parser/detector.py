from adapters import LMEvalAdapter, LightEvalAdapter, ResultAdapter


def detect_framework(json_data: dict) -> ResultAdapter:
    """
    Determine if eval results are from LM-Eval or LightEval, using unique keys in both
    """
    if "config_general" in json_data:
        return LightEvalAdapter()
    
    if all(key in json_data for key in ["results", "configs", "config"]):
        return LMEvalAdapter()
    
    raise ValueError(
        "Unable to detect framework. JSON structure does not match LM-Eval or LightEval format."
    )
