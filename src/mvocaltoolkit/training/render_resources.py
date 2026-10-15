"""Cheap validation before an ONNX preview job or bank workflow is submitted."""

def validate_render_resources(options: dict) -> None:
    threads = options.get("cpu_threads", 2)
    if type(threads) is not int or not 1 <= threads <= 32:
        raise ValueError("cpu_threads must be a whole number from 1 to 32")
    if type(options.get("cache_models", True)) is not bool:
        raise ValueError("cache_models must be a boolean")
