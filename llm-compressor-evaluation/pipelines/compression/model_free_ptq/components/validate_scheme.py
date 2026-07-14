from kfp import dsl


@dsl.component(
    base_image="python:3.12",
    packages_to_install=["compressed-tensors"]
)
def validate_scheme(
    scheme: str,
) -> None:
    """Validate that the scheme string resolves to a known quantization preset."""

    from compressed_tensors.quantization import preset_name_to_scheme

    if not scheme or not scheme.strip():
        raise ValueError("scheme must be a non-empty string")

    scheme = scheme.strip()

    try:
        resolved = preset_name_to_scheme(scheme)
    except Exception as e:
        raise ValueError(
            f"Unknown quantization scheme '{scheme}'. "
            f"Examples of valid schemes: FP8_DYNAMIC, FP8_BLOCK, NVFP4A16, "
            f"MXFP4, MXFP8, W4A16, W8A8. Error: {e}"
        )

    print(f"Scheme '{scheme}' resolved successfully")
    print(f"Weights config: {resolved.weights}")
    if resolved.input_activations:
        print(f"Input activations config: {resolved.input_activations}")
