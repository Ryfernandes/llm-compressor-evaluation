from kfp import dsl


@dsl.component(
    base_image="python:3.12",
    packages_to_install=["pyyaml"]
)
def validate_yaml_config(
    yaml_filename: str,
    yamls_mount_path: str = "/yamls",
) -> None:
    """Validate that the YAML recipe file exists and is syntactically valid."""

    import yaml
    from pathlib import Path

    yaml_path = Path(yamls_mount_path) / yaml_filename
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML recipe file not found: {yaml_path}")

    print(f"Validating YAML recipe file: {yaml_path}")

    with open(yaml_path, "r") as f:
        try:
            recipe_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax in {yaml_filename}: {e}")

    if not recipe_data:
        raise ValueError(f"YAML recipe file is empty: {yaml_filename}")

    if not isinstance(recipe_data, dict):
        raise ValueError(f"YAML recipe must be a mapping of stages, got {type(recipe_data).__name__}")

    # Validate basic recipe structure: should have at least one stage with modifiers
    for stage_name, stage_content in recipe_data.items():
        if not isinstance(stage_content, dict):
            raise ValueError(
                f"Stage '{stage_name}' must be a mapping of modifier groups, "
                f"got {type(stage_content).__name__}"
            )
        print(f"Stage '{stage_name}' found with {len(stage_content)} modifier group(s)")

    print(f"\nYAML recipe validation passed: {yaml_filename}")
