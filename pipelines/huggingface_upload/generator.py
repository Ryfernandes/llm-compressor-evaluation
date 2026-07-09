import kfp
from pathlib import Path
from pipeline import pipeline

PIPELINE_PREFIX = "huggingface-upload"
ARTIFACTS_DIR = Path("./artifacts")

def get_next_version() -> str:
    """Get the next version number by counting existing files in artifacts directory."""
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    existing_files = list(ARTIFACTS_DIR.glob(f"{PIPELINE_PREFIX}-*.yaml"))
    version = len(existing_files)

    while (ARTIFACTS_DIR / f"{PIPELINE_PREFIX}-{version:03d}.yaml").exists():
        version += 1

    return f"{version:03d}"

if __name__ == "__main__":
    version = get_next_version()
    output_path = ARTIFACTS_DIR / f"{PIPELINE_PREFIX}-{version}.yaml"

    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline, package_path=str(output_path)
    )
    print(f"Pipeline compiled to: {output_path}")
