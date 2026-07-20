import kfp
from pathlib import Path


def compile_pipeline(pipeline_func, prefix, artifacts_dir=None):
    if artifacts_dir is None:
        artifacts_dir = Path("./artifacts")
    else:
        artifacts_dir = Path(artifacts_dir)

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    existing = list(artifacts_dir.glob(f"{prefix}-*.yaml"))
    version = len(existing)

    while (artifacts_dir / f"{prefix}-{version:03d}.yaml").exists():
        version += 1

    output_path = artifacts_dir / f"{prefix}-{version:03d}.yaml"
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline_func, package_path=str(output_path)
    )
    print(f"Pipeline compiled to: {output_path}")
