from kfp import dsl

@dsl.component(
    base_image="python:3.12",
)
def cleanup_model_staging(
    session_id: str,
    tier2_mount_path: str = "/tier2",
) -> None:
    import shutil
    from pathlib import Path

    tier2_staging_dir = Path(tier2_mount_path) / "model-staging" / session_id

    if tier2_staging_dir.exists():
        print(f"[*] Cleaning up tier2 staging directory: {tier2_staging_dir}")
        shutil.rmtree(tier2_staging_dir)
        print(f"[+] Tier2 staging directory removed")
    else:
        print(f"[*] Tier2 staging directory does not exist (already cleaned up): {tier2_staging_dir}")