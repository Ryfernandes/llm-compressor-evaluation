from kfp import dsl

@dsl.component()
def validate_session_id(session_id: str, mount_path: str) -> None:
    """
    Validate that the session_id is unique. It should not
    match the name of any directory already in the artifacts
    PVC. If it does, raise a ValueError. If it does not, create
    the session directory to "claim" the session_id.
    """
    
    from pathlib import Path

    artifacts_path = Path(mount_path) / "evaluation-artifacts" / "sessions"
    if not artifacts_path.exists():
        raise ValueError(f"Artifacts path {artifacts_path} does not exist")

    existing_sessions = [d.name for d in artifacts_path.iterdir() if d.is_dir()]
    if session_id in existing_sessions:
        raise ValueError(f"Session ID '{session_id}' already exists in {artifacts_path}")
    
    # Create the session directory to "claim" the session_id
    session_path = artifacts_path / session_id
    session_path.mkdir(parents=True, exist_ok=False)
