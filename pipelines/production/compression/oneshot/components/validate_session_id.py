from kfp import dsl


@dsl.component(
    base_image="python:3.12",
)
def validate_session_id(
    session_id: str,
    models_mount_path: str = "/models",
) -> None:
    """
    Validate that the session_id is unique. It should not match the name of
    any directory already in /models/sessions. If it does not exist, create
    the session directory to "claim" the session_id.
    """

    from pathlib import Path

    sessions_path = Path(models_mount_path) / "sessions"
    if not sessions_path.exists():
        sessions_path.mkdir(parents=True, exist_ok=True)
        print(f"Created sessions directory: {sessions_path}")

    existing_sessions = [d.name for d in sessions_path.iterdir() if d.is_dir()]
    if session_id in existing_sessions:
        raise ValueError(f"Session ID '{session_id}' already exists in {sessions_path}")

    session_path = sessions_path / session_id
    session_path.mkdir(parents=True, exist_ok=False)
    print(f"Session directory created: {session_path}")
