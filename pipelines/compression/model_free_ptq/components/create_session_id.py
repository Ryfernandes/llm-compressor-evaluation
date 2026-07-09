from kfp import dsl


@dsl.component(
    base_image="python:3.12",
)
def create_session_id() -> str:
    """Generate a unique session ID in MMDDYYYY-uuid[:4] format."""

    from datetime import datetime
    import uuid

    date_str = datetime.now().strftime("%m%d%Y")
    short_uuid = uuid.uuid4().hex[:4]
    session_id = f"{date_str}-{short_uuid}"

    print(f"Generated session ID: {session_id}")
    return session_id
