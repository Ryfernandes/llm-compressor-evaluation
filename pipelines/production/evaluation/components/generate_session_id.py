from kfp import dsl

@dsl.component()
def generate_session_id() -> str:
    from datetime import datetime, timezone
    import uuid

    timestamp = datetime.now(timezone.utc).strftime("%y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"{timestamp}-{suffix}"