def delete_pod_if_exists(
    pod_name: str,
    namespace: str,
    core,
) -> None:
    from kubernetes.client.rest import ApiException
    import time

    try:
        core.delete_namespaced_pod(name=pod_name, namespace=namespace)
        print(f"Deleted existing pod {pod_name}; waiting for deletion...")
    except ApiException as e:
        if e.status != 404:
            raise
        return

    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            core.read_namespaced_pod(name=pod_name, namespace=namespace)
            time.sleep(5)
        except ApiException as e:
            if e.status == 404:
                return
            raise
    raise TimeoutError(f"Timed out waiting for old pod {pod_name} to delete")

def create_or_patch_service(
    service_name: str,
    namespace: str,
    service_manifest: dict,
    core,
) -> None:
    from kubernetes.client.rest import ApiException

    try:
        core.create_namespaced_service(namespace=namespace, body=service_manifest)
        print(f"Created service {service_name}")
    except ApiException as e:
        if e.status ==409:
            core.patch_namespaced_service(
                name=service_name,
                namespace=namespace,
                body=service_manifest,
            )
            print(f"Patched existing service {service_name}")
        else:
            raise

def create_pod(
    pod_name: str,
    namespace: str,
    pod_manifest: dict,
    core,
) -> None:
    core.create_namespaced_pod(namespace=namespace, body=pod_manifest)
    print(f"Created pod {pod_name}")

def wait_for_pod_ready(
    pod_name: str,
    namespace: str,
    core,
    wait_timeout_seconds: int = 1800,
) -> None:
    import time

    deadline = time.time() + wait_timeout_seconds

    while time.time() < deadline:
        pod = core.read_namespaced_pod(name=pod_name, namespace=namespace)
        phase = pod.status.phase
        conditions = pod.status.conditions or []

        ready = any(
            condition.type == "Ready" and condition.status == "True"
            for condition in conditions
        )

        print(f"Pod phase={phase}, ready={ready}")

        if ready:
            return

        if phase in {"Failed", "Succeeded"}:
            raise RuntimeError(f"Pod ended before becoming ready: phase={phase}")

        time.sleep(15)

    raise TimeoutError(f"Timed out waiting for pod {pod_name} to become ready")

def wait_for_service_healthy(
    service_url: str,
    wait_timeout_seconds: int = 1800,
) -> None:
    import time
    import requests

    deadline = time.time() + wait_timeout_seconds
    url = f"{service_url}/v1/models"

    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=10)
            if response.ok:
                print(f"Service healthy: {response.text[:500]}")
                return
            print(f"Service responded with HTTP {response.status_code}: {response.text[:300]}")
        except Exception as e:
            print(f"Service not ready yet: {e}")

        time.sleep(10)

    raise TimeoutError(f"Timed out waiting for {url}")