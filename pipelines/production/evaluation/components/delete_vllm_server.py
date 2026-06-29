from kfp import dsl

@dsl.component(
    base_image="python:3.12",
    packages_to_install=["kubernetes"]
)
def delete_vllm_server(
    session_id: str,
    namespace: str = "machine-learning",
    pod_suffix: str = "baseline-server",
    tier2_pvc_name: str = "evaluation-pipeline-model-server-tier-2",
    delete_tier2_pvc: bool = False,
) -> None:
    import time
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException

    config.load_incluster_config()
    core = client.CoreV1Api()

    pod_name = f"{namespace}--{session_id}--{pod_suffix}"
    service_name = f"{pod_name}-svc"

    def ignore_404(fn, description: str) -> None:
        try:
            fn()
            print(f"Deleted {description}")
        except ApiException as e:
            if e.status == 404:
                print(f"{description} did not exist")
            else:
                raise
    
    ignore_404(
        lambda: core.delete_namespaced_pod(name=pod_name, namespace=namespace),
        f"pod/{pod_name}",
    )

    ignore_404(
        lambda: core.delete_namespaced_service(name=service_name, namespace=namespace),
        f"service/{service_name}",
    )

    tier1_pvc_name = f"{pod_name}-tier1"
    ignore_404(
        lambda: core.delete_namespaced_persistent_volume_claim(
            name=tier1_pvc_name,
            namespace=namespace,
        ),
        f"pvc/{tier1_pvc_name}",
    )

    if delete_tier2_pvc:
        ignore_404(
            lambda: core.delete_namespaced_persistent_volume_claim(
                name=tier2_pvc_name,
                namespace=namespace,
            ),
            f"pvc/{tier2_pvc_name}",
        )
    else:
        print(f"Preserved tier2 PV: {tier2_pvc_name}")

    # Wait for pod deletion to complete
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            core.read_namespaced_pod(name=pod_name, namespace=namespace)
            print("Waiting for pod deletion...")
            time.sleep(5)
        except ApiException as e:
            if e.status == 404:
                print("Pod deletion complete")
                return
            raise

    print("Timed out waiting for pod deletion, but delete was requested")