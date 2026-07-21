from kfp import dsl

@dsl.component(
    base_image="python:3.12",
    packages_to_install=["requests"],
)
def test_vllm_server(
    # Model spec
    model: str,
    # Service spec
    service_url: str,
) -> None:
    import json
    import requests

    models_url = f"{service_url}/v1/models"
    print(f"Checking {models_url}")

    models_response = requests.get(models_url, timeout=30)
    models_response.raise_for_status()
    print(json.dumps(models_response.json(), indent=2))

    chat_url = f"{service_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Briefly explain methods to evaluate large language models.",
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.6,
        "chat_template_kwargs": {
            "enable_thinking": True,
        },
    }

    response = requests.post(chat_url, json=payload, timeout=120)
    response.raise_for_status()

    data = response.json()
    print(json.dumps(data, indent=2))
    print("Assistant response:")
    print(data["choices"][0]["message"]["content"])

    print("Test inference successful")
