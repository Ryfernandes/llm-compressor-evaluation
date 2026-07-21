def send_test_request(service_url: str, model: str) -> dict:
    import requests

    response = requests.post(
        f"{service_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": "What is the capital of Massachusetts?"}
            ],
            "max_tokens": 256,
            "temperature": 0.7,
            "thinking_token_budget": 20
        },
        timeout=120,
    )
    response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    reasoning = data["choices"][0]["message"]["reasoning"]

    print(f"Reasoning: {reasoning}\n\nResponse: {content}")

    return {
        "content": content,
        "usage": data.get("usage", {}),
        "finish_reason": data["choices"][0].get("finish_reason"),
    }
