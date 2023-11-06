import requests


def test_api():
    request = {
        "prompt": "A cat sat on a mat",
        "stream": False,
        "max_new_tokens": 4,
    }
    response = requests.post("http://localhost:8000/generate", json=request)
    assert response.status_code == 200
    response.raise_for_status()
    assert "response" in response.json(), "Response should contain a 'response' field"
    assert isinstance(response.json()["response"], str), "Response should be a string"
    assert len(response.json()["response"]) > len(request["prompt"]), "Response should be longer than the prompt"
