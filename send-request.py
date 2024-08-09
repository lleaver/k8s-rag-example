import requests

url = "http://localhost:8450/v1/chat/completions"
headers = {
    "accept": "application/json",
    "Content-Type": "application/json"
}
data = {
    "model": "meta/llama3-8b-instruct",
    "messages": [{"role": "user", "content": "What does NIM mean?"}],
    "max_tokens": 64
}

response = requests.post(url, headers=headers, json=data)

print(response.status_code)
print(response.json())
