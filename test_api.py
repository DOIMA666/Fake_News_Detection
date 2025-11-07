import requests
import json

API_URL = "http://localhost:8000/api/check"

# Test data
data = {
    "content": "Việt Nam đã tiêm hơn 200 triệu liều vaccine COVID-19 cho người dân trong năm 2021-2022",
    "input_type": "text",
    "num_sources": 3
}

print("Sending request...")
response = requests.post(API_URL, json=data)

print("\nStatus Code:", response.status_code)
print("\nResponse:")
result = response.json()
print(json.dumps(result, indent=2, ensure_ascii=False))

if result.get('success'):
    print("\n" + "="*50)
    print(f"VERDICT: {result['verdict']['label']}")
    print(f"Similarity: {result['verdict']['similarity_percentage']}%")
    print(f"Confidence: {result['verdict']['confidence_percentage']}%")
    print("="*50)