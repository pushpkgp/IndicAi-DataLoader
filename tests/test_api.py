import requests

# Define the API endpoint
url = "http://localhost:8000/api/load/images"

# Define the request payload
payload = {
    "index_path": "data/metadata/image.csv",
    "modality": "image",
    "split": "train"
}

# Send POST request
response = requests.post(url, json=payload)

# Parse and display response
if response.status_code == 200:
    print("✅ API Response:")
    print(response.json())
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)
