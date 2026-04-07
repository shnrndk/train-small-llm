from openai import OpenAI

# --- Configuration ---
UTSA_API_KEY = "gpustack_50e00c9281422bc5_0c0696dfcb1696d7635e58a2e56d6282"
UTSA_BASE_URL = "http://10.246.100.230/v1"

def list_models():
    client = OpenAI(
        api_key=UTSA_API_KEY,
        base_url=UTSA_BASE_URL
    )

    print("Fetching available models from the server...\n")
    try:
        models = client.models.list()
        for model in models.data:
            print(f"Exact Model ID: '{model.id}'")
    except Exception as e:
        print(f"Failed to fetch models: {e}")

if __name__ == "__main__":
    list_models()