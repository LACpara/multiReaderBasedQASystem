import os
from dotenv import load_dotenv

if __name__ == "__main__":
    # load_dotenv()

    api_key = os.environ.get("API_KEY")
    url = os.environ.get("BASE_URL")
    headers = {
        
    }

    payload = {

    }
    
    import httpx

    with httpx.Client() as client:
        with client.stream("POST", url, headers=headers, json=payload) as response:
            pass