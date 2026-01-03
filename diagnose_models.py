import google.generativeai as genai
import toml
import time
from google.api_core import exceptions

def test_model(model_name):
    print(f"\n--- Testing {model_name} ---")
    try:
        model = genai.GenerativeModel(model_name)
        # Generate a very simple, 1-token response to minimize cost
        response = model.generate_content("Hi", request_options={"timeout": 10})
        print(f"✅ SUCCESS! Response: {response.text.strip()}")
        return True
    except exceptions.NotFound:
        print("❌ 404 NOT FOUND - Model not available for this key.")
        return False
    except exceptions.ResourceExhausted as e:
        print(f"⚠️ QUOTA EXCEEDED (429) - {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

try:
    secrets = toml.load(".streamlit/secrets.toml")
    api_key = secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
    print(f"Validated API Key structure. Starting Probe...")
    
    candidates = [
        "gemini-1.5-flash",
        "gemini-1.5-flash-001",
        "gemini-1.5-flash-002",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
        "gemini-1.0-pro"
    ]
    
    working_models = []
    
    for m in candidates:
        if test_model(m):
            working_models.append(m)
            # Don't break, check all to give options
            
    print("\n" + "="*30)
    print("DIAGNOSTIC RESULTS")
    print("="*30)
    if working_models:
        print(f"Available Models: {working_models}")
        print(f"RECOMMENDATION: Update app.py to use '{working_models[0]}'")
    else:
        print("NO WORKING MODELS FOUND. All are either 404 (Invalid) or 429 (Quota Exhausted).")
        print("Solution: Wait 24 hours for daily quota reset.")

except Exception as e:
    print(f"Setup Failed: {e}")
