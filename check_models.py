import google.generativeai as genai
import os
import streamlit as st

# Function to try loading key from secrets or env
def get_key():
    try:
        # Try loading from secrets.toml if it works (but this is a script, not streamlit app running)
        # So we better parse it manually or rely on user input in a real interactive session.
        # But for this script, I'll checking if I can read the file directly or use a dummy.
        # Actually, let's just ask the library to use what it has or error out.
        # Better: Read the secrets file manually.
        import toml
        secrets = toml.load(".streamlit/secrets.toml")
        return secrets["GOOGLE_API_KEY"]
    except Exception as e:
        print(f"Could not load from secrets.toml: {e}")
        return None

key = get_key()
if not key:
    print("No API Key found in .streamlit/secrets.toml")
else:
    print(f"Using API Key: {key[:5]}...")
    genai.configure(api_key=key)
    try:
        print("Listing available models...")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
    except Exception as e:
        print(f"Error listing models: {e}")
