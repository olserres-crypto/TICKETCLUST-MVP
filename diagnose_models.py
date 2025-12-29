import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="Gemini Diagnostic", page_icon="🕵️")

st.title("🕵️ Gemini Model Diagnostic")

st.markdown("""
This utility helps verify which Google Gemini models are available to your API Key.
This helps resolve `404 Not Found` errors.
""")

api_key = st.text_input("Enter your Google API Key", type="password")

if st.button("List Available Models"):
    if not api_key:
        st.error("Please enter a key first.")
    else:
        try:
            genai.configure(api_key=api_key)
            st.info("Querying Google AI API...")
            
            models = list(genai.list_models())
            
            st.subheader("✅ Available Generative Models:")
            found_flash = False
            
            for m in models:
                if 'generateContent' in m.supported_generation_methods:
                    name = m.name.replace("models/", "")
                    if "flash" in name:
                        st.success(f"**{name}** (Flash Model - Recommended)")
                        found_flash = True
                    else:
                        st.write(f"- {name}")
            
            if not found_flash:
                st.warning("⚠️ No 'flash' models found. Your key might be restricted to 'gemini-pro' or older models.")
                
        except Exception as e:
            st.error(f"❌ Error listing models: {e}")
