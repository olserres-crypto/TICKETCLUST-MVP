import streamlit as st
import pandas as pd
import random
import time
import json
import plotly.express as px
import google.generativeai as genai
from google.api_core import exceptions

# Set page configuration
st.set_page_config(
    page_title="TicketClust: AI-Powered Support Audit",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for "Enterprise" look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #0052cc;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #003d99;
    }
    h1 {
        color: #172b4d;
        font-weight: 700;
    }
    h2, h3 {
        color: #42526e;
    }
    .report-box {
        background-color: white;
        padding: 24px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #dfe1e6;
        margin-top: 20px;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        border-left: 6px solid #0052cc;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    .citation-box {
        background-color: #deebff;
        color: #0747a6;
        padding: 10px;
        border-radius: 4px;
        font-size: 0.9em;
        margin-top: 10px;
        border: 1px solid #b3d4ff;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

BENCHMARKS = {
    "Password Reset Assistant": {
        "rate": 0.90,
        "source": "Zendesk CX Trends: High deflection on transactional tickets",
        "url": "https://www.zendesk.com/cx-trends/"
    },
    "Routine Queries": {
        "rate": 0.80,
        "source": "IBM Report: AI reduces customer service costs",
        "url": "https://www.ibm.com/downloads/cas/42118481"
    },
    "Hybrid Support Model": {
        "rate": 0.50,
        "source": "Gartner: GenAI impact on Tier-1 Support",
        "url": "https://www.gartner.com/en/articles/generative-ai-impact-customer-service"
    },
    "Autonomous Customer Queries": {
        "rate": 0.70,
        "source": "Intercom: Automating Feedback Loops & Triage",
        "url": "https://www.intercom.com/blog/ai-customer-service-metrics/"
    },
     "Autonomous Service": {
        "rate": 0.75,
        "source": "Salesforce State of Service: Automation in Returns",
        "url": "https://www.salesforce.com/resources/articles/state-of-service/"
    },
    "General Inbound Volume": {
        "rate": 0.30,
        "source": "Salesforce State of Service Report",
        "url": "https://www.salesforce.com/resources/articles/state-of-service/"
    },
    "Internal IT Support": {
         "rate": 0.60,
         "source": "HBR: AI Augmentation for Complex Troubleshooting",
         "url": "https://hbr.org/2023/11/how-generative-ai-changes-productivity"
    },
    "Default": {
        "rate": 0.30,
        "source": "Salesforce State of Service Report",
        "url": "https://www.salesforce.com/resources/articles/state-of-service/"
    }
}

VALID_CATEGORIES = list(BENCHMARKS.keys()) 
# Remove "Default" from the list used for categorization guidance
if "Default" in VALID_CATEGORIES:
    VALID_CATEGORIES.remove("Default")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def validate_columns(df):
    """
    Checks if the dataframe contains at least one column suitable for analysis.
    Returns: (is_valid, text_column_name, message)
    """
    # Potential column names for text analysis
    potential_cols = ['Ticket Description', 'Ticket Subject', 'Description', 'Subject', 'Message', 'Body', 'Text', 'Issue']
    
    # Check for exact matches first
    for col in potential_cols:
        if col in df.columns:
            return True, col, "Found standard column."
            
    # Check for partial matches (case-insensitive)
    for col in df.columns:
        col_lower = str(col).lower()
        if any(keyword.lower() in col_lower for keyword in potential_cols):
            return True, col, "Found likely column."
    
    return False, None, "No suitable text column found (e.g., 'Ticket Description', 'Subject')."

def mock_analyze_tickets(df, text_col):
    """
    Fallback mock logic if no API key is provided.
    Simulates AI analysis by clustering rows based on keywords.
    """
    clusters = []
    sentiments = []
    
    texts = df[text_col].fillna('').astype(str).tolist()

    for text in texts:
        text_lower = text.lower()
        
        if "login" in text_lower or "password" in text_lower:
            clusters.append("Password Reset Assistant")
        elif "status" in text_lower or "where" in text_lower:
             clusters.append("Routine Queries")
        elif "billing" in text_lower or "invoice" in text_lower:
            clusters.append("Autonomous Customer Queries") # Mapped to one of the new categories
        elif "complex" in text_lower or "tier 2" in text_lower:
            clusters.append("Hybrid Support Model")
        elif "internal" in text_lower or "vpn" in text_lower:
             clusters.append("Internal IT Support")
        else:
            clusters.append("General Inbound Volume")
            
        random.seed(text) 
        base_sentiment = random.random()
        sentiments.append(base_sentiment)

    df['Predicted_Cluster'] = clusters
    df['Sentiment_Score'] = sentiments
    return df, None # Return None for raw_json as this is simulation

def analyze_with_gemini(df, text_col, api_key):
    """
    Real AI analysis using Google Gemini 1.5 Flash.
    Uses batching to process tickets efficiently.
    Returns: (enriched_df, raw_json_list)
    """
    genai.configure(api_key=api_key)
    # Use JSON mode for robustness
    # Using specific model version -001 to avoid alias resolution issues
    model_name = 'gemini-1.5-flash-001'
    try:
        model = genai.GenerativeModel(model_name, generation_config={"response_mime_type": "application/json"})
    except Exception:
        # Fallback to generic if specific fails, though unlikely to help if 404 comes from generation
        model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})
    
    # Batch configuration
    BATCH_SIZE = 10 
    records = df[[text_col]].fillna('No content').to_dict('records')
    
    # Add temporary ID for tracking
    for idx, r in enumerate(records):
        r['temp_id'] = idx

    results_map = {} # temp_id -> {category, sentiment}
    all_raw_json = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
    
    # Error tracking
    error_count = 0
    last_error = None

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        
        status_text.text(f"Analying batch {batch_num}/{total_batches} with Gemini 1.5 Flash...")
        progress_bar.progress(i / len(records))

        # Create Prompt
        prompt_data = [{"id": r['temp_id'], "text": str(r[text_col])[:500]} for r in batch]
        
        prompt = f"""
        You are a Customer Support Analyst. 
        Categorize these tickets into exactly one of these categories: {VALID_CATEGORIES}.
        Also analyze sentiment (0.0 to 1.0, where 1.0 is positive).
        
        Return a strict JSON list of objects:
        [{{ "id": <ticket_id>, "category": "<category_name>", "sentiment": <0.0-1.0> }}]
        
        Input Data:
        {json.dumps(prompt_data)}
        """
        
        try:
            response = model.generate_content(prompt)
            # With response_mime_type="application/json", response.text is already clean JSON
            cleaned_response = response.text.strip()
            batch_results = json.loads(cleaned_response)
            
            all_raw_json.extend(batch_results)
            
            for res in batch_results:
                results_map[res['id']] = res
                
        except json.JSONDecodeError as e:
            error_msg = f"Batch {batch_num} JSON Error: {e}"
            st.error(error_msg)
            print(error_msg)
            error_count += 1
            last_error = str(e)
            # Fallback for this batch
            for r in batch:
                results_map[r['temp_id']] = {"category": "General Inbound Volume", "sentiment": 0.5}
                
        except exceptions.ResourceExhausted:
            st.warning("Rate limit hit. Pausing for 10 seconds...")
            time.sleep(10)
            # Retry logic could be added, but for now fallback and warn
            error_count += 1
            last_error = "Rate Limit Exceeded"
            for r in batch:
                results_map[r['temp_id']] = {"category": "General Inbound Volume", "sentiment": 0.5}
                
        except Exception as e:
            error_msg = f"Batch {batch_num} Error: {str(e)}"
            st.error(error_msg)
            print(error_msg)
            error_count += 1
            last_error = str(e)
            for r in batch:
                results_map[r['temp_id']] = {"category": "General Inbound Volume", "sentiment": 0.5}

    progress_bar.progress(1.0)
    status_text.text("Analysis Complete!")
    
    if error_count > 0:
        st.warning(f"⚠️ Completed with {error_count} batch errors. Some tickets were assigned default values. Last error: {last_error}")
    
    # Map back to DF
    categories = []
    sentiments = []
    
    for idx in range(len(df)):
        res = results_map.get(idx, {"category": "General Inbound Volume", "sentiment": 0.5})
        # Normalize category if LLM hallucinated a slight variation
        cat = res.get('category', 'General Inbound Volume')
        if cat not in VALID_CATEGORIES:
             cat = "General Inbound Volume" # Fallback
             
        categories.append(cat)
        sentiments.append(res.get('sentiment', 0.5))
        
    df['Predicted_Cluster'] = categories
    df['Sentiment_Score'] = sentiments
    return df, all_raw_json

# -----------------------------------------------------------------------------
# Dashboard Rendering
# -----------------------------------------------------------------------------

def render_dashboard(analyzed_df, raw_json=None):
    """
    Renders the metrics, charts, and consultant report for the given dataframe.
    """
    # --- Metrics ---
    col1, col2, col3 = st.columns(3)
    
    num_tickets = len(analyzed_df)
    
    # Calculate Weighted Savings based on benchmarks per row
    total_potential_savings = 0
    for cluster in analyzed_df['Predicted_Cluster']:
        rate = BENCHMARKS.get(cluster, BENCHMARKS["Default"])["rate"]
        # Cost per ticket = 15 mins (0.25h) * 30 EUR/hr = 7.50 EUR
        savings_per_ticket = 7.50 * rate 
        total_potential_savings += savings_per_ticket
        
    avg_sentiment = analyzed_df['Sentiment_Score'].mean()

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Total Tickets Analyzed</h4>
            <h2>{num_tickets}</h2>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #ff5630;">
            <h4>Est. Total Wasted Budget</h4>
            <h2>€{total_potential_savings:,.2f}</h2>
            <small><i>Based on industry automation rates</i></small>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        sentiment_color = "#36b37e" if avg_sentiment > 0.5 else "#ffab00"
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {sentiment_color};">
            <h4>Avg. Customer Sentiment</h4>
            <h2>{avg_sentiment:.2f} / 1.0</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Visualizations ---
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Ticket Distribution by Issue Type")
        cluster_counts = analyzed_df['Predicted_Cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Issue Type', 'Count']
        
        fig = px.bar(
            cluster_counts, 
            x='Issue Type', 
            y='Count', 
            title="Volume by Category",
            color='Count',
            color_continuous_scale=px.colors.sequential.Bluyl
        )
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("Data Preview")
        # Show specific columns if they exist, otherwise just show what we have
        cols_to_show = [c for c in ['Predicted_Cluster', 'Sentiment_Score'] if c in analyzed_df.columns]
        # Attempt to find the text column to show preview
        text_cols = [c for c in analyzed_df.columns if 'description' in c.lower() or 'subject' in c.lower()]
        if text_cols:
             cols_to_show.insert(0, text_cols[0])
             
        st.dataframe(analyzed_df[cols_to_show].head(10), use_container_width=True)

    # --- Consultant Report ---
    st.markdown("---")
    st.subheader("🔍 Consultant Recommendations")
    
    # Identify top pain point
    top_cluster = cluster_counts.iloc[0]['Issue Type']
    top_count = cluster_counts.iloc[0]['Count']
    
    benchmark_data = BENCHMARKS.get(top_cluster, BENCHMARKS["Default"])
    deflection_rate = benchmark_data["rate"]
    citation_source = benchmark_data["source"]
    citation_url = benchmark_data["url"]
    
    # Calculate specific savings for this cluster
    cluster_cost = top_count * 0.25 * 30
    cluster_savings = cluster_cost * deflection_rate
    
    report_html = f"""
<div class="report-box">
<h3>🚀 Strategic Opportunity: Automate "{top_cluster}"</h3>
<p><strong>Observation:</strong> Your highest volume of support tickets ({top_count} tickets) falls under the category of <b>{top_cluster}</b>.</p>
<p><strong>Impact:</strong> This category alone consumes approximately <b>€{cluster_cost:,.2f}</b> of your resource budget.</p>
<p><strong>Recommendation:</strong> Implement a specialized RAG (Retrieval-Augmented Generation) agent trained on your knowledge base.</p>
<ul>
<li>Projected Deflection Rate: <b>{int(deflection_rate * 100)}%</b></li>
<li>Estimated Savings: <b>€{cluster_savings:,.2f}</b></li>
</ul>
<div class="citation-box">
ℹ️ <b>Benchmark Source:</b> Calculation based on industry standard deflection rates for {top_cluster}.<br>
According to <i>{citation_source}</i>, automation in this vertical achieves high efficiency.<br>
<a href="{citation_url}" target="_blank">Read the full report 🔗</a>
</div>
<br>
<button style="
background-color: #0052cc;
color: white;
border: none;
padding: 10px 20px;
text-align: center;
text-decoration: none;
display: inline-block;
font-size: 16px;
margin: 4px 2px;
cursor: pointer;
border-radius: 4px;">
Schedule Audit Consult
</button>
</div>
"""
    st.markdown(report_html, unsafe_allow_html=True)

    # --- JSON Debug Inspector ---
    if raw_json:
        st.markdown("---")
        with st.expander("🔍 View Raw AI JSON Output (Developer Mode)"):
            st.json(raw_json)

# -----------------------------------------------------------------------------
# Main App Layout
# -----------------------------------------------------------------------------

def main():
    # Header
    with st.container():
        st.title("TicketClust")
        st.markdown("### AI-Powered Support Audit & Opportunity Detector")
        st.markdown("---")

    # File Uploader
    st.sidebar.header("Data Input")
    st.sidebar.info("🔒 **Privacy Note:** Your data is processed in-memory. If providing an API key, it is only used for this session.")
    
    # --- API Key Management ---
    # Attempt to load from secret file
    api_key_from_secrets = None
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            api_key_from_secrets = st.secrets["GOOGLE_API_KEY"]
    except FileNotFoundError:
        pass # No secrets file, that's fine

    api_key = api_key_from_secrets
    
    # If not in secrets (or placeholder), show input. 
    # If in secrets, show disabled masked field or just a boolean toggle.
    if not api_key or "PASTE_YOUR_KEY" in api_key:
         api_key = st.sidebar.text_input("Google Gemini API Key", type="password", help="Enter your Gemini API key for real semantic analysis. Leave empty to use simulation mode.")
    else:
        st.sidebar.success("✅ **Gemini API Key Loaded** from backend.")
        # Optional: Allow override? For now, assume backend key is authoritative.

    # --- Simulation Toggle ---
    # Allow user to force simulation even if Key is present
    force_simulation = st.sidebar.checkbox("Force Simulation Mode (Keywords)", value=False, help="Check this to skip AI analysis and use keyword matching instead.")

    st.sidebar.markdown("""
    **Required Format:**
    Upload a CSV with at least a **Description** or **Subject** column.
    *(Compatible with Kaggle Customer Support datasets)*
    """)
    
    with st.sidebar.expander("ℹ️ How it Works & Troubleshooting", expanded=False):
        st.markdown("""
        **Process Overview:**
        1. **Input:** We process your CSV in batches (10 tickets at a time).
        2. **Analysis:** Google Gemini 1.5 Flash analyzes the text to determine:
           - **Category:** Matches strictly to the 7 defined Benchmarks.
           - **Sentiment:** Scores from 0.0 (Negative) to 1.0 (Positive).
        3. **Output:** A unified report with economic opportunities.
        
        **Error Control:**
        - If the AI fails (Rate Limit, API Error, JSON Error), we default that batch to:
          - **Category:** "General Inbound Volume"
          - **Sentiment:** 0.5 (Neutral)
        - You will see clear warnings if errors occur.
        
        **Common Issues:**
        - *All results look the same?* -> Check for error messages. Your API Key might be invalid or quota exceeded.
        """)

    uploaded_file = st.sidebar.file_uploader("Upload your Support Tickets (CSV)", type=['csv'])

    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            if df.empty:
                st.error("The uploaded CSV is empty. Please upload a file with data.")
                return

            # Validate Columns
            is_valid, text_col, msg = validate_columns(df)
            
            if not is_valid:
                st.error("❌ **Invalid Data Format**")
                st.warning("We couldn't find a column containing ticket descriptions. Please ensure your CSV has a column named 'Ticket Description', 'Subject', or similar.")
                with st.expander("See expected format example"):
                    st.markdown("""
                    | Ticket ID | **Ticket Description** | Date |
                    |-----------|------------------------|------|
                    | 1         | Login failed error...  | ...  |
                    """)
                return
            
            # Decide Analysis Mode
            analyzed_df = None
            raw_json_output = None
            
            # Use AI if Key exists AND simulation is NOT forced
            if api_key and not force_simulation:
                with st.spinner(f"Connecting to Gemini 1.5 Flash..."):
                     analyzed_df, raw_json_output = analyze_with_gemini(df.copy(), text_col, api_key)
            else:
                mode_label = "Simulation Mode (Forced)" if force_simulation else "Simulation Mode (No Key)"
                st.warning(f"⚠️ Running in **{mode_label}** (Keyword Matching).")
                with st.spinner(f"Simulating analysis..."):
                    time.sleep(1.0) 
                    analyzed_df, _ = mock_analyze_tickets(df.copy(), text_col)
            
            # Render results
            render_dashboard(analyzed_df, raw_json_output)

        except pd.errors.EmptyDataError:
            st.error("The file is valid CSV but contains no data.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            
    else:
        # Empty State
        st.info("👋 Welcome to TicketClust. Please upload a CSV file to generate your audit report.")
        
        # Create a sample dataframe for demonstration if the user has no file
        if st.checkbox("Or use sample data"):
            data = {
                'Ticket ID': range(1, 21),
                'Ticket Description': [
                    "Cannot login to my account", "Billing error on invoice #332", "App crashes when clicking save",
                    "How do I reset password?", "Feature request: Dark mode", "Login failed invalid credentials",
                    "I was charged twice", "Bug in the reporting module", "System is slow today", "Where can I find the API key?",
                     "Cannot login", "Invoice is wrong", "Application error 500", "Password reset link not working",
                     "New feature idea", "Login issue again", "Payment declided", "Bug on dashboard", "Nothing works", "Help with settings"
                ]
            }
            sample_df = pd.DataFrame(data)
            
            analyzed_df = None
            raw_json_output = None
            
            if api_key and not force_simulation:
                 analyzed_df, raw_json_output = analyze_with_gemini(sample_df.copy(), 'Ticket Description', api_key)
            else:
                 msg = "⚠️ Running Sample Data in Simulation Mode " + ("(Forced)" if force_simulation else "(No Key)")
                 st.warning(msg)
                 analyzed_df, _ = mock_analyze_tickets(sample_df.copy(), 'Ticket Description')
            
            # Render results
            render_dashboard(analyzed_df, raw_json_output)
            
            st.markdown("---")
            csv = sample_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Sample CSV",
                data=csv,
                file_name="sample_tickets.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()
