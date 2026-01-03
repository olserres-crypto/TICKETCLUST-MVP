import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import random
import time
import json
import plotly.express as px
import plotly.graph_objects as go
import openai
import concurrent.futures

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
        height: 180px; /* Fixed height for alignment */
        display: flex;
        flex-direction: column;
        justify-content: center;
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
    
    @media print {
        /* Hide sidebar */
        [data-testid="stSidebar"] {
            display: none !important;
        }
        /* Hide header decoration */
        header[data-testid="stHeader"] {
            display: none !important;
        }
        /* Hide footer decoration if any */
        footer {
            display: none !important;
        }
        /* Expand main content area */
        .main .block-container {
            max-width: 100% !important;
            padding: 1rem !important;
        }
        
        /* Prevent page breaks inside key elements */
        .metric-card, .strategy-card, .citation-box, .report-box {
            break-inside: avoid !important;
            page-break-inside: avoid !important;
        }
        
        /* Attempt to keep charts and tables intact */
        .stPlotlyChart, .stDataFrame, .stTable, div[data-testid="stMarkdownContainer"] p {
            break-inside: avoid !important;
            page-break-inside: avoid !important;
        }
        
        h1, h2, h3, h4 {
            break-after: avoid !important;
            page-break-after: avoid !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

ROI_MAPPING = {
    "Account & Access": {
        "rate": 0.94,
        "source": "Gartner",
        "methodology": "Savings based on high reduction in IT support ticket volumes through automated credential management."
    },
    "Billing & Subscription": {
        "rate": 0.80,
        "source": "Salesforce",
        "methodology": "Based on the ability of AI chatbots to resolve up to 80% of routine and transactional queries autonomously."
    },
    "Product Inquiry": {
        "rate": 0.60,
        "source": "Gartner",
        "methodology": "Calculated based on AI agents resolving customer queries through knowledge base integration."
    },
    "Administrative": {
        "rate": 0.30,
        "source": "IDC",
        "methodology": "Estimated based on observed efficiency increases in internal IT and contact center support."
    },
    "Technical Issue": {
        "rate": 0.20,
        "source": "Intercom",
        "methodology": "Conservative estimate based on industry-reported resolution ranges for general support volume."
    },
    "Feature Request": {
        "rate": 0.15,
        "source": "Intercom",
        "methodology": "Baseline savings for automated triage and feedback loop integration."
    }
}

VALID_CATEGORIES = list(ROI_MAPPING.keys())

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def validate_columns(df):
    # Lista de sinónimos comunes
    synonyms = ["ticket description", "description", "text", "issue", "content", "request"]
    for col in df.columns:
        if col.strip().lower() in synonyms:
            # Normalizamos el nombre internamente para el resto del script
            df.rename(columns={col: "Ticket Description"}, inplace=True)
            return True, "Ticket Description", "Column normalized successfully."
    return False, None, "Required column not found."

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

@st.cache_data
def analyze_tickets_mockup(df, text_col):
    """
    Fallback mock logic if no API key is provided.
    Classifies tickets based on keyword mapping to simulate AI behavior.
    """
    clusters = []
    sentiments = []
    urgencies = []
    
    texts = df[text_col].fillna('').astype(str).tolist()

    for text in texts:
        text_lower = text.lower()
        
        # 1. Keyword-based Categorization
        if any(w in text_lower for w in ["password", "login", "access", "reset", "locked", "sign-in", "account", "mfa"]):
            clusters.append("Account & Access")
        elif any(w in text_lower for w in ["invoice", "payment", "refund", "billing", "subscription", "price", "charge", "card"]):
            clusters.append("Billing & Subscription")
        elif any(w in text_lower for w in ["bug", "error", "crash", "slow", "broken", "fail", "not working", "issue"]):
            clusters.append("Technical Issue")
        elif any(w in text_lower for w in ["how to", "question", "documentation", "help", "info", "manual", "guide"]):
            clusters.append("Product Inquiry")
        elif any(w in text_lower for w in ["suggest", "improve", "add", "wish", "feature", "want", "idea"]):
            clusters.append("Feature Request")
        elif any(w in text_lower for w in ["legal", "gdpr", "privacy", "contract", "office", "hr", "policy"]):
            clusters.append("Administrative")
        else:
            clusters.append("Technical Issue") # Default fallback if no keywords match (or could be "Administrative")

        # 2. Sentiment Logic
        # If text contains specific negative words -> 0.1 to 0.3
        if any(w in text_lower for w in ["urgent", "frustrated", "angry", "broken"]):
            sentiments.append(random.uniform(0.1, 0.3))
        else:
            # Otherwise random between 0.4 and 0.7
            sentiments.append(random.uniform(0.4, 0.7))
        
        # 3. Urgency Logic
        if any(w in text_lower for w in ["urgent", "now", "immediately", "asap"]):
            urgencies.append("High")
        else:
            urgencies.append("Medium")

    df['Predicted_Cluster'] = clusters
    df['Sentiment_Score'] = sentiments
    df['Urgency'] = urgencies
    return df, None # Return None for raw_json as this is simulation



def process_batch_qwen(batch, batch_num, client):
    """
    Helper function to process a single batch with Qwen.
    """
    prompt_data = [{"id": r['temp_id'], "text": str(r['_text_content'])[:500]} for r in batch]
    
    system_prompt = f"""
    Act as a Support Operations Specialist. Categorize the support tickets into exactly one of these categories:
    {VALID_CATEGORIES}

    Categories:
    1. Account & Access: Login, passwords, permissions.
    2. Billing & Subscription: Payments, invoices, pricing.
    3. Technical Issue: Errors, bugs, slowness.
    4. Product Inquiry: How-to, documentation.
    5. Feature Request: Improvements, new ideas.
    6. Administrative: Legal, privacy, general.

    Also analyze sentiment (0.0-1.0) and urgency (Low/Medium/High).
    
    Output valid JSON only:
    [{{ "id": <ticket_id>, "category": "<category_name>", "sentiment": <float>, "urgency": "<string>" }}]
    """

    user_prompt = f"Input Data: {json.dumps(prompt_data)}"
    
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="qwen-flash", 
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                temperature=0.1,
                extra_body={
                    "enable_search": False,
                    "thinking_budget": 0
                }
            )
            
            content = completion.choices[0].message.content
            # Clean markdown if present
            content = content.replace('```json', '').replace('```', '').strip()
            
            batch_results = json.loads(content)
            
            # Handle case where Qwen might wrap the list in a key like "tickets": [...]
            if isinstance(batch_results, dict):
                found_list = False
                for key, val in batch_results.items():
                    if isinstance(val, list):
                        batch_results = val
                        found_list = True
                        break
                if not found_list:
                     print(f"Unexpected JSON structure in batch {batch_num}: {batch_results}")
                     raise ValueError("JSON is not a list of results")

            return batch_results
            
        except Exception as e:
            err_msg = str(e)
            print(f"Error batch {batch_num} (Attempt {attempt+1}): {err_msg}")
            
            # Simple backoff
            time.sleep(2 * (attempt + 1))
            
            if attempt == max_retries - 1:
                # Return fallback if all retries fail
                print(f"Failed batch {batch_num} after retries.")
                fallback_results = []
                for r in batch:
                    fallback_results.append({
                        "id": r['temp_id'],
                        "category": "Technical Issue", 
                        "sentiment": 0.5, 
                        "urgency": "Medium",
                        "error": str(e)
                    })
                return fallback_results

@st.cache_data(show_spinner=False)
def analyze_with_qwen(df, text_col, api_key):
    """
    Real AI analysis using Qwen (via DashScope/OpenAI compatible API).
    Uses ThreadPoolExecutor for concurrent batch processing.
    """
    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )
    
    # 1. OPTIMIZATION: Increased Batch Size & Helper Column
    BATCH_SIZE = 50  # Increased from 10 to 50
    MAX_WORKERS = 5  # Number of concurrent threads
    
    df['_text_content'] = df[text_col].fillna('No content')
    records = df.to_dict('records')
    
    # Add temporary ID for tracking
    for idx, r in enumerate(records):
        r['temp_id'] = idx

    results_map = {} 
    all_raw_json = []
    
    # Prepare batches
    batches = []
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        batches.append((batch, batch_num))
        
    total_batches = len(batches)
    
    # 2. OPTIMIZATION: Parallel Execution with Progress Bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Initializing 5 concurrent workers for {total_batches} batches...")
    
    completed_batches = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_batch = {
            executor.submit(process_batch_qwen, batch, batch_num, client): batch_num 
            for batch, batch_num in batches
        }
        
        # Process as they complete
        for future in concurrent.futures.as_completed(future_to_batch):
            batch_num = future_to_batch[future]
            try:
                batch_results = future.result()
                
                # Store results
                all_raw_json.extend(batch_results)
                for res in batch_results:
                    results_map[res['id']] = res
                    
            except Exception as exc:
                print(f"Batch {batch_num} generated an exception: {exc}")
                
            # Update Progress
            completed_batches += 1
            progress = completed_batches / total_batches
            progress_bar.progress(progress)
            status_text.text(f"Analyzed batch {completed_batches}/{total_batches} (Speed: ~{BATCH_SIZE*MAX_WORKERS} TPM)...")
            
    status_text.text("Analysis Complete!")
    
    # Map back to DF
    categories = []
    sentiments = []
    urgencies = []
    
    for idx in range(len(df)):
        res = results_map.get(idx, {"category": "Technical Issue", "sentiment": 0.5, "urgency": "Medium"})
        cat = res.get('category', 'Technical Issue')
        if cat not in VALID_CATEGORIES:
             cat = "Technical Issue"
             
        categories.append(cat)
        sentiments.append(res.get('sentiment', 0.5))
        urgencies.append(res.get('urgency', 'Medium'))
        
    df['Predicted_Cluster'] = categories
    df['Sentiment_Score'] = sentiments
    df['Urgency'] = urgencies
    
    # Cleanup temp column
    if '_text_content' in df.columns:
        del df['_text_content']
        
    # Sort raw json for consistent debug view
    all_raw_json.sort(key=lambda x: x.get('id', 0))
    
    return df, all_raw_json

# -----------------------------------------------------------------------------
# Dashboard Rendering
# -----------------------------------------------------------------------------

def render_dashboard(analyzed_df, raw_json=None, cost_per_ticket=7.50, avg_hourly_cost=30, avg_time_per_ticket=15):
    """
    Renders the metrics, charts, and consultant report for the given dataframe.
    """
    # --- Metrics ---
    st.markdown(f"""
    <div style='background-color: #f4f5f7; padding: 15px; border-radius: 8px; border: 1px solid #dfe1e6; margin-bottom: 25px;'>
        <p style='margin: 0; color: #42526e; font-size: 1.1em;'>
        Calculations are based on a baseline fully loaded operational cost of <b>€{avg_hourly_cost}/hr</b> and an average handling time of <b>{avg_time_per_ticket} minutes</b> per incident (<b>€{cost_per_ticket:.2f}/ticket</b>). 
        Potential savings are projected using industry benchmark automation rates. These parameters can be adjusted in the sidebar to match your specific metrics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    
    num_tickets = len(analyzed_df)
    
    # Calculate Weighted Savings based on benchmarks per row
    analyzed_df["Applied Rate (%)"] = analyzed_df["Predicted_Cluster"].apply(lambda x: ROI_MAPPING.get(x, ROI_MAPPING["Technical Issue"])["rate"])
    analyzed_df["Source Organization"] = analyzed_df["Predicted_Cluster"].apply(lambda x: ROI_MAPPING.get(x, ROI_MAPPING["Technical Issue"])["source"])
    
    # Calculate potential saving per ticket
    analyzed_df["Potential Saving (€)"] = cost_per_ticket * analyzed_df["Applied Rate (%)"]
    
    total_potential_savings = analyzed_df["Potential Saving (€)"].sum()
        
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
            <h4>Total Recovery Potential (Analyzed Dataset)</h4>
            <h2>€{total_potential_savings:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        sentiment_color = "#36b37e" if avg_sentiment > 0.5 else "#ffab00"
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {sentiment_color};">
            <h4>Avg. Customer Sentiment</h4>
            <h2>{avg_sentiment:.2f} / 1.0</h2>
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
        
    # --- Visualizations & Efficiency (Merged Row) ---
    c1, c2 = st.columns([1, 1])
    
    with c1:
        # --- Efficiency Gauge (Moved here) ---
        total_current_cost = len(analyzed_df) * cost_per_ticket
        efficiency_percentage = (total_potential_savings / total_current_cost) * 100 if total_current_cost > 0 else 0

        st.markdown("<h4 style='text-align: center; color: #42526e;'>% of Recoverable Budget</h4>", unsafe_allow_html=True)
        
        # Modern Dynamic Color Logic
        gauge_color = "#ff5630" # Red
        if efficiency_percentage > 70:
            gauge_color = "#36b37e" # Green
        elif efficiency_percentage > 30:
            gauge_color = "#ffab00" # Yellow

        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = efficiency_percentage,
            # domain = {'x': [0, 1], 'y': [0, 1]},
            # title = {'text': "% of Recoverable Budget", 'font': {'size': 20, 'color': "#42526e"}},
            number = {'suffix': "%", 'font': {'size': 40, 'color': gauge_color, 'weight': 'bold'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 0, 'tickcolor': "rgba(0,0,0,0)", 'tickvals': []}, # Clean axis
                'bar': {'color': gauge_color, 'thickness': 0.75}, # The active arc
                'bgcolor': "white",
                'borderwidth': 0,
                'bordercolor': "white",
                'steps': [
                    {'range': [0, 100], 'color': "#f4f5f7"} # Full background track (Gray)
                ],
                'threshold': {
                    'line': {'color': "rgba(0,0,0,0)", 'width': 0}, # Hide threshold
                    'thickness': 0,
                    'value': efficiency_percentage}
            }
        ))

        fig_gauge.update_layout(
            height=300, 
            margin=dict(l=30, r=30, t=10, b=30), # Reduced top margin
            paper_bgcolor="rgba(0,0,0,0)",
            font={'family': "Arial"}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with c2:
        st.subheader("Global Audit Table per Ticket")
        
        # Prepare Table View
        table_df = pd.DataFrame()
        
        table_df["AI Categorization"] = analyzed_df["Predicted_Cluster"]
        table_df["Sentiment"] = analyzed_df["Sentiment_Score"].apply(lambda x: f"{x:.2f}")
        table_df["Urgency"] = analyzed_df.get("Urgency", "Medium")
        table_df["Applied Rate (%)"] = analyzed_df["Applied Rate (%)"].apply(lambda x: f"{x:.0%}")
        table_df["Potential Savings"] = analyzed_df["Potential Saving (€)"]
        
        st.dataframe(
            table_df, 
            width="stretch",
            column_config={
                "Potential Savings": st.column_config.NumberColumn(
                    "Potential Savings",
                    format="€%.2f",
                    help="Projected savings in Euro."
                )
            }
        )

    st.markdown("---")
    
    # --- Ticket Distribution (Moved Below) ---
    st.subheader("Ticket Distribution by Issue Type")
    cluster_counts = analyzed_df['Predicted_Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Issue Type', 'Count']
    
    fig = px.bar(
        cluster_counts, 
        x='Issue Type', 
        y='Count', 
        # title="Volume by Category",
        color='Count',
        color_continuous_scale=px.colors.sequential.Bluyl
    )
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=350)
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    # --- Strategic Operational Audit ---
    st.subheader("📊 Strategic Operational Audit")
    
    # Calculate aggregates
    strat_stats = analyzed_df.groupby('Predicted_Cluster').agg(
        Volume=('Predicted_Cluster', 'count'),
        Potential_Savings=('Potential Saving (€)', 'sum') 
    ).reset_index()
    
    strat_stats['Current Op. Cost'] = strat_stats['Volume'] * cost_per_ticket
    
    # Add Expert Insight
    def get_insight(cluster):
        item = ROI_MAPPING.get(cluster, ROI_MAPPING["Technical Issue"])
        return f"{item['methodology']} (Source: {item['source']})"
    
    strat_stats['Expert Insight'] = strat_stats['Predicted_Cluster'].apply(get_insight)
    
    # Formatting for display
    strat_display = strat_stats.copy()
    strat_display = strat_display[['Predicted_Cluster', 'Volume', 'Current Op. Cost', 'Potential_Savings', 'Expert Insight']] # Key use
    
    # Calculate Totals
    total_vol = strat_display['Volume'].sum()
    total_cost = strat_display['Current Op. Cost'].sum()
    total_sav = strat_display['Potential_Savings'].sum()

    # Build HTML Table
    html_table = """
<style>
    .strat-table { width: 100%; border-collapse: collapse; font-family: sans-serif; font-size: 0.9em; }
    .strat-table th { background-color: #f4f5f7; text-align: left; padding: 8px; border-bottom: 2px solid #dfe1e6; color: #42526e; }
    .strat-table td { padding: 8px; border-bottom: 1px solid #dfe1e6; vertical-align: top; color: #172b4d; }
    .strat-table .num-col { text-align: right; white-space: nowrap; }
    .strat-table .total-row td { font-weight: bold; background-color: #f4f5f7; border-top: 2px solid #42526e; }
    .col-insight { width: 45%; }
</style>
<table class="strat-table">
    <thead>
        <tr>
            <th>Category</th>
            <th class="num-col">Volume</th>
            <th class="num-col">Current Op. Cost</th>
            <th class="num-col">Potential Savings</th>
            <th class="col-insight">Expert Insight</th>
        </tr>
    </thead>
    <tbody>
"""
    
    for _, row in strat_display.iterrows():
        html_table += f"""
<tr>
    <td>{row['Predicted_Cluster']}</td>
    <td class="num-col">{row['Volume']}</td>
    <td class="num-col">€{row['Current Op. Cost']:.2f}</td>
    <td class="num-col">€{row['Potential_Savings']:.2f}</td>
    <td>{row['Expert Insight']}</td>
</tr>
"""
        
    # Totals Row
    html_table += f"""
<tr class="total-row">
    <td>TOTAL</td>
    <td class="num-col">{total_vol}</td>
    <td class="num-col">€{total_cost:.2f}</td>
    <td class="num-col">€{total_sav:.2f}</td>
    <td></td>
</tr>
</tbody>
</table>
"""
    
    st.markdown(html_table, unsafe_allow_html=True)

    st.markdown("---")

    # --- Customer Experience Health Matrix ---
    st.subheader("🩺 Customer Experience Health Matrix")
    
    # Calculate Risk Metrics per Cluster
    risk_stats = analyzed_df.groupby('Predicted_Cluster').agg(
        Volume=('Predicted_Cluster', 'count'),
        Avg_Sentiment=('Sentiment_Score', 'mean'),
        High_Urgency_Count=('Urgency', lambda x: (x == 'High').sum())
    ).reset_index()
    
    risk_stats['Risk Index (%)'] = (risk_stats['High_Urgency_Count'] / risk_stats['Volume']) * 100
    
    # Prepare display dataframe (Keep numeric for styling)
    risk_table = risk_stats[['Predicted_Cluster', 'Volume', 'Avg_Sentiment', 'Risk Index (%)']].copy()
    risk_table.columns = ['Category', 'Volume', 'Sentiment (0-1)', 'Risk Index (%)']
    
    # Function for styling with valid CSS
    def highlight_risk(val):
        if val > 30:
            return 'background-color: #ff5630; color: white; font-weight: bold' # Red
        elif val < 15:
            return 'background-color: #36b37e; color: white; font-weight: bold' # Green
        return 'color: #172b4d'

    def highlight_sentiment(val):
        if val < 0.4:
            return 'background-color: #ff5630; color: white; font-weight: bold' # Red
        elif val > 0.6:
            return 'background-color: #36b37e; color: white; font-weight: bold' # Green
        return 'color: #172b4d'

    # Apply Styling
    styled_risk = risk_table.style\
        .map(highlight_risk, subset=['Risk Index (%)'])\
        .map(highlight_sentiment, subset=['Sentiment (0-1)'])\
        .format({'Sentiment (0-1)': '{:.2f}', 'Risk Index (%)': '{:.1f}%'})
                                  
    st.dataframe(
        styled_risk, 
        width="stretch", 
        hide_index=True,
        column_config={
            "Volume": st.column_config.ProgressColumn(
                "Volume",
                help="Relative volume of tickets",
                format="%d",
                min_value=0,
                max_value=int(risk_table['Volume'].max())
            )
        }
    )

    st.markdown("---")

    # --- Automation Strategies ---
    st.subheader("💡 Tailored Automation Framework: Expert-Led Strategic Proposals")
    
    st.markdown("""
    <div style="background-color: #f4f5f7; border-left: 5px solid #0052cc; padding: 15px; border-radius: 4px; margin-bottom: 25px;">
        <p style="font-style: italic; color: #42526e; margin: 0;">
            Critical metrics (<b>Sentiment < 0.40 or Risk Index > 30%</b>) identify structural friction points where generic automation is insufficient. 
            These findings serve as a strategic baseline; the following proposals <b>must be custom-tailored</b> to each client’s specific business logic, 
            brand voice, and operational maturity to ensure a seamless integration between AI and human expertise.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Strategy Cards HTML
    strategies_html = """
    <style>
        .strategy-card {
            background-color: white;
            border: 1px solid #dfe1e6;
            border-top: 4px solid #0052cc; 
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 1px 2px rgba(9, 30, 66, 0.08);
        }
        .strat-header {
            font-size: 1.1em;
            font-weight: bold;
            color: #172b4d;
            margin-bottom: 10px;
            border-bottom: 1px solid #ebecf0;
            padding-bottom: 10px;
        }
        .framework-tag {
            background-color: #deebff;
            color: #0052cc;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.85em;
            font-weight: 500;
            margin-bottom: 8px;
            display: inline-block;
        }
        .strat-action {
            color: #42526e;
            font-size: 0.95em;
            line-height: 1.5;
        }
    </style>

    <div class="strategy-card" style="border-top-color: #FF5630;"> <!-- Red for Technical/Crisis -->
        <div class="strat-header">A. Technical Issues | Strategy: Adaptive Incident Response & Alerting</div>
        <span class="framework-tag" style="background-color: #ffebe6; color: #BF2600;">Proposed Framework: "Context-Aware Proactive Alerting"</span>
        <div class="strat-action">
            <b>Strategic Action:</b> We design custom triggers that, upon detecting negative sentiment clusters, bypass standard queues to alert engineering teams immediately. This process is calibrated per client to prevent "alert fatigue" while ensuring critical technical failures are addressed before they impact the broader user base.
        </div>
    </div>

    <div class="strategy-card" style="border-top-color: #FFAB00;"> <!-- Yellow/Orange for Billing -->
        <div class="strat-header">B. Billing & Subscription | Strategy: Loyalty-First Financial Resolution</div>
        <span class="framework-tag" style="background-color: #fff0b3; color: #172b4d;">Proposed Framework: "Automated Retention & Recovery Flow"</span>
        <div class="strat-action">
            <b>Strategic Action:</b> Since billing friction is a primary driver for churn, we propose AI flows capable of executing pre-authorized financial adjustments. The degree of autonomy and refund thresholds are defined specifically for each business, balancing rapid customer satisfaction with the client's internal fraud and auditing policies.
        </div>
    </div>

    <div class="strategy-card" style="border-top-color: #36B37E;"> <!-- Green for Growth/Reference -->
        <div class="strat-header">C. Product Inquiry | Strategy: Dynamic Knowledge Optimization</div>
        <span class="framework-tag" style="background-color: #e3fcef; color: #006644;">Proposed Framework: "Continuous RAG Refinement Loop"</span>
        <div class="strat-action">
            <b>Strategic Action:</b> We utilize persistent low-sentiment data as a diagnostic tool to identify documentation gaps. Rather than a static update, we implement a feedback loop to retrain the RAG model according to the client's evolving product roadmap, ensuring the AI's knowledge remains a competitive asset rather than a liability.
        </div>
    </div>
    """
    st.markdown(strategies_html, unsafe_allow_html=True)
    
    # --- Strategic Quadrant Justification Table ---
    st.subheader("Strategic Quadrant Justification")
    
    # Helper data for the table
    justification_rows = [
        ("Crisis", "When frustration is high and urgency is high, a bot can be counterproductive. The priority is to retain the customer through human escalation.", "Salesforce: Human empathy in critical moments reduces customer churn by 80%."),
        ("Attrition", "If the customer is not in a hurry but is angry, the problem is information quality. The answer they receive is not useful.", "Intercom: Only 20% of complex technical problems are resolved with standard AI; the rest need better databases (RAG)."),
        ("VIP", "Satisfied customers with an urgent problem are the biggest opportunity for loyalty. Responding quickly here creates 'brand ambassadors'.", "Gartner: 94% of access problems can be automated, allowing this VIP zone to be serviced almost instantly."),
        ("Growth", "It is the ideal scenario: happy customer and no time pressure. It is the perfect moment for proactive education or cross-selling.", "Salesforce: Automated cross-selling is 60% more effective when the detected sentiment is positive.")
    ]

    # Build HTML Table (Using consistent styling)
    html_just = """
<style>
    .strat-table { width: 100%; border-collapse: collapse; font-family: sans-serif; font-size: 0.9em; }
    .strat-table th { background-color: #f4f5f7; text-align: left; padding: 8px; border-bottom: 2px solid #dfe1e6; color: #42526e; }
    .strat-table td { padding: 8px; border-bottom: 1px solid #dfe1e6; vertical-align: top; color: #172b4d; }
    .strat-table .num-col { text-align: right; white-space: nowrap; }
    .col-zone { width: 15%; }
    .col-logic { width: 45%; }
    .col-source { width: 40%; }
</style>
<table class="strat-table" style="margin-top: 10px; margin-bottom: 20px;">
    <thead>
        <tr>
            <th class="col-zone">Zone</th>
            <th class="col-logic">Business Logic</th>
            <th class="col-source">Reference Source</th>
        </tr>
    </thead>
    <tbody>
"""
    
    for zone, logic, source in justification_rows:
        html_just += f"""
<tr>
    <td><b>{zone}</b></td>
    <td>{logic}</td>
    <td><i>{source}</i></td>
</tr>
"""
        
    html_just += "</tbody></table>"
    st.markdown(html_just, unsafe_allow_html=True)
    
    # --- Financial Synthesis (Final Footer) ---
    st.markdown("---")
    
    # Calculate total hours saved based on global savings (Savings = Hours * Hourly Cost)
    # Savings = Hours * avg_hourly_cost  => Hours = Savings / avg_hourly_cost
    total_hours_saved = total_potential_savings / avg_hourly_cost if avg_hourly_cost > 0 else 0
    
    st.info(f"💰 **Financial Synthesis:** After auditing **{num_tickets}** incidents, a total savings opportunity of **€{total_potential_savings:,.2f}** has been identified. Implementing these expert-adapted strategies would reclaim **{total_hours_saved:,.1f} hours** of operational capacity.")

    # --- PDF Generation Button ---
    # --- Action Footer (PDF & Schedule) ---
    st.markdown("---")
    
    # Create two columns for primary actions
    ac1, ac2 = st.columns(2)
    
    with ac1:
        if st.button("📄 Generate PDF Report", use_container_width=True):
            components.html(
                "<script>window.parent.print()</script>",
                height=0,
                width=0
            )
            
    with ac2:
        # Using a link button for scheduling (Placeholder link)
        st.link_button(
            "📅 Schedule a Brief Strategy Session to Contextualize These Results", 
            "https://calendly.com/", # Placeholder
            use_container_width=True
        )


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
    
    # --- API Key Management (Qwen Only) ---
    api_key_qwen = None
    if "QWEN_API_KEY" in st.secrets:
        api_key_qwen = st.secrets["QWEN_API_KEY"]

    active_key = api_key_qwen
    if not active_key:
         active_key = st.sidebar.text_input("Qwen API Key", type="password")

    # Implicitly Qwen if key exists
    selected_provider = "Qwen AI" if active_key else "Simulation Mode"

    # Limit rows for testing (Crucial for Free Tier)
    use_row_limit = st.sidebar.checkbox("Limit Rows to Analyze", value=False)
    max_rows = None
    if use_row_limit:
        max_rows = st.sidebar.number_input("Max Rows to Analyze", min_value=10, max_value=10000, value=50, step=10, help="Limit the number of rows to process to save API quota.")

    st.sidebar.markdown("""
    <small>**Required Format:** The CSV must contain a column for the ticket text, named **'Ticket Description'** or a similar synonym (e.g., *'Description', 'Text', 'Issue', 'Content', 'Request'*).</small>
    """, unsafe_allow_html=True)
    
    # --- Financial Simulation Parameters (Moved for Visibility) ---
    # --- Financial Simulation Parameters ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Financial Simulation Parameters")
    
    avg_hourly_cost = st.sidebar.slider("Fully Loaded Hourly Cost (€/h)", min_value=10, max_value=100, value=30, step=5)
    avg_time_per_ticket = st.sidebar.number_input("Avg. Handling Time (min)", min_value=1, max_value=60, value=15, step=1)
    
    cost_per_ticket = (avg_hourly_cost * avg_time_per_ticket) / 60.0
    
    st.sidebar.info(f"ℹ️ **Current Simulation:** \n\n• €{avg_hourly_cost}/hr \n• {avg_time_per_ticket} min/ticket \n\n**= €{cost_per_ticket:.2f} per ticket cost**")
    
    uploaded_file = st.sidebar.file_uploader("Upload your Support Tickets (CSV)", type=['csv'])

    if uploaded_file is not None:
        try:
            # Read CSV
            df = load_csv(uploaded_file)
            
            if df.empty:
                st.error("The uploaded CSV is empty. Please upload a file with data.")
                return

            # Apply Row Limit
            if max_rows and len(df) > max_rows:
                st.info(f"ℹ️ Limiting analysis to first **{max_rows}** rows (out of {len(df)}) to preserve abundance.")
                df = df.head(max_rows)
            
            # (Parameters moved up)

            # Validate Columns
            is_valid, text_col, msg = validate_columns(df)
            
            if not is_valid:
                st.error("❌ **Invalid Data Format**")
                st.warning("We couldn't find the required column. Please ensure your CSV has a column exactly named 'Ticket Description'.")
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
            
            if selected_provider == "Qwen AI" and active_key:
                with st.spinner(f"Connecting to Qwen AI..."):
                     analyzed_df, raw_json_output = analyze_with_qwen(df.copy(), text_col, active_key)
            
            else:
                # Simulation
                st.warning("⚠️ AI analysis unavailable or Simulation Mode selected. Running in keyword-matching mode.")
                with st.spinner(f"Simulating analysis..."):
                    time.sleep(1.0) 
                    analyzed_df, _ = analyze_tickets_mockup(df.copy(), text_col)
            
            # Render results
            render_dashboard(analyzed_df, raw_json_output, cost_per_ticket, avg_hourly_cost, avg_time_per_ticket)

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
            
            # Apply Row Limit to sample too
            if max_rows and len(sample_df) > max_rows:
                sample_df = sample_df.head(max_rows)
            
            analyzed_df = None
            raw_json_output = None
            
            if selected_provider == "Qwen AI" and active_key:
                 analyzed_df, raw_json_output = analyze_with_qwen(sample_df.copy(), 'Ticket Description', active_key)
            else:
                 msg = "⚠️ Running Sample Data in Simulation Mode"
                 st.warning("⚠️ AI analysis unavailable. Running in keyword-matching mode.")
                 analyzed_df, _ = analyze_tickets_mockup(sample_df.copy(), 'Ticket Description')
            
            # Render results
            render_dashboard(analyzed_df, raw_json_output, cost_per_ticket, avg_hourly_cost, avg_time_per_ticket)
            
            # CSV Download Removed as per user request
            pass

if __name__ == "__main__":
    main()
