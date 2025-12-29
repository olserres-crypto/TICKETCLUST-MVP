# TicketClust MVP

TicketClust is an AI-powered SaaS support audit tool designed to analyze support tickets, estimate wasted budget, and recommend RAG architecture implementations.

## Installation

1.  **Clone the repository** (if applicable) or navigate to the project folder.
2.  **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the application**:
    ```bash
    streamlit run app.py
    ```
2.  **Open your browser**: The app should automatically open at `http://localhost:8501`.
3.  **Upload Data**: Drag and drop a CSV file containing a column with ticket descriptions/text.
    - *Tip: If you don't have simulated data, checking "Or use sample data" in the sidebar will allow you to download a sample CSV to test with.*

## Features

- **CSV Upload**: Easy drag-and-drop interface.
- **AI Simulation**: Simulates clustering of tickets into categories (Login, Billing, Bug, Feature, etc.) and sentiment analysis.
- **Budget Impact**: Calculates potential wasted budget based on ticket volume.
- **Consultant Report**: specific recommendations for RAG architecture based on data analysis.
