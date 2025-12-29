# Knowledge Base

This document serves as a central repository for the user's projects, summarizing their purpose, key features, and recent development status.

## 1. TicketClust
**Type:** Streamlit Web Application
**Domain:** Customer Success / Support Ticket Analysis

### Description
A tool designed to analyze customer support tickets from CSV uploads. It helps clustering tickets to identify common issues and calculates potential cost savings based on automation benchmarks.

### Key Features
-   **Data Privacy:** Processes data in-memory without persistent storage to ensure privacy.
-   **CSV Processing:** Robust validation for specific column requirements ("Ticket Description", "Ticket Subject").
-   **Analysis:** Simulates clustering and sentiment analysis.
-   **Economy Breakdown:** Calculates financial impact using industry benchmarks for deflection rates.
-   **Benchmarks:** Integrates a dictionary of automation rates and sources for credible reporting.

### Recent Work
-   Enhanced the "Economy Breakdown Display" to be more detailed and source-backed.
-   Improved simulation logic to avoid repetitive outputs.

---

## 2. Finance App (AppFinance)
**Type:** Python Application (Cloud Run)
**Domain:** Personal Finance Management

### Description
A comprehensive system for managing personal finances, synchronizing bank extracts with Google Sheets, and providing a data visualization interface.

### Key Features
-   **Architecture:** Deployed on Google Cloud Run with Docker.
-   **Data Source:** Parses bank extracts (CSV/XLSX) and syncs with valid master data in Google Sheets (`FinanceApp_DB_Prod.gsheet`).
-   **Authentication:** Google OAuth2 implementation for secure access.
-   **Duplicate Management:** Logic to detect and handle duplicate transactions (migrating from automatic to manual validation).
-   **Categorization:** Automated categorization rules (Family/Category) based on transaction descriptions.

### Recent Work
-   Fixed Google OAuth persistence and consent screen branding.
-   Transitioned duplicate management to a user-controlled UI.
-   Debugged "Cloud Mode" vs "Local Mode" data synchronization issues.

---

## 3. OlserresPortfolio
**Type:** Web Website (Planned)
**Domain:** Personal Portfolio / Showcase

### Description
A "Vitrina" (showcase) website intended to display the user's professional work and projects.

### Status
-   **Initial Setup:** Project workspace initialized.
-   **Tools:** Planned usage of a `mock_generator.py` script to generate sample data for the portfolio.
-   **Goal:** To serve as a refined display of technical capabilities.
