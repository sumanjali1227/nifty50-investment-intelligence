# 🧠 AI-Powered Investment Intelligence Platform

### Data-Driven Investment Intelligence Using NIFTY-50 Market Data

An end-to-end AI-powered investment intelligence platform built using historical NIFTY-50 stock market data from the National Stock Exchange (NSE) of India. The platform combines machine learning, portfolio optimization, risk analytics, explainable AI, and interactive visualization to transform raw market data into actionable investment insights.

---

# 📌 Project Overview

Financial markets generate vast amounts of data every day, making it difficult for investors to identify meaningful patterns and make informed decisions.

This project addresses that challenge by developing a complete investment intelligence platform capable of:

* Analyzing historical stock performance
* Forecasting future stock prices
* Generating BUY / HOLD / SELL recommendations
* Constructing investor-specific portfolios
* Assessing portfolio and stock risk
* Providing explainable investment insights
* Supporting data-driven investment decisions

The platform was developed using only the datasets provided in the competition guidelines.

---

# 🎯 Objectives

The system helps investors:

* Analyze historical stock behavior
* Identify investment opportunities
* Compare risk-return tradeoffs
* Construct optimized portfolios
* Understand portfolio risk exposure
* Generate actionable investment insights

---

# 📊 Dataset

### Primary Dataset

**NIFTY-50 Stock Market Dataset**

Coverage:

* January 2000 – April 2021
* 49 NIFTY-50 constituent companies
* Multiple sectors including:

  * Banking
  * Information Technology
  * Energy
  * Consumer Goods
  * Pharmaceuticals
  * Financial Services
  * Infrastructure
  * Metals
  * Automobile

### Features Available

* Open Price
* High Price
* Low Price
* Close Price
* Volume
* Turnover
* Company Metadata
* Sector Information

---

# ⚙️ Project Pipeline

The project was implemented in six phases.

## Phase 1: Exploratory Data Analysis

Performed extensive analysis of:

* Historical stock performance
* Sector-wise trends
* Correlation analysis
* Return distributions
* Volatility patterns
* Market behavior over time

---

## Phase 2: Feature Engineering

Generated technical indicators including:

### Trend Indicators

* MA20
* MA50
* MA200
* EMA

### Momentum Indicators

* RSI
* MACD
* Momentum

### Volatility Indicators

* Bollinger Bands
* Rolling Volatility
* Daily Returns

These engineered features form the input for the predictive models.

---

## Phase 3: Stock Prediction Engine

### Machine Learning Model

* XGBoost Regressor

### Objectives

* Predict future stock prices
* Estimate expected returns
* Generate investment signals

### Outputs

* Predicted Price
* Predicted Return (%)
* BUY Recommendation
* HOLD Recommendation
* SELL Recommendation

### Evaluation Metrics

* RMSE
* R² Score
* Directional Accuracy

---

## Phase 4: Portfolio Construction Module

Three investor-specific portfolios were created:

### Conservative Portfolio

Designed for:

* Low risk tolerance
* Capital preservation

### Balanced Portfolio

Designed for:

* Moderate risk tolerance
* Balanced growth and risk

### Aggressive Portfolio

Designed for:

* High risk tolerance
* Maximum growth potential

Portfolio recommendations are generated using quantitative return-risk analysis.

---

## Phase 5: Risk Assessment Module

The platform evaluates risk for both individual stocks and portfolios.

### Risk Metrics

* Annual Return
* Volatility
* Sharpe Ratio
* Sortino Ratio
* Maximum Drawdown
* Value at Risk (VaR)
* Conditional Value at Risk (CVaR)

These metrics help investors understand downside risk and risk-adjusted performance.

---

## Phase 6: Interactive Dashboard

A complete Streamlit-based investment intelligence dashboard was developed.

### Dashboard Modules

#### 📈 Overview Dashboard

* Market summary
* Sector analysis
* Key statistics

#### 🔍 Stock Explorer

* Historical price visualization
* Technical indicator analysis
* Stock-level insights

#### 🤖 Predictor & Signals

* Future price forecasts
* BUY / HOLD / SELL recommendations
* Model performance metrics

#### 💼 Portfolio Builder

* Portfolio comparison
* Allocation visualization
* Investor profile selection

#### ⚠️ Risk Dashboard

* Risk metric exploration
* Risk-return analysis
* Portfolio risk comparison

---

# 🔍 Explainability

The platform incorporates explainability through:

* Feature importance analysis
* Technical indicator interpretation
* Transparent signal generation
* Quantitative portfolio justification

This enables users to understand why recommendations are generated rather than treating the model as a black box.

---

# 📁 Project Structure

```text
NIFTY50_Project/
│
├── app.py
│
├── 01_EDA_and_Feature_Engineering.ipynb
├── 02_Stock_Predictor.ipynb
├── 03_Portfolio_Construction.ipynb
├── 04_Risk_Assessment.ipynb
│
├── nifty50_processed_data.csv
├── nifty50_signals.csv
├── nifty50_model_results.csv
├── nifty50_portfolios.csv
├── nifty50_portfolio_performance.csv
├── nifty50_stock_risk.csv
├── nifty50_portfolio_risk.csv
│
├── README.md
├── requirements.txt
└── report.pdf
```

---

# 🚀 Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

The application will open automatically in your browser.

Default URL:

```text
http://localhost:8501
```

---

# 🔁 Reproducing Results

Execute notebooks in the following order:

```text
1. 01_EDA_and_Feature_Engineering.ipynb
2. 02_Stock_Predictor.ipynb
3. 03_Portfolio_Construction.ipynb
4. 04_Risk_Assessment.ipynb
```

The notebooks generate all output CSV files used by the dashboard.

---

# 🛠️ Technology Stack

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* XGBoost
* Scikit-Learn

### Portfolio Optimization

* SciPy

### Explainability

* SHAP

### Visualization

* Plotly
* Matplotlib

### Dashboard

* Streamlit

---

# 📌 Key Outcomes

The platform successfully integrates:

* Predictive Analytics
* Portfolio Optimization
* Risk Management
* Explainable AI
* Interactive Visualization

into a single decision-support system capable of assisting investors in making data-driven investment decisions.

---

# ⚠️ Disclaimer

This project was developed for educational and competition purposes only.

All predictions, portfolio recommendations, and investment signals are based on historical market data and should not be interpreted as financial advice. Past market performance does not guarantee future results.
