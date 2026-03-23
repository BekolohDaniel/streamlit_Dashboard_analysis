import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import IsolationForest

# =========================
# App Config
# =========================
st.set_page_config(layout="wide", page_title="Transaction & Fraud Dashboard")

# =========================
# Load Data
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("Dataset.csv")

transactions = load_data()

st.title("💳 Transaction Analytics & Fraud Detection Dashboard")

# =========================
# Data Preparation
# =========================
transactions['TransactionStartTime'] = pd.to_datetime(transactions['TransactionStartTime'])
transactions['Date'] = transactions['TransactionStartTime'].dt.date
transactions['Hour'] = transactions['TransactionStartTime'].dt.hour
transactions['Day'] = transactions['TransactionStartTime'].dt.day
transactions['Month'] = transactions['TransactionStartTime'].dt.month

# =========================
# Sidebar Filters
# =========================
st.sidebar.header("🔎 Filters")

# ---- Date Filter ----
start_date = transactions['Date'].min()
end_date = transactions['Date'].max()

selected_dates = st.sidebar.date_input(
    "Select Date Range",
    [start_date, end_date]
)

if len(selected_dates) == 2:
    transactions = transactions[
        (transactions['Date'] >= selected_dates[0]) &
        (transactions['Date'] <= selected_dates[1])
    ]

# ---- Product Category ----
selected_categories = st.sidebar.multiselect(
    "Product Category",
    transactions['ProductCategory'].unique(),
    default=transactions['ProductCategory'].unique()
)

transactions = transactions[
    transactions['ProductCategory'].isin(selected_categories)
]

# ---- Channel Filter ----
selected_channels = st.sidebar.multiselect(
    "Channel",
    transactions['ChannelId'].unique(),
    default=transactions['ChannelId'].unique()
)

transactions = transactions[
    transactions['ChannelId'].isin(selected_channels)
]

# ---- Pricing Strategy ----
selected_pricing = st.sidebar.multiselect(
    "Pricing Strategy",
    transactions['PricingStrategy'].unique(),
    default=transactions['PricingStrategy'].unique()
)

transactions = transactions[
    transactions['PricingStrategy'].isin(selected_pricing)
]

# =========================
# KPIs
# =========================
st.subheader("📌 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Transactions", len(transactions))
col2.metric("Total Amount", f"{transactions['Amount'].sum():,.0f}")
col3.metric("Total Value", f"{transactions['Value'].sum():,.0f}")
col4.metric("Fraud Cases", transactions['FraudResult'].sum())

# =========================
# Time Trend
# =========================
st.subheader("📈 Transactions Over Time")

time_trend = transactions.groupby('Date')['Amount'].sum().reset_index()

fig_time = px.line(
    time_trend,
    x='Date',
    y='Amount',
    title="Total Transaction Amount Over Time"
)

st.plotly_chart(fig_time, use_container_width=True)

# =========================
# Distribution
# =========================
st.subheader("📊 Amount Distribution")

fig_hist = px.histogram(
    transactions,
    x='Amount',
    nbins=30,
    color='ProductCategory'
)

st.plotly_chart(fig_hist, use_container_width=True)

# =========================
# Category Analysis
# =========================
st.subheader("📊 Average Amount by Product Category")

category_analysis = transactions.groupby('ProductCategory')['Amount'].mean().reset_index()

fig_cat = px.bar(
    category_analysis,
    x='ProductCategory',
    y='Amount',
    color='ProductCategory'
)

st.plotly_chart(fig_cat, use_container_width=True)

# =========================
# Channel Analysis
# =========================
st.subheader("📡 Channel Performance")

channel_analysis = transactions.groupby('ChannelId')['Amount'].sum().reset_index()

fig_channel = px.bar(
    channel_analysis,
    x='ChannelId',
    y='Amount',
    color='ChannelId'
)

st.plotly_chart(fig_channel, use_container_width=True)

# =========================
# Hourly Pattern
# =========================
st.subheader("⏰ Transactions by Hour")

hourly = transactions.groupby('Hour')['Amount'].count().reset_index()

fig_hour = px.bar(
    hourly,
    x='Hour',
    y='Amount',
    title="Number of Transactions per Hour"
)

st.plotly_chart(fig_hour, use_container_width=True)

# =========================
# Correlation Heatmap
# =========================
st.subheader("🔥 Correlation Heatmap")

numeric_cols = ['Amount', 'Value', 'PricingStrategy', 'Hour', 'Day']

corr = transactions[numeric_cols].corr()

fig_corr, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)

st.pyplot(fig_corr)

# =========================
# Fraud Analysis (REAL DATA)
# =========================
st.subheader("🚨 Actual Fraud Analysis")

fraud_distribution = transactions['FraudResult'].value_counts().reset_index()
fraud_distribution.columns = ['Fraud', 'Count']

fig_fraud_real = px.pie(
    fraud_distribution,
    names='Fraud',
    values='Count',
    title="Actual Fraud vs Normal"
)

st.plotly_chart(fig_fraud_real, use_container_width=True)

# =========================
# AI Fraud Detection
# =========================
st.subheader("🤖 AI Fraud Detection")

contamination = st.slider("Fraud Sensitivity", 0.01, 0.10, 0.02)

model = IsolationForest(
    contamination=contamination,
    random_state=42
)

transactions['Anomaly'] = model.fit_predict(transactions[numeric_cols])
transactions['PredictedFraud'] = transactions['Anomaly'].apply(lambda x: 1 if x == -1 else 0)

predicted_fraud_count = transactions['PredictedFraud'].sum()

st.write(f"🔍 AI Detected Fraud Cases: **{predicted_fraud_count}**")

fig_fraud_ai = px.pie(
    transactions,
    names='PredictedFraud',
    title="AI Fraud Detection"
)

st.plotly_chart(fig_fraud_ai, use_container_width=True)

# =========================
# Pricing Strategy Analysis
# =========================
st.subheader("💰 Pricing Strategy Performance")

pricing_analysis = transactions.groupby('PricingStrategy')['Amount'].sum().reset_index()

fig_price = px.bar(
    pricing_analysis,
    x='PricingStrategy',
    y='Amount',
    color='PricingStrategy'
)

st.plotly_chart(fig_price, use_container_width=True)

# =========================
# Raw Data
# =========================
with st.expander("👀 View Raw Data"):
    st.dataframe(transactions)

# =========================
# Download
# =========================
csv = transactions.to_csv(index=False).encode('utf-8')

st.download_button(
    "⬇️ Download Filtered Data",
    csv,
    "transactions_filtered.csv",
    "text/csv"
)
