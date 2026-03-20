import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Page setup (wide layout for better dashboard view)
st.set_page_config(layout="wide")

# =========================
# 1. Load Data (with caching)
# =========================
@st.cache_data
def load_data():
    # Load dataset once and cache it for speed
    return pd.read_csv("Dataset.csv")

df = load_data()

st.title("📊 Interactive Financial Transactions & Fraud Dashboard")

# =========================
# 2. Sidebar Filters
# =========================
st.sidebar.header("🔍 Filters")

# ---- Date Filter ----
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])

    date_min = df['Date'].min()
    date_max = df['Date'].max()

    date_range = st.sidebar.date_input(
        "Select Date Range",
        [date_min, date_max]
    )

    if len(date_range) == 2:
        df = df[
            (df['Date'] >= pd.to_datetime(date_range[0])) &
            (df['Date'] <= pd.to_datetime(date_range[1]))
        ]

# ---- Categorical Filters ----
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

for col in categorical_cols:
    values = df[col].dropna().unique().tolist()

    selected = st.sidebar.multiselect(
        f"Filter {col}",
        values,
        default=values[:5] if len(values) > 5 else values  # 👈 limit default selection
    )

    if selected:
        df = df[df[col].isin(selected)]

# =========================
# 3. Variable Selection
# =========================
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

col_x = st.sidebar.selectbox("X (Category)", categorical_cols)
col_y = st.sidebar.selectbox("Y (Numeric)", numeric_cols)
col_color = st.sidebar.selectbox("Color (Optional)", [None] + categorical_cols)

# =========================
# 4. Charts
# =========================

# ---- Time Trend ----
if 'Date' in df.columns:
    st.subheader("📈 Time Trend")

    line_data = df.groupby('Date')[col_y].mean().reset_index()

    fig_line = px.line(
        line_data,
        x='Date',
        y=col_y,
        title=f"Average {col_y} Over Time"
    )

    st.plotly_chart(fig_line, use_container_width=True)

# ---- Histogram ----
st.subheader("📊 Distribution")

bins = st.slider("Select number of bins", 10, 100, 30)  # 👈 NEW interactive feature

fig_hist = px.histogram(
    df,
    x=col_y,
    color=col_color,
    nbins=bins
)

st.plotly_chart(fig_hist, use_container_width=True)

# ---- Bar Chart ----
st.subheader("📊 Average by Category")

agg_data = df.groupby(col_x)[col_y].mean().reset_index()

fig_bar = px.bar(
    agg_data,
    x=col_x,
    y=col_y,
    color=col_x,
    title=f"Average {col_y} by {col_x}"
)

st.plotly_chart(fig_bar, use_container_width=True)

# ---- Correlation Heatmap ----
st.subheader("🔥 Correlation Heatmap")

corr = df[numeric_cols].corr()

fig_corr, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)

st.pyplot(fig_corr)

# ---- Pie Chart ----
if col_x:
    st.subheader("🥧 Category Distribution")

    fig_pie = px.pie(
        df,
        names=col_x,
        title=f"Distribution of {col_x}"
    )

    st.plotly_chart(fig_pie, use_container_width=True)

# =========================
# 5. Table Analysis
# =========================
st.subheader("📋 Table Analysis")

groupby_col = st.selectbox("Group by", categorical_cols)

agg_col = st.multiselect(
    "Select columns to analyze",
    numeric_cols,
    default=numeric_cols[:1]
)

if groupby_col and agg_col:
    result = df.groupby(groupby_col)[agg_col].agg(['mean', 'sum', 'count']).round(2)
    st.dataframe(result)

# =========================
# 6. Raw Data Preview
# =========================
with st.expander("👀 Show Raw Data"):
    st.dataframe(df)

# =========================
# 7. Download Filtered Data
# =========================
csv = df.to_csv(index=False).encode('utf-8')

st.download_button(
    "⬇️ Download Filtered Data",
    csv,
    "filtered_transactions.csv",
    "text/csv"
)
