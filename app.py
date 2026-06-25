"""
app.py
=======
Smart Retail Sales Analysis — Streamlit Dashboard
----------------------------------------------------
A professional, interactive retail analytics dashboard featuring:
    - KPI summary cards
    - Sidebar filters (Date range, Region, Category, Segment)
    - Sales / Profit / Regional / Category analysis tabs
    - Sales prediction tool (Random Forest model)
    - Downloadable PDF-style summary report (CSV + chart bundle)

Run locally:
    streamlit run app.py

Author : Senior Data Analyst
Project: Smart Retail Sales Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io
import os
import sys
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "dashboard"))
from utils import kpi_summary, format_currency, format_percent  # noqa: E402

# ------------------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Retail Sales Analysis",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# CUSTOM CSS — distinctive theme (deep teal + warm coral accent)
# ------------------------------------------------------------------------------
PRIMARY = "#0F4C5C"      # deep teal (headers, sidebar)
ACCENT = "#E76F51"       # warm coral (highlights, CTA)
ACCENT_2 = "#2A9D8F"     # seafoam (secondary accent)
BG_CARD = "#FFFFFF"
TEXT_MUTED = "#5C6B73"

st.markdown(f"""
<style>
    .main {{ background-color: #F7F9F9; }}
    h1, h2, h3 {{ color: {PRIMARY}; font-family: 'Segoe UI', sans-serif; }}

    /* KPI card */
    .kpi-card {{
        background: linear-gradient(135deg, {BG_CARD} 0%, #F0F6F6 100%);
        border-left: 6px solid {ACCENT};
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 10px;
    }}
    .kpi-label {{
        font-size: 13px;
        color: {TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }}
    .kpi-value {{
        font-size: 28px;
        color: {PRIMARY};
        font-weight: 800;
        margin-top: 4px;
    }}
    .kpi-delta {{
        font-size: 13px;
        font-weight: 600;
        margin-top: 2px;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {PRIMARY};
    }}
    section[data-testid="stSidebar"] * {{
        color: #FFFFFF !important;
    }}
    section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {{
        color: {ACCENT};
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #EAF1F1;
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
        font-weight: 600;
        color: {PRIMARY};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {PRIMARY} !important;
        color: white !important;
    }}

    footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

sns.set_style("whitegrid")
PALETTE = [PRIMARY, ACCENT, ACCENT_2, "#F4A261", "#264653"]

# ------------------------------------------------------------------------------
# DATA LOADING (cached)
# ------------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "retail_sales_clean.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_sales_model.pkl")
MODEL_NAME_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model_name.txt")


@st.cache_data
def load_data(path):
    df = pd.read_csv(path, parse_dates=["Order Date"])
    return df


@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None


try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        "Cleaned dataset not found. Please run `notebooks/01_data_cleaning_eda.py` "
        "first to generate `data/retail_sales_clean.csv`."
    )
    st.stop()

model = load_model(MODEL_PATH)
best_model_name = "Random Forest"
if os.path.exists(MODEL_NAME_PATH):
    with open(MODEL_NAME_PATH) as f:
        best_model_name = f.read().strip()

# ------------------------------------------------------------------------------
# SIDEBAR — FILTERS
# ------------------------------------------------------------------------------
st.sidebar.markdown("## 🛍️ Smart Retail\n### Sales Analysis")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔎 Filters")

min_date, max_date = df["Order Date"].min(), df["Order Date"].max()
date_range = st.sidebar.date_input(
    "Order Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

regions = sorted(df["Region"].unique())
selected_regions = st.sidebar.multiselect("Region", regions, default=regions)

categories = sorted(df["Category"].unique())
selected_categories = st.sidebar.multiselect("Category", categories, default=categories)

segments = sorted(df["Customer Segment"].unique())
selected_segments = st.sidebar.multiselect("Customer Segment", segments, default=segments)

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit • Pandas • Scikit-learn")
st.sidebar.caption(f"Best Model: **{best_model_name}**")

# Apply filters
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
else:
    start_d, end_d = min_date, max_date

mask = (
    (df["Order Date"] >= pd.Timestamp(start_d))
    & (df["Order Date"] <= pd.Timestamp(end_d))
    & (df["Region"].isin(selected_regions))
    & (df["Category"].isin(selected_categories))
    & (df["Customer Segment"].isin(selected_segments))
)
fdf = df.loc[mask].copy()

if fdf.empty:
    st.warning("No data matches the selected filters. Please broaden your filter selection.")
    st.stop()

# ------------------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------------------
st.markdown(
    f"<h1>🛍️ Smart Retail Sales Analysis Dashboard</h1>"
    f"<p style='color:{TEXT_MUTED}; font-size:16px; margin-top:-10px;'>"
    f"End-to-end retail performance, regional insight & sales forecasting</p>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# KPI CARDS
# ------------------------------------------------------------------------------
total_sales = fdf["Sales"].sum()
total_profit = fdf["Profit"].sum()
total_orders = fdf["Order ID"].nunique()
avg_order_value = fdf["Sales"].mean()
profit_margin = (total_profit / total_sales * 100) if total_sales else 0

k1, k2, k3, k4, k5 = st.columns(5)

kpi_data = [
    (k1, "💰 Total Sales", format_currency(total_sales)),
    (k2, "📈 Total Profit", format_currency(total_profit)),
    (k3, "🧾 Total Orders", f"{total_orders:,}"),
    (k4, "🛒 Avg Order Value", f"${avg_order_value:,.2f}"),
    (k5, "📊 Profit Margin", format_percent(profit_margin)),
]
for col, label, value in kpi_data:
    col.markdown(
        f"""<div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
            </div>""",
        unsafe_allow_html=True,
    )

st.markdown("")

# ------------------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------------------
tab_sales, tab_profit, tab_region, tab_category, tab_predict, tab_report = st.tabs(
    ["📈 Sales Analysis", "💵 Profit Analysis", "🗺️ Regional Analysis",
     "📦 Category Analysis", "🤖 Prediction", "📄 Report"]
)

# ---------------- SALES ANALYSIS TAB ------------------------------------------
with tab_sales:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Monthly Sales Trend")
        monthly = fdf.groupby(fdf["Order Date"].dt.to_period("M"))["Sales"].sum()
        monthly.index = monthly.index.astype(str)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(monthly.index, monthly.values, marker="o", color=PRIMARY, linewidth=2)
        ax.fill_between(range(len(monthly)), monthly.values, color=PRIMARY, alpha=0.08)
        ax.set_ylabel("Sales ($)")
        plt.xticks(rotation=70, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Sales Distribution")
        fig, ax = plt.subplots(figsize=(5, 4.5))
        sns.histplot(fdf["Sales"], bins=30, kde=True, color=ACCENT, ax=ax)
        ax.set_xlabel("Sales ($)")
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Top 10 Products by Sales")
    top_products = fdf.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=top_products.values, y=top_products.index, palette="crest", ax=ax)
    ax.set_xlabel("Total Sales ($)")
    plt.tight_layout()
    st.pyplot(fig)

# ---------------- PROFIT ANALYSIS TAB ------------------------------------------
with tab_profit:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Monthly Profit Trend")
        monthly_p = fdf.groupby(fdf["Order Date"].dt.to_period("M"))["Profit"].sum()
        monthly_p.index = monthly_p.index.astype(str)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(monthly_p.index, monthly_p.values, marker="o", color=ACCENT, linewidth=2)
        ax.fill_between(range(len(monthly_p)), monthly_p.values, color=ACCENT, alpha=0.10)
        ax.set_ylabel("Profit ($)")
        plt.xticks(rotation=70, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Correlation Heatmap")
        corr = fdf[["Sales", "Profit", "Quantity", "Discount"]].corr()
        fig, ax = plt.subplots(figsize=(5, 4.5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Profit Margin Distribution by Category")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.boxplot(data=fdf, x="Category", y="Profit Margin (%)", palette=PALETTE, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

# ---------------- REGIONAL ANALYSIS TAB ----------------------------------------
with tab_region:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Region-wise Sales")
        region_sales = fdf.groupby("Region")["Sales"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.barplot(x=region_sales.index, y=region_sales.values, palette="mako", ax=ax)
        ax.set_ylabel("Sales ($)")
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Region-wise Profit")
        region_profit = fdf.groupby("Region")["Profit"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.barplot(x=region_profit.index, y=region_profit.values, palette="rocket", ax=ax)
        ax.set_ylabel("Profit ($)")
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Customer Segment Share")
    col3, col4 = st.columns(2)
    with col3:
        seg_sales = fdf.groupby("Customer Segment")["Sales"].sum()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(seg_sales.values, labels=seg_sales.index, autopct="%1.1f%%",
               colors=PALETTE, startangle=90,
               wedgeprops={"edgecolor": "white", "linewidth": 1.5})
        ax.set_title("Sales Share by Segment")
        st.pyplot(fig)
    with col4:
        st.dataframe(
            fdf.groupby("Customer Segment")[["Sales", "Profit"]].sum()
               .style.format("${:,.0f}"),
            use_container_width=True,
        )

# ---------------- CATEGORY ANALYSIS TAB -----------------------------------------
with tab_category:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Category-wise Sales")
        cat_sales = fdf.groupby("Category")["Sales"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.barplot(x=cat_sales.values, y=cat_sales.index, palette="Blues_r", ax=ax)
        ax.set_xlabel("Sales ($)")
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Sub-Category-wise Sales")
        subcat_sales = fdf.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.barplot(x=subcat_sales.values, y=subcat_sales.index, palette="viridis", ax=ax)
        ax.set_xlabel("Sales ($)")
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Category Performance Table")
    cat_table = fdf.groupby("Category").agg(
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Profit", "sum"),
        Orders=("Order ID", "nunique"),
        Avg_Profit_Margin=("Profit Margin (%)", "mean"),
    ).round(2).sort_values("Total_Sales", ascending=False)
    st.dataframe(cat_table.style.format({
        "Total_Sales": "${:,.0f}", "Total_Profit": "${:,.0f}",
        "Avg_Profit_Margin": "{:.1f}%"
    }), use_container_width=True)

# ---------------- PREDICTION TAB --------------------------------------------------
with tab_predict:
    st.subheader("🤖 Sales Prediction Tool")
    st.caption(
        f"Predicts order Sales value using the trained **{best_model_name}** model "
        "(trained on Quantity, Discount, Profit, Month, Category, Sub-Category, Region, and Segment)."
    )

    if model is None:
        st.warning(
            "No trained model found. Please run `notebooks/02_sales_prediction_model.py` "
            "to train and save a model to the `/models` folder."
        )
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            in_category = st.selectbox("Category", sorted(df["Category"].unique()))
            sub_options = sorted(df.loc[df["Category"] == in_category, "Sub-Category"].unique())
            in_subcategory = st.selectbox("Sub-Category", sub_options)
            in_region = st.selectbox("Region", sorted(df["Region"].unique()))
        with c2:
            in_segment = st.selectbox("Customer Segment", sorted(df["Customer Segment"].unique()))
            in_quantity = st.slider("Quantity", 1, 14, 4)
            in_month = st.slider("Order Month", 1, 12, 6)
        with c3:
            in_discount = st.slider("Discount", 0.0, 0.3, 0.1, step=0.05)
            in_profit = st.number_input("Expected Profit ($)", min_value=-500.0,
                                         max_value=3000.0, value=150.0, step=10.0)

        if st.button("🔮 Predict Sales", use_container_width=True):
            input_df = pd.DataFrame([{
                "Quantity": in_quantity,
                "Discount": in_discount,
                "Profit": in_profit,
                "Month": in_month,
                "Category": in_category,
                "Sub-Category": in_subcategory,
                "Region": in_region,
                "Customer Segment": in_segment,
            }])
            prediction = model.predict(input_df)[0]
            st.success(f"### 💲 Predicted Sales: ${prediction:,.2f}")

        st.markdown("---")
        st.subheader("Model Performance Summary")
        comparison_path = os.path.join(os.path.dirname(__file__), "models", "model_comparison.csv")
        if os.path.exists(comparison_path):
            comp_df = pd.read_csv(comparison_path)
            st.dataframe(comp_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2 Score": "{:.4f}"}),
                         use_container_width=True)
            st.caption(
                "**MAE** = Mean Absolute Error · **RMSE** = Root Mean Squared Error · "
                "**R²** = proportion of variance explained (closer to 1 is better)."
            )

# ---------------- REPORT TAB -----------------------------------------------------
with tab_report:
    st.subheader("📄 Download Analysis Report")
    st.write(
        "Generate a CSV summary report of the currently filtered data, "
        "including KPIs and category / region breakdowns — ready to share or "
        "attach to a stakeholder email."
    )

    summary_lines = [
        ["Report Generated On", datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["Date Range", f"{start_d} to {end_d}"],
        ["Total Sales", f"${total_sales:,.2f}"],
        ["Total Profit", f"${total_profit:,.2f}"],
        ["Total Orders", total_orders],
        ["Average Order Value", f"${avg_order_value:,.2f}"],
        ["Overall Profit Margin", f"{profit_margin:.2f}%"],
    ]
    summary_df = pd.DataFrame(summary_lines, columns=["Metric", "Value"])

    category_summary = fdf.groupby("Category").agg(
        Total_Sales=("Sales", "sum"), Total_Profit=("Profit", "sum")
    ).reset_index()

    region_summary = fdf.groupby("Region").agg(
        Total_Sales=("Sales", "sum"), Total_Profit=("Profit", "sum")
    ).reset_index()

    buffer = io.StringIO()
    buffer.write("SMART RETAIL SALES ANALYSIS — SUMMARY REPORT\n\n")
    summary_df.to_csv(buffer, index=False)
    buffer.write("\nCATEGORY BREAKDOWN\n")
    category_summary.to_csv(buffer, index=False)
    buffer.write("\nREGION BREAKDOWN\n")
    region_summary.to_csv(buffer, index=False)

    st.download_button(
        label="⬇️ Download Summary Report (CSV)",
        data=buffer.getvalue(),
        file_name=f"retail_sales_report_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.download_button(
        label="⬇️ Download Filtered Raw Data (CSV)",
        data=fdf.to_csv(index=False),
        file_name="filtered_retail_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("Preview: Filtered Dataset")
    st.dataframe(fdf.head(50), use_container_width=True)

# ------------------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "📊 Smart Retail Sales Analysis Dashboard | Built with Streamlit, Pandas, Scikit-learn "
    "| © 2026 — Portfolio Project"
)
