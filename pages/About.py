import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pathlib import Path
DATA_CANDIDATES = [
    Path("data/Online Retail.xlsx"),
    Path("Online Retail.xlsx"),
    Path("pages/Online Retail.xlsx"),
]

st.title("About Page: Data Visualizations for dataset for Sales revenue")

@st.cache_data
def load_data():
    df = pd.read_excel("Online Retail.xlsx")

    # Work on a copy of original df
    df_clean = df.copy()

    # Drop columns we are NOT using
    df_clean = df_clean.drop(columns=["CustomerID", "Country"], errors="ignore")

    # Ensure correct data types
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'], errors="coerce")
    df_clean["Quantity"]    = pd.to_numeric(df_clean["Quantity"], errors="coerce")
    df_clean["UnitPrice"]   = pd.to_numeric(df_clean["UnitPrice"], errors="coerce")

    # Remove missing essentials
    df_clean = df_clean.dropna(subset=["InvoiceDate", "Quantity", "UnitPrice"])

    # Remove returns / free items
    before = len(df_clean)
    df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["UnitPrice"] > 0)].copy()
    after = len(df_clean)
    print(f"Kept {after:,} rows ({after/before:.1%}) after filtering returns/free items.")

    # Create Revenue column
    df_clean['Revenue'] = df_clean['Quantity'] * df_clean['UnitPrice']

    # Time-based features
    df_clean['YearMonth'] = df_clean['InvoiceDate'].dt.to_period('M')
    df_clean['Week']      = df_clean['InvoiceDate'].dt.isocalendar().week
    df_clean['Day']       = df_clean['InvoiceDate'].dt.date
    df_clean['DayOfWeek'] = df_clean['InvoiceDate'].dt.day_name()

    # Aggregate monthly revenue
    monthly_df = (df_clean.groupby(df_clean['InvoiceDate'].dt.to_period('M'))['Revenue']
                  .sum()
                  .reset_index())

    monthly_df['InvoiceDate'] = monthly_df['InvoiceDate'].dt.to_timestamp()
    monthly_df = monthly_df.set_index('InvoiceDate').sort_index()

    print(monthly_df.head())
    print(df_clean.head())

    return df_clean

df_clean = load_data()

# 1) Revenue distribution plot
fig1 = plt.figure(figsize=(10,6))
sns.histplot(df_clean['Revenue'], bins=50, kde=False)
plt.title('Distribution of Revenue')
plt.xlabel('Revenue (£)')
plt.ylabel('Frequency')
st.pyplot(fig1)

# 2) Monthly revenue trend
fig2 = plt.figure(figsize=(10,6))
monthly_revenue = df_clean.groupby('YearMonth')['Revenue'].sum().reset_index()
monthly_revenue['YearMonth'] = monthly_revenue['YearMonth'].dt.to_timestamp()
plt.plot(monthly_revenue['YearMonth'], monthly_revenue['Revenue'], marker='o')
plt.title('Monthly Revenue Trend')
plt.xlabel('Month')
plt.ylabel('Total Revenue (£)')
plt.grid(True)
st.pyplot(fig2)

# 3) Correlation heatmap
fig3 = plt.figure(figsize=(10, 6))
corr = df_clean.select_dtypes(include=[float, int]).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
st.pyplot(fig3)

# 4) You can also add tables if needed
st.subheader("Data Summary")
st.dataframe(df_clean.describe())

# Add more visualizations from your notebook similarly.

# Optionally, you can add interaction widgets for dynamic filtering, etc.
