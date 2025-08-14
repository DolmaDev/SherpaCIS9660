import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime
from pathlib import Path

# --- Page config ---
st.set_page_config(page_title="#SalesRevenuePredictor Simulator", page_icon="💰", layout="wide")

# --- Title & disclaimer ---
st.markdown("""
# 💰 #SalesRevenuePredictor — Daily Simulator with Analytics
**📚 Academic Notice:**  
This tool is built for academic purposes and has known limitations.  
It’s a **simple simulator** that estimates daily revenue from historical-like patterns.  
Further data analysis is provided below. **Do not** use for real business decisions.
""")


# --- Generate comprehensive mock dataset ---
@st.cache_data
def generate_mock_dataset():
    np.random.seed(42)
    
    # Generate dates for full year
    dates = pd.date_range("2024-01-01", "2024-12-31", freq='D')
    
    # Countries and products for realistic data
    countries = ['USA', 'Canada', 'UK', 'Germany', 'France', 'Australia', 'Japan', 'Brazil', 'India', 'Mexico']
    products = ['ProductA', 'ProductB', 'ProductC', 'ProductD', 'ProductE', 'ProductF', 'ProductG', 'ProductH']
    
    # Create detailed transaction data
    data = []
    
    for date in dates:
        # Season factor
        day_of_year = date.timetuple().tm_yday
        season_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Weekend factor (lower sales on weekends)
        weekend_factor = 0.7 if date.weekday() >= 5 else 1.0
        
        # Number of transactions per day (random between 50-200)
        num_transactions = np.random.randint(50, 200)
        
        for _ in range(num_transactions):
            # Generate transaction
            country = np.random.choice(countries, p=[0.3, 0.15, 0.12, 0.1, 0.08, 0.06, 0.05, 0.05, 0.05, 0.04])
            product = np.random.choice(products)
            
            # Base revenue per transaction
            base_revenue = np.random.uniform(50, 500)
            
            # Apply factors
            revenue = base_revenue * season_factor * weekend_factor
            
            # Add some randomness
            revenue *= np.random.normal(1, 0.2)
            revenue = max(revenue, 10)  # Minimum transaction value
            
            # Random hour (business hours weighted)
            raw_hour_weights = np.array([0.02]*6 + [0.05]*3 + [0.12]*8 + [0.08]*4 + [0.03]*3, dtype=float)
            hour_weights = raw_hour_weights / raw_hour_weights.sum()  # normalize to sum=1
            hour = np.random.choice(np.arange(24), p=hour_weights)
            
            data.append({
                'date': date,
                'datetime': datetime.combine(date, datetime.min.time().replace(hour=hour)),
                'country': country,
                'product': product,
                'revenue': revenue,
                'hour': hour,
                'month': date.month,
                'day_of_week': date.strftime('%A'),
                'is_weekend': date.weekday() >= 5
            })
    
    df = pd.DataFrame(data)
    return df

# Generate the dataset
dataset = generate_mock_dataset()

# --- Main prediction interface ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎯 Revenue Prediction")
    
    # Calendar dropdown for date selection
    user_date = st.date_input(
        "📅 Select a date for revenue simulation",
        value=date(2024, 6, 15),
        min_value=date(2024, 1, 1),
        max_value=date(2024, 12, 31)
    )
    
    # Format for display with day of week
    formatted_date = pd.to_datetime(user_date).strftime("%A, %B %d, %Y")
    
    # Revenue simulation logic
    day_of_year = pd.to_datetime(user_date).timetuple().tm_yday
    season_adj = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
    base_daily_revenue = 20000
    
    # Weekend adjustment
    is_weekend = pd.to_datetime(user_date).weekday() >= 5
    weekend_adj = 0.7 if is_weekend else 1.0
    
    predicted_revenue = base_daily_revenue * season_adj * weekend_adj
    
    # Get actual historical data for comparison if available
    historical_data = dataset[dataset['date'] == pd.to_datetime(user_date)]
    actual_revenue = historical_data['revenue'].sum() if not historical_data.empty else None
    
    # Display prediction
    st.metric(
        label=f"Predicted Revenue",
        value=f"${predicted_revenue:,.2f}",
        delta=f"vs ${actual_revenue:,.2f} actual" if actual_revenue else None
    )
    
    st.caption(f"📅 {formatted_date}")
    if is_weekend:
        st.caption("🏖️ Weekend - typically lower sales")

with col2:
    st.subheader("📈 Quick Stats")
    
    # Daily aggregated data
    daily_revenue = dataset.groupby('date')['revenue'].sum().reset_index()
    
    # Key metrics
    avg_daily = daily_revenue['revenue'].mean()
    max_daily = daily_revenue['revenue'].max()
    min_daily = daily_revenue['revenue'].min()
    
    col2_1, col2_2 = st.columns(2)
    with col2_1:
        st.metric("Avg Daily Revenue", f"${avg_daily:,.0f}")
        st.metric("Max Daily Revenue", f"${max_daily:,.0f}")
    with col2_2:
        st.metric("Min Daily Revenue", f"${min_daily:,.0f}")
        st.metric("Total Transactions", f"{len(dataset):,}")

# --- Data Overview Section ---
st.markdown("---")
st.subheader("📊 Dataset Overview & Analytics")

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📈 Revenue Trends", "🌍 Geographic Analysis", "📦 Product Analysis", "📋 Data Summary"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily revenue trend
        fig_daily = px.line(
            daily_revenue, 
            x='date', 
            y='revenue',
            title='Daily Revenue Trend (2024)',
            labels={'revenue': 'Revenue ($)', 'date': 'Date'}
        )
        fig_daily.update_layout(height=400)
        st.plotly_chart(fig_daily, use_container_width=True)
    
    with col2:
        # Revenue by month
        monthly_revenue = dataset.groupby('month')['revenue'].sum().reset_index()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_revenue['month_name'] = [month_names[i-1] for i in monthly_revenue['month']]
        
        fig_monthly = px.bar(
            monthly_revenue, 
            x='month_name', 
            y='revenue',
            title='Revenue by Month',
            labels={'revenue': 'Total Revenue ($)', 'month_name': 'Month'}
        )
        fig_monthly.update_layout(height=400)
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Revenue by hour
    hourly_revenue = dataset.groupby('hour')['revenue'].mean().reset_index()
    fig_hourly = px.line(
        hourly_revenue, 
        x='hour', 
        y='revenue',
        title='Average Revenue by Hour of Day',
        labels={'revenue': 'Avg Revenue ($)', 'hour': 'Hour'}
    )
    fig_hourly.update_layout(height=350)
    st.plotly_chart(fig_hourly, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        # Top countries by revenue
        country_revenue = dataset.groupby('country')['revenue'].sum().sort_values(ascending=False).reset_index()
        
        fig_countries = px.bar(
            country_revenue.head(10), 
            x='country', 
            y='revenue',
            title='Top 10 Countries by Total Revenue',
            labels={'revenue': 'Total Revenue ($)', 'country': 'Country'}
        )
        fig_countries.update_layout(height=400)
        st.plotly_chart(fig_countries, use_container_width=True)
    
    with col2:
        # Country revenue pie chart
        fig_pie = px.pie(
            country_revenue.head(8), 
            values='revenue', 
            names='country',
            title='Revenue Distribution by Top 8 Countries'
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        # Product revenue
        product_revenue = dataset.groupby('product')['revenue'].sum().sort_values(ascending=False).reset_index()
        
        fig_products = px.bar(
            product_revenue, 
            x='product', 
            y='revenue',
            title='Revenue by Product',
            labels={'revenue': 'Total Revenue ($)', 'product': 'Product'}
        )
        fig_products.update_layout(height=400)
        st.plotly_chart(fig_products, use_container_width=True)
    
    with col2:
        # Product transaction count
        product_count = dataset.groupby('product').size().sort_values(ascending=False).reset_index()
        product_count.columns = ['product', 'transaction_count']
        
        fig_count = px.bar(
            product_count, 
            x='product', 
            y='transaction_count',
            title='Transaction Count by Product',
            labels={'transaction_count': 'Number of Transactions', 'product': 'Product'}
        )
        fig_count.update_layout(height=400)
        st.plotly_chart(fig_count, use_container_width=True)

with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Summary**")
        summary_stats = {
            'Metric': ['Total Records', 'Date Range', 'Countries', 'Products', 'Avg Transaction Value', 'Total Revenue'],
            'Value': [
                f"{len(dataset):,}",
                f"{dataset['date'].min().strftime('%Y-%m-%d')} to {dataset['date'].max().strftime('%Y-%m-%d')}",
                f"{dataset['country'].nunique()}",
                f"{dataset['product'].nunique()}",
                f"${dataset['revenue'].mean():,.2f}",
                f"${dataset['revenue'].sum():,.2f}"
            ]
        }
        st.dataframe(pd.DataFrame(summary_stats), use_container_width=True, hide_index=True)
    
    with col2:
        st.write("**Revenue Statistics**")
        revenue_stats = dataset['revenue'].describe()
        stats_df = pd.DataFrame({
            'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', '25%', '50%', '75%', 'Max'],
            'Value': [f"${x:,.2f}" if x != revenue_stats['count'] else f"{x:,.0f}" 
                     for x in revenue_stats.values]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # Missing values check
    st.write("**Data Quality Check**")
    missing_data = dataset.isnull().sum()
    if missing_data.sum() == 0:
        st.success("✅ No missing values detected in the dataset")
    else:
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': (missing_data.values / len(dataset)) * 100
        })
        st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)

# --- How it works section ---
st.markdown("---")
with st.expander("ℹ️ How this simulator works"):
    st.markdown(f"""
    **About this enhanced simulator**  
    
    **Prediction Logic:**
    - **Date chosen:** {formatted_date}  
    - Starts with a baseline average daily revenue of **$20,000**
    - Applies a **seasonal adjustment** based on time of year (sinusoidal pattern)
    - Applies **weekend adjustment** (30% reduction for weekends)
    - Returns a deterministic simulation value
    
    **Dataset Features:**
    - **{len(dataset):,} transactions** across **{dataset['date'].nunique()} days** in 2024
    - **{dataset['country'].nunique()} countries** and **{dataset['product'].nunique()} products**
    - Realistic seasonal patterns and business hour distributions
    - Weekend vs weekday sales variations
    
    **Visualizations Include:**
    - Daily, monthly, and hourly revenue trends
    - Geographic revenue distribution
    - Product performance analysis
    - Statistical summaries and data quality metrics
    
    **Limitations**  
    - Uses simulated data with predetermined patterns
    - Does not account for real promotions, events, or market conditions
    - This is for learning/demo purposes only
    - **Do not use for actual business decisions**
    """)

# --- Footer ---
st.markdown("---")
st.caption("💡 This is a demonstration tool showing how to combine prediction interfaces with comprehensive data visualization using Streamlit and Plotly.")
