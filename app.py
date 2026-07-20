import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- Page Configuration ---
st.set_page_config(page_title="Sales & Feedback Dashboard", layout="wide", page_icon="📈")

# --- Load Data ---
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'sales_feedback_data.csv')
    if not os.path.exists(file_path):
        st.error("Data file not found. Please run 'python data/generate_data.py' first.")
        st.stop()
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# --- Sidebar Filters ---
st.sidebar.title("Dashboard Filters")
st.sidebar.markdown("Filter the massive dataset dynamically.")

selected_region = st.sidebar.multiselect("Select Region", options=df['Region'].unique(), default=df['Region'].unique())
selected_category = st.sidebar.multiselect("Select Category", options=df['Category'].unique(), default=df['Category'].unique())
selected_sentiment = st.sidebar.multiselect("Select Sentiment", options=df['Sentiment_Label'].unique(), default=df['Sentiment_Label'].unique())

# Apply filters
filtered_df = df[
    (df['Region'].isin(selected_region)) & 
    (df['Category'].isin(selected_category)) &
    (df['Sentiment_Label'].isin(selected_sentiment))
]

# --- Main Dashboard ---
st.title("📊 Global Sales & Customer Feedback Dashboard")
st.markdown("Analyzing 100,000+ customer interactions using NLP (TextBlob) and Plotly.")

# --- KPIs ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Total Records", value=f"{len(filtered_df):,}")
with col2:
    total_sales = filtered_df['Sales_Amount'].sum()
    st.metric(label="Total Sales", value=f"${total_sales:,.2f}")
with col3:
    avg_rating = filtered_df['Rating'].mean()
    st.metric(label="Average Rating", value=f"{avg_rating:.2f} / 5.0")
with col4:
    avg_polarity = filtered_df['Sentiment_Polarity'].mean()
    st.metric(label="Avg Text Sentiment", value=f"{avg_polarity:.2f} (Polarity)")

st.divider()

# --- Charts ---
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Sales Volume Over Time")
    # Group by month to avoid over-crowded lines with 100k rows
    sales_time = filtered_df.groupby(filtered_df['Date'].dt.to_period('M')).size().reset_index(name='Transaction_Count')
    sales_time['Date'] = sales_time['Date'].dt.to_timestamp()
    fig1 = px.line(sales_time, x='Date', y='Transaction_Count', template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)

with col_chart2:
    st.subheader("NLP Sentiment Breakdown")
    sentiment_counts = filtered_df['Sentiment_Label'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    # Match colors logically
    color_map = {'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
    fig2 = px.pie(sentiment_counts, names='Sentiment', values='Count', hole=0.4, 
                  color='Sentiment', color_discrete_map=color_map)
    st.plotly_chart(fig2, use_container_width=True)

col_chart3, col_chart4 = st.columns(2)

with col_chart3:
    st.subheader("Revenue by Product Category")
    category_sales = filtered_df.groupby('Category')['Sales_Amount'].sum().reset_index()
    fig3 = px.bar(category_sales, x='Category', y='Sales_Amount', color='Category', template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

with col_chart4:
    st.subheader("Sentiment Polarity Distribution")
    fig4 = px.histogram(filtered_df, x='Sentiment_Polarity', nbins=50, 
                        color='Sentiment_Label', color_discrete_map=color_map, template="plotly_white")
    st.plotly_chart(fig4, use_container_width=True)

# --- Raw Data Expander ---
with st.expander("View Raw Data Snippet"):
    st.dataframe(filtered_df[['Date', 'Region', 'Category', 'Sales_Amount', 'Rating', 'Feedback', 'Sentiment_Label', 'Sentiment_Polarity']].head(500))
