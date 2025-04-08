import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import f_oneway, chi2_contingency

# Set page config
st.set_page_config(
    page_title="NYC Green Taxi Analysis",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #e8f5e9;
    }
    h1 {
        color: #2e7d32;
    }
    .st-bq {
        border-left: 5px solid #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Load data function with caching
@st.cache_data
def load_data():
    df = pd.read_parquet("D:\SEM-VI\PA\project-Aug-2023\green_tripdata_2023-08.parquet")
    
    # Preprocessing steps from notebook
    df.drop(columns=['ehail_fee', 'fare_amount'], inplace=True, errors='ignore')
    df['trip_duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
    df['weekday'] = df['lpep_dropoff_datetime'].dt.day_name()
    df['hourofday'] = df['lpep_dropoff_datetime'].dt.hour
    
    # Add week of month calculation
    df['day_of_month'] = df['lpep_pickup_datetime'].dt.day
    df['week_of_month'] = ((df['day_of_month'] - 1) // 7) + 1
    df['week_name'] = df['week_of_month'].map({
        1: "1st Week",
        2: "2nd Week",
        3: "3rd Week",
        4: "4th Week",
        5: "5th Week"  # In case the month has a partial 5th week
    })
    
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(method='ffill', inplace=True)
    
    # Map payment types to meaningful names
    payment_mapping = {
        1: "Credit Card",
        2: "Cash",
        3: "No Charge",
    }
    df['payment_type_name'] = df['payment_type'].map(payment_mapping)
    
    # Map trip types to meaningful names if column exists
    if 'trip_type' in df.columns:
        trip_type_mapping = {
            1: "Street-hail",
            2: "Dispatch"
        }
        df['trip_type_name'] = df['trip_type'].map(trip_type_mapping)
    
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filter Data")

# Week selector (new filter)
week_options = ["All Weeks", "1st Week", "2nd Week", "3rd Week", "4th Week", "5th Week"]
selected_week = st.sidebar.selectbox(
    "Select Week",
    options=week_options,
    index=0
)

# Weekday selector (single selection)
weekday_options = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
selected_weekday = st.sidebar.selectbox(
    "Select Weekday",
    options=weekday_options,
    index=0
)

# Payment type selector (single selection)
payment_options = ["Credit Card", "Cash", "No Charge"]
selected_payment = st.sidebar.selectbox(
    "Select Payment Method",
    options=payment_options,
    index=0
)

# Passenger count selector (single selection)
passenger_options = sorted(df['passenger_count'].unique().tolist())
selected_passenger = st.sidebar.selectbox(
    "Select Passenger Count",
    options=passenger_options,
    index=0
)

# Trip type selector (single selection) - if column exists
if 'trip_type_name' in df.columns:
    trip_type_options = ["Street-hail", "Dispatch"]
    selected_trip_type = st.sidebar.selectbox(
        "Select Trip Type",
        options=trip_type_options,
        index=0
    )

# Hour selector (single selection)
hour_options = sorted(df['hourofday'].unique().tolist())
selected_hour = st.sidebar.selectbox(
    "Select Hour of Day",
    options=hour_options,
    index=0
)

# Apply filters
filtered_df = df.copy()

# Apply week filter (new filter)
if selected_week != "All Weeks":
    filtered_df = filtered_df[filtered_df['week_name'] == selected_week]

# Apply other filters (single selection)
filtered_df = filtered_df[filtered_df['weekday'] == selected_weekday]
filtered_df = filtered_df[filtered_df['payment_type_name'] == selected_payment]
filtered_df = filtered_df[filtered_df['passenger_count'] == selected_passenger]

if 'trip_type_name' in df.columns:
    filtered_df = filtered_df[filtered_df['trip_type_name'] == selected_trip_type]

filtered_df = filtered_df[filtered_df['hourofday'] == selected_hour]

# Main content
st.title("🚕 NYC Green Taxi Trip Analysis - August 2023")
st.markdown("Explore patterns and insights from NYC Green Taxi trips in August 2023")

# Key metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Trips", len(filtered_df))
col2.metric("Avg. Trip Distance", f"{filtered_df['trip_distance'].mean():.2f} miles")
col3.metric("Avg. Trip Duration", f"{filtered_df['trip_duration'].mean():.2f} mins")
col4.metric("Avg. Total Amount", f"${filtered_df['total_amount'].mean():.2f}")

st.markdown("---")

# Visualization section
st.header("Data Visualizations")

# Tab layout
tab1, tab2, tab3, tab4 = st.tabs(["Trip Patterns", "Financial Analysis", "Statistical Tests", "Raw Data"])

with tab1:
    st.subheader("Trip Patterns by Time")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Trips by weekday
        weekday_counts = filtered_df['weekday'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        fig = px.bar(
            weekday_counts,
            title="Trips by Weekday",
            labels={'index': 'Weekday', 'value': 'Number of Trips'},
            color=weekday_counts.index,
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Trips by hour
        hour_counts = filtered_df['hourofday'].value_counts().sort_index()
        fig = px.line(
            hour_counts,
            title="Trips by Hour of Day",
            labels={'index': 'Hour of Day', 'value': 'Number of Trips'},
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Trip duration vs distance
    st.subheader("Trip Duration vs Distance")
    sample_df = filtered_df.sample(min(1000, len(filtered_df)))  # Sample for performance
    fig = px.scatter(
        sample_df,
        x='trip_distance',
        y='trip_duration',
        color='weekday',
        title="Trip Duration vs Distance",
        labels={'trip_distance': 'Distance (miles)', 'trip_duration': 'Duration (minutes)'},
        hover_data=['total_amount']
    )
    st.plotly_chart(fig, use_container_width=True)

    # Add a new visualization for trips by week
    st.subheader("Trips by Week of Month")
    week_counts = filtered_df['week_name'].value_counts().sort_index()
    fig = px.bar(
        week_counts,
        title="Trip Distribution by Week of Month",
        labels={'index': 'Week', 'value': 'Number of Trips'},
        color=week_counts.index,
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Financial Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Payment type distribution
        payment_counts = filtered_df['payment_type_name'].value_counts()
        fig = px.pie(
            payment_counts,
            values=payment_counts.values,
            names=payment_counts.index,
            title="Payment Type Distribution",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Trip type distribution
        if 'trip_type_name' in filtered_df.columns:
            trip_counts = filtered_df['trip_type_name'].value_counts()
            fig = px.pie(
                trip_counts,
                values=trip_counts.values,
                names=trip_counts.index,
                title="Trip Type Distribution",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Average total amount by weekday
    st.subheader("Average Total Amount by Weekday")
    avg_amount = filtered_df.groupby('weekday')['total_amount'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    fig = px.bar(
        avg_amount,
        title="Average Total Amount by Weekday",
        labels={'index': 'Weekday', 'value': 'Average Total Amount ($)'},
        color=avg_amount.index,
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # New visualization: Average total amount by week of month
    st.subheader("Average Total Amount by Week")
    avg_amount_by_week = filtered_df.groupby('week_name')['total_amount'].mean().reindex(["1st Week", "2nd Week", "3rd Week", "4th Week", "5th Week"])
    fig = px.bar(
        avg_amount_by_week,
        title="Average Total Amount by Week of Month",
        labels={'index': 'Week', 'value': 'Average Total Amount ($)'},
        color=avg_amount_by_week.index,
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    numeric_cols = ['trip_distance', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 
                   'improvement_surcharge', 'congestion_surcharge', 'trip_duration', 
                   'passenger_count', 'total_amount']
    corr = filtered_df[numeric_cols].corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='Viridis',
        title="Correlation Between Numeric Features"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Statistical Tests")
    
    # Let users choose which filters to apply to tests
    col1, col2, col3 = st.columns(3)
    with col1:
        test_weekday = st.selectbox(
            "Weekday for tests (select 'All' for no filter)", 
            ['All'] + weekday_options
        )
    with col2:
        test_payment = st.selectbox(
            "Payment type for tests (select 'All' for no filter)", 
            ['All'] + payment_options
        )
    with col3:
        test_week = st.selectbox(
            "Week for tests (select 'All' for no filter)",
            ['All'] + week_options[1:]  # Skip the "All Weeks" option
        )
    
    # Apply selected filters
    test_df = df.copy()
    if test_weekday != 'All':
        test_df = test_df[test_df['weekday'] == test_weekday]
    if test_payment != 'All':
        test_df = test_df[test_df['payment_type_name'] == test_payment]
    if test_week != 'All':
        test_df = test_df[test_df['week_name'] == test_week]
    
    st.markdown("""
    ### ANOVA Tests
    These tests examine whether there are statistically significant differences in means between groups.
    """)
    
    # ANOVA: total_amount ~ trip_type
    if 'trip_type_name' in test_df.columns:
        trip_type_groups = [g['total_amount'] for _, g in test_df.groupby('trip_type_name')]
        
        if len(trip_type_groups) >= 2 and all(len(group) > 0 for group in trip_type_groups):
            f_stat, p_value = f_oneway(*trip_type_groups)
            st.markdown(f"""
            **Total Amount by Trip Type**
            - F-statistic: {f_stat:.2f}
            - p-value: {p_value:.4f}
            - {'Significant difference' if p_value < 0.05 else 'No significant difference'} between trip types
            """)
        else:
            st.warning("Cannot perform ANOVA test on trip types - need at least 2 groups with data")
    
    # ANOVA: total_amount ~ weekday
    weekday_groups = [g['total_amount'] for _, g in test_df.groupby('weekday')]
    
    if len(weekday_groups) >= 2 and all(len(group) > 0 for group in weekday_groups):
        f_stat, p_value = f_oneway(*weekday_groups)
        st.markdown(f"""
        **Total Amount by Weekday**
        - F-statistic: {f_stat:.2f}
        - p-value: {p_value:.4f}
        - {'Significant difference' if p_value < 0.05 else 'No significant difference'} between weekdays
        """)
    else:
        st.warning(f"Cannot perform ANOVA test on weekdays - need at least 2 weekdays with data (found {len(weekday_groups)})")
    
    # New ANOVA test: total_amount ~ week_of_month
    week_groups = [g['total_amount'] for _, g in test_df.groupby('week_name')]
    
    if len(week_groups) >= 2 and all(len(group) > 0 for group in week_groups):
        f_stat, p_value = f_oneway(*week_groups)
        st.markdown(f"""
        **Total Amount by Week of Month**
        - F-statistic: {f_stat:.2f}
        - p-value: {p_value:.4f}
        - {'Significant difference' if p_value < 0.05 else 'No significant difference'} between weeks of the month
        """)
    else:
        st.warning(f"Cannot perform ANOVA test on weeks - need at least 2 weeks with data (found {len(week_groups)})")
    
    # Chi-square test
    if 'trip_type_name' in test_df.columns:
        st.markdown("""
        ### Chi-square Test
        This test examines the association between trip type and payment type.
        """)
        
        contingency = pd.crosstab(test_df['trip_type_name'], test_df['payment_type_name'])
        
        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            chi2, p, _, _ = chi2_contingency(contingency)
            st.markdown(f"""
            **Trip Type vs Payment Type**
            - Chi-square statistic: {chi2:.2f}
            - p-value: {p:.4f}
            - {'Significant association' if p < 0.05 else 'No significant association'} between trip type and payment type
            """)
            st.dataframe(contingency)
        else:
            st.warning(f"Cannot perform chi-square test - need at least 2 categories in both dimensions (current shape: {contingency.shape})")

with tab4:
    st.subheader("Raw Data Preview")
    st.dataframe(filtered_df.head(100))
    
    # Download button
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_green_taxi_data.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
**Data Source**: NYC Green Taxi Trip Data - August 2023  
**Analysis**: Exploratory analysis of trip patterns, financial metrics, and statistical relationships
""")