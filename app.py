import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Marketing Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 8rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 6rem !important;
        padding: 1rem;
        background: linear-gradient(90deg, #374151 0%, #6b7280 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        border: 1px solid rgba(255,255,255,0.1);
        overflow: hidden;
        word-wrap: break-word;
    }
    
    .kpi-card h3.kpi-title {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        font-family: 
        opacity: 0.95;
        margin: 0 0 0.15rem 0 !important;
        text-transform: uppercase;
        letter-spacing: 0.1px;
        line-height: 0.5 !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        color: white !important;
    }
    
    .kpi-card h1.kpi-value {
        font-size: 2rem !important;
        font-weight: 700 !important;
        margin: 0.1rem 0 !important;
        line-height: 0.5 !important;
        color: #ffffff !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .kpi-card p.kpi-delta {
        font-size: 0.4rem !important;
        margin: 0 !important;
        font-weight: 500 !important;
        opacity: 0.9;
        line-height: 1 !important;
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stSelectbox > div > div {
        background-color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the processed data files"""
    try:
        master_daily = pd.read_csv('master_daily.csv')
        marketing_campaigns = pd.read_csv('marketing_by_campaign.csv')
        
        # Convert date columns
        master_daily['date'] = pd.to_datetime(master_daily['date'])
        marketing_campaigns['date'] = pd.to_datetime(marketing_campaigns['date'])
        
        return master_daily, marketing_campaigns
    except FileNotFoundError as e:
        st.error(f"Data files not found. Please run clean_data.py first. Error: {e}")
        return None, None

def create_kpi_card(title, value, delta=None, format_type="number"):
    """Create a styled KPI card with uniform sizing and readable text"""
    if pd.isna(value):
        formatted_value = "N/A"
    elif format_type == "currency":
        formatted_value = f"${value:,.0f}"
    elif format_type == "percentage":
        formatted_value = f"{value:.1%}"
    elif format_type == "decimal":
        formatted_value = f"{value:.2f}"
    else:
        formatted_value = f"{value:,.0f}"
    
    delta_html = ""
    if delta is not None and not pd.isna(delta):
        delta_color = "#10b981" if delta >= 0 else "#ef4444"
        delta_symbol = "↗" if delta >= 0 else "↘"
        delta_html = f'<p class="kpi-delta" style="color: {delta_color};">{delta_symbol} {abs(delta):.1f}%</p>'
    
    st.markdown(f"""
    <div class="kpi-card">
        <h3 class="kpi-title">{title}</h3>
        <h1 class="kpi-value">{formatted_value}</h1>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header"> Marketing Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    master_daily, marketing_campaigns = load_data()
    
    if master_daily is None or marketing_campaigns is None:
        st.stop()
    
    # Sidebar filters
    st.sidebar.markdown("## 🎛️ Filters")
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "📅 Select Date Range",
        value=(master_daily['date'].min(), master_daily['date'].max()),
        min_value=master_daily['date'].min(),
        max_value=master_daily['date'].max()
    )
    
    # Channel filter
    channels = ['All'] + list(master_daily['channel'].unique())
    selected_channel = st.sidebar.selectbox("📢 Channel", channels)
    
    # Campaign filter (for campaign data)
    campaigns = ['All'] + list(marketing_campaigns['campaign'].unique())
    selected_campaign = st.sidebar.selectbox("🎯 Campaign", campaigns)
    
    # Filter data
    if len(date_range) == 2:
        master_filtered = master_daily[
            (master_daily['date'] >= pd.to_datetime(date_range[0])) &
            (master_daily['date'] <= pd.to_datetime(date_range[1]))
        ]
        campaigns_filtered = marketing_campaigns[
            (marketing_campaigns['date'] >= pd.to_datetime(date_range[0])) &
            (marketing_campaigns['date'] <= pd.to_datetime(date_range[1]))
        ]
    else:
        master_filtered = master_daily.copy()
        campaigns_filtered = marketing_campaigns.copy()
    
    if selected_channel != 'All':
        master_filtered = master_filtered[master_filtered['channel'] == selected_channel]
        campaigns_filtered = campaigns_filtered[campaigns_filtered['channel'] == selected_channel]
    
    if selected_campaign != 'All':
        campaigns_filtered = campaigns_filtered[campaigns_filtered['campaign'] == selected_campaign]
    
    # Main KPIs Row
    st.markdown("## 📈 Key Performance Indicators")
    
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    
    with kpi1:
        total_spend = master_filtered['spend'].sum()
        create_kpi_card("Total Spend", total_spend, format_type="currency")
    
    with kpi2:
        total_revenue = master_filtered['total_revenue'].sum()
        create_kpi_card("Total Revenue", total_revenue, format_type="currency")
    
    with kpi3:
        avg_roas = master_filtered['roas'].mean()
        create_kpi_card("Avg ROAS", avg_roas, format_type="decimal")
    
    with kpi4:
        total_orders = master_filtered['orders'].sum()
        create_kpi_card("Total Orders", total_orders)
    
    with kpi5:
        avg_cac = master_filtered['cac'].mean()
        create_kpi_card("Avg CAC", avg_cac, format_type="currency")
    
    with kpi6:
        avg_aov = master_filtered['aov'].mean()
        create_kpi_card("Avg AOV", avg_aov, format_type="currency")
    
    st.markdown("---")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💰 Revenue vs Spend Trend")
        
        # Aggregate by date for trend
        daily_trend = master_filtered.groupby('date').agg({
            'spend': 'sum',
            'total_revenue': 'sum',
            'attributed_revenue': 'sum'
        }).reset_index()
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=daily_trend['date'], 
            y=daily_trend['spend'],
            mode='lines+markers',
            name='Marketing Spend',
            line=dict(color='#ef4444', width=3)
        ))
        fig_trend.add_trace(go.Scatter(
            x=daily_trend['date'], 
            y=daily_trend['total_revenue'],
            mode='lines+markers',
            name='Total Revenue',
            line=dict(color='#22c55e', width=3),
            yaxis='y2'
        ))
        
        fig_trend.update_layout(
            xaxis_title="Date",
            yaxis=dict(title="Marketing Spend ($)", side="left", color='#ef4444'),
            yaxis2=dict(title="Revenue ($)", side="right", overlaying="y", color='#22c55e'),
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Channel Performance")
        
        channel_perf = master_filtered.groupby('channel').agg({
            'spend': 'sum',
            'attributed_revenue': 'sum',
            'clicks': 'sum',
            'impression': 'sum'
        }).reset_index()
        
        channel_perf['roas'] = channel_perf['attributed_revenue'] / channel_perf['spend']
        
        fig_channel = px.bar(
            channel_perf, 
            x='channel', 
            y='spend',
            color='roas',
            color_continuous_scale='viridis',
            title="Spend by Channel (colored by ROAS)"
        )
        fig_channel.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        st.plotly_chart(fig_channel, use_container_width=True)
    
    # Charts Row 2
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### 🎯 Campaign Performance")
        
        # Top 10 campaigns by spend
        top_campaigns = campaigns_filtered.groupby('campaign').agg({
            'spend': 'sum',
            'attributed_revenue': 'sum',
            'clicks': 'sum'
        }).reset_index().sort_values('spend', ascending=False).head(10)
        
        top_campaigns['roas'] = top_campaigns['attributed_revenue'] / top_campaigns['spend']
        
        fig_campaigns = px.scatter(
            top_campaigns,
            x='spend',
            y='attributed_revenue',
            size='clicks',
            color='roas',
            hover_data=['campaign'],
            color_continuous_scale='plasma',
            title="Campaign Performance (Spend vs Revenue)"
        )
        fig_campaigns.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        st.plotly_chart(fig_campaigns, use_container_width=True)
    
    with col4:
        st.markdown("### 📈 Key Metrics Over Time")
        
        # Create subplot with secondary y-axis
        fig_metrics = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add CAC line
        fig_metrics.add_trace(
            go.Scatter(x=master_filtered['date'], y=master_filtered['cac'],
                      mode='lines+markers', name='CAC', line=dict(color='#f59e0b')),
            secondary_y=False,
        )
        
        # Add AOV line
        fig_metrics.add_trace(
            go.Scatter(x=master_filtered['date'], y=master_filtered['aov'],
                      mode='lines+markers', name='AOV', line=dict(color='#8b5cf6')),
            secondary_y=True,
        )
        
        fig_metrics.update_xaxes(title_text="Date")
        fig_metrics.update_yaxes(title_text="CAC ($)", secondary_y=False)
        fig_metrics.update_yaxes(title_text="AOV ($)", secondary_y=True)
        fig_metrics.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    # Detailed Tables
    st.markdown("---")
    st.markdown("## 📋 Detailed Data")
    
    tab1, tab2 = st.tabs(["📊 Daily Summary", "🎯 Campaign Details"])
    
    with tab1:
        st.markdown("### Daily Performance Summary")
        
        # Prepare display data
        display_master = master_filtered.copy()
        display_master = display_master.round(2)
        display_master['date'] = display_master['date'].dt.strftime('%Y-%m-%d')
        
        # Select key columns for display
        display_cols = ['date', 'channel', 'spend', 'total_revenue', 'attributed_revenue', 
                       'orders', 'new_customers', 'ctr', 'cpc', 'roas', 'cac', 'aov']
        
        available_cols = [col for col in display_cols if col in display_master.columns]
        
        st.dataframe(
            display_master[available_cols],
            use_container_width=True,
            height=400
        )
    
    with tab2:
        st.markdown("### Campaign Performance Details")
        
        display_campaigns = campaigns_filtered.copy()
        display_campaigns = display_campaigns.round(2)
        display_campaigns['date'] = display_campaigns['date'].dt.strftime('%Y-%m-%d')
        
        campaign_cols = ['date', 'channel', 'campaign', 'impression', 'clicks', 
                        'spend', 'attributed_revenue', 'ctr', 'cpc', 'roas']
        
        available_campaign_cols = [col for col in campaign_cols if col in display_campaigns.columns]
        
        st.dataframe(
            display_campaigns[available_campaign_cols],
            use_container_width=True,
            height=400
        )
    
    # Summary Statistics
    st.markdown("---")
    st.markdown("## 📈 Summary Statistics")
    
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    
    with col_stats1:
        st.markdown("### 💰 Financial Metrics")
        financial_stats = {
            'Total Marketing Spend': f"${master_filtered['spend'].sum():,.0f}",
            'Total Revenue': f"${master_filtered['total_revenue'].sum():,.0f}",
            'Marketing ROI': f"{((master_filtered['total_revenue'].sum() / master_filtered['spend'].sum()) - 1) * 100:.1f}%",
            'Avg Daily Spend': f"${master_filtered['spend'].mean():,.0f}"
        }
        
        for metric, value in financial_stats.items():
            st.metric(metric, value)
    
    with col_stats2:
        st.markdown("### 🎯 Performance Metrics")
        performance_stats = {
            'Avg CTR': f"{master_filtered['ctr'].mean():.2%}",
            'Avg CPC': f"${master_filtered['cpc'].mean():.2f}",
            'Avg ROAS': f"{master_filtered['roas'].mean():.2f}x",
            'Avg CAC': f"${master_filtered['cac'].mean():.2f}"
        }
        
        for metric, value in performance_stats.items():
            st.metric(metric, value)
    
    with col_stats3:
        st.markdown("### 🛍️ Business Metrics")
        business_stats = {
            'Total Orders': f"{master_filtered['orders'].sum():,.0f}",
            'New Customers': f"{master_filtered['new_customers'].sum():,.0f}",
            'Avg AOV': f"${master_filtered['aov'].mean():.2f}",
            'Avg Gross Margin': f"{master_filtered['gross_margin_pct'].mean():.1%}"
        }
        
        for metric, value in business_stats.items():
            st.metric(metric, value)

if __name__ == "__main__":
    main()
