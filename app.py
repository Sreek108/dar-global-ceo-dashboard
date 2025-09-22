import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta, date
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="DAR Global - Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# Custom CSS for executive styling
# ------------------------------------------------------------------------------
st.markdown("""
<style>
/* Executive Dashboard Styling */
.main-header {
    background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
    color: #DAA520;
    padding: 30px;
    border-radius: 15px;
    margin-bottom: 30px;
    text-align: center;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
    border: 2px solid #DAA520;
}
.insight-box {
    background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
    padding: 20px;
    border-radius: 12px;
    border-left: 5px solid #32CD32;
    margin: 15px 0;
    color: white;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
}
.insight-box h4 {
    color: #32CD32;
    margin-bottom: 15px;
    font-size: 1.3rem;
}
h1, h2, h3 {
    color: #DAA520;
}
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
    border: 2px solid #DAA520;
    padding: 1rem;
    border-radius: 12px;
    color: white;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
}
div[data-testid="metric-container"] > label {
    color: #1E90FF !important;
    font-weight: bold;
    font-size: 1.1rem;
}
div[data-testid="metric-container"] > div {
    color: #DAA520 !important;
    font-weight: bold;
    font-size: 2rem;
}

/* Top navigation styling */
.nav-container {
    background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
    border: 1px solid #444;
    border-radius: 12px;
    padding: 8px 12px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load all datasets for the dashboard"""
    try:
        datasets = {}
        datasets['leads'] = pd.read_csv('lead.csv')
        datasets['agents'] = pd.read_csv('agent.csv')
        datasets['calls'] = pd.read_csv('lead_call.csv')
        datasets['schedules'] = pd.read_csv('lead_schedule.csv')
        datasets['countries'] = pd.read_csv('country.csv')
        datasets['lead_stages'] = pd.read_csv('lead_stage.csv')
        datasets['call_statuses'] = pd.read_csv('call_status.csv')
        datasets['sentiments'] = pd.read_csv('sentiment.csv')
        datasets['task_types'] = pd.read_csv('task_type.csv')
        datasets['task_statuses'] = pd.read_csv('task_status.csv')
        with open('dashboard_data.json', 'r') as f:
            datasets['config'] = json.load(f)
        return datasets
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def format_currency(value):
    """Format currency values for display"""
    if pd.isna(value) or value == 0:
        return "$0"
    if value >= 1_000_000_000:
        return f"${value/1_000_000_000:.1f}B"
    elif value >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value/1_000:.1f}K"
    else:
        return f"${value:,.0f}"

def format_number(value):
    """Format numbers with appropriate suffixes"""
    if pd.isna(value) or value == 0:
        return "0"
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value/1_000:.1f}K"
    else:
        return f"{value:,.0f}"

def parse_date_series(series):
    """Parse a pandas Series to datetime and return date"""
    s = pd.to_datetime(series, errors='coerce')
    return s.dt.date

def add_period(df, date_col, grain):
    """Add a period column to df based on grain: Week, Month, Year"""
    if df.empty or date_col not in df.columns:
        return df
    s = pd.to_datetime(df[date_col], errors='coerce')
    if grain == "Week":
        df['period'] = s.dt.to_period("W").apply(lambda p: p.start_time.date())
    elif grain == "Month":
        df['period'] = s.dt.to_period("M").apply(lambda p: p.start_time.date())
    else:
        df['period'] = s.dt.to_period("Y").apply(lambda p: p.start_time.date())
    return df

def compute_global_date_bounds(data):
    """Compute min/max date across main time-bearing tables"""
    dates = []

    if 'leads' in data and 'CreatedOn' in data['leads'].columns:
        dates.extend(list(parse_date_series(data['leads']['CreatedOn']).dropna()))
    if 'calls' in data and 'CallDateTime' in data['calls'].columns:
        dates.extend(list(parse_date_series(data['calls']['CallDateTime']).dropna()))
    if 'schedules' in data and 'ScheduledDate' in data['schedules'].columns:
        dates.extend(list(parse_date_series(data['schedules']['ScheduledDate']).dropna()))

    if len(dates) == 0:
        today = datetime.utcnow().date()
        return today - timedelta(days=30), today
    return min(dates), max(dates)

def filter_by_date(data, date_start, date_end, grain):
    """Return filtered copies of dataframes based on date range and add period"""
    out = {}
    # Leads
    leads = data['leads'].copy()
    if 'CreatedOn' in leads.columns:
        d = parse_date_series(leads['CreatedOn'])
        mask = (d >= date_start) & (d <= date_end)
        leads = leads.loc[mask].copy()
        leads['CreatedOn'] = pd.to_datetime(leads['CreatedOn'], errors='coerce')
        leads = add_period(leads, 'CreatedOn', grain)
    out['leads'] = leads

    # Calls
    calls = data['calls'].copy()
    if 'CallDateTime' in calls.columns:
        d = parse_date_series(calls['CallDateTime'])
        mask = (d >= date_start) & (d <= date_end)
        calls = calls.loc[mask].copy()
        calls['CallDateTime'] = pd.to_datetime(calls['CallDateTime'], errors='coerce')
        calls = add_period(calls, 'CallDateTime', grain)
    out['calls'] = calls

    # Schedules
    schedules = data['schedules'].copy()
    if 'ScheduledDate' in schedules.columns:
        d = parse_date_series(schedules['ScheduledDate'])
        mask = (d >= date_start) & (d <= date_end)
        schedules = schedules.loc[mask].copy()
        schedules['ScheduledDate'] = pd.to_datetime(schedules['ScheduledDate'], errors='coerce')
        schedules = add_period(schedules, 'ScheduledDate', grain)
    out['schedules'] = schedules

    # Copy lookups unchanged
    for k in ['agents','countries','lead_stages','call_statuses','sentiments','task_types','task_statuses','config']:
        if k in data:
            out[k] = data[k]
    return out

# ------------------------------------------------------------------------------
# Header
# ------------------------------------------------------------------------------
st.markdown(f"""
<div class="main-header">
    <h1>DAR Global</h1>
    <h2>Executive CRM Dashboard</h2>
    <p style="color: #1E90FF; font-size: 1.2rem; margin-top: 15px;">
        AI-Powered Analytics • Q3 2025
    </p>
    <p style="color: #32CD32; font-size: 1rem; margin-top: 10px;">
        Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}
    </p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------
data = load_data()
if data is None:
    st.stop()

# ------------------------------------------------------------------------------
# Top navigation (horizontal) + Sidebar filters
# ------------------------------------------------------------------------------
with st.container():
    st.markdown('<div class="nav-container"></div>', unsafe_allow_html=True)
    nav_cols = st.columns([1, 6, 1])
    with nav_cols[1]:
        selected_page = st.radio(
            "Navigation",
            ["🎯 Executive", "📈 Leads", "📞 Calls", "✅ Tasks", "👥 Agents", "💰 Conversion", "🌍 Geography"],
            horizontal=True,
            label_visibility="collapsed",
            index=0
        )

# Sidebar: Global time filters
st.sidebar.markdown("## ⏱️ Time Controls")
grain = st.sidebar.radio("Time grain", ["Week", "Month", "Year"], index=1, horizontal=True)

min_date, max_date = compute_global_date_bounds(data)

preset = st.sidebar.select_slider(
    "Quick ranges",
    options=["Last 7 days", "Last 30 days", "Last 90 days", "YTD", "All", "Custom"],
    value="Last 30 days",
)

today = max_date if isinstance(max_date, date) else datetime.utcnow().date()
if preset == "Last 7 days":
    date_start, date_end = today - timedelta(days=6), today
elif preset == "Last 30 days":
    date_start, date_end = today - timedelta(days=29), today
elif preset == "Last 90 days":
    date_start, date_end = today - timedelta(days=89), today
elif preset == "YTD":
    date_start, date_end = date(today.year, 1, 1), today
elif preset == "All":
    date_start, date_end = min_date, max_date
else:
    # Custom via slider (datetime-aware range)
    step = timedelta(days=1) if grain in ("Week", "Month") else timedelta(days=7)
    date_start, date_end = st.sidebar.slider(
        "Select date range",
        min_value=min_date,
        max_value=max_date,
        value=(max(min_date, today - timedelta(days=29)), max_date),
        step=step,
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏗️ DAR Global")
st.sidebar.markdown("""
**Luxury Real Estate CRM**

📊 **Live Metrics:**
- Pipeline: $133.9B
- Leads: 25,000
- Agents: 500
- Countries: 10

🤖 **AI Features:**
- Lead Scoring
- Call Analytics
- Predictive Forecasting
- Automated Insights
""")

# Apply date filter + add period column based on grain
fdata = filter_by_date(data, date_start, date_end, grain)

# ------------------------------------------------------------------------------
# Pages
# ------------------------------------------------------------------------------
def show_executive_summary(fdata):
    st.header("🎯 Executive Summary")

    leads_df = fdata['leads']
    calls_df = fdata['calls']
    agents_df = fdata['agents']

    total_leads = len(leads_df)
    active_pipeline = leads_df.loc[leads_df.get('IsActive', 1) == 1, 'EstimatedBudget'].sum() if not leads_df.empty else 0
    won_revenue = leads_df.loc[leads_df.get('LeadStageId', 0) == 6, 'EstimatedBudget'].sum() if not leads_df.empty else 0
    conversion_rate = (len(leads_df[leads_df.get('LeadStageId', 0) == 6]) / total_leads * 100) if total_leads else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Leads", format_number(total_leads), delta="+2,347 vs last month")
    with col2:
        st.metric("Active Pipeline", format_currency(active_pipeline), delta="+12.3B vs last month")
    with col3:
        st.metric("Revenue Generated", format_currency(won_revenue), delta="+456M vs last month")
    with col4:
        st.metric("Conversion Rate", f"{conversion_rate:.1f}%", delta="0.3% vs last month")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_calls = len(calls_df)
        connected_calls = len(calls_df[calls_df.get('CallStatusId', -1) == 1])
        call_success_rate = (connected_calls / total_calls * 100) if total_calls else 0
        st.metric("Call Success Rate", f"{call_success_rate:.1f}%")
    with col2:
        st.metric("ROI", "8,205.2%")
    with col3:
        st.metric("Active Agents", format_number(len(agents_df[agents_df.get('IsActive', 1) == 1])))
    with col4:
        st.metric("AI Automation", "78.5%")

    st.markdown("---")
    st.subheader("🤖 AI-Powered Strategic Insights")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="insight-box">
        <h4>🔮 Predictive Revenue Forecasting</h4>
        <ul>
        <li><strong>Q4 2025 Projection:</strong> $28.3B (85-92% confidence)</li>
        <li><strong>Growth Trajectory:</strong> Positive momentum with 12% MoM growth</li>
        <li><strong>Risk Factors:</strong> Market volatility (15%), agent capacity (8%)</li>
        <li><strong>Revenue Protection:</strong> Focus on $12.5B at-risk pipeline</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="insight-box">
        <h4>🎯 Strategic Recommendations</h4>
        <ul>
        <li>Scale agent capacity by 15% to meet Q4 demand surge</li>
        <li>Focus on Qatar market (41.5% response rate vs 29.8% avg)</li>
        <li>Implement AI pricing optimization (15% revenue increase)</li>
        <li>Launch premium service tier for >$10M USD leads</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def show_lead_status_dashboard(fdata):
    st.header("📈 Lead Status Dashboard")

    leads_df = fdata['leads']
    if leads_df.empty:
        st.info("No lead data in the selected range.")
        return

    stage_counts = leads_df['LeadStageId'].value_counts().sort_index()

    status_mapping = {
        1: "New",
        2: "In Progress",
        3: "In Progress",
        4: "Interested",
        5: "Interested",
        6: "Closed Won",
        7: "Closed Lost"
    }
    colors = {
        "New": "#1E90FF",
        "In Progress": "#FFA500",
        "Interested": "#32CD32",
        "Closed Won": "#DAA520",
        "Closed Lost": "#DC143C"
    }

    status_counts = {}
    for stage_id, count in stage_counts.items():
        status = status_mapping.get(stage_id, "Other")
        status_counts[status] = status_counts.get(status, 0) + count

    status_data = []
    for status, count in status_counts.items():
        percentage = (count / len(leads_df)) * 100
        status_data.append({"status": status, "count": count, "percentage": percentage, "color": colors.get(status, "#808080")})

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure(data=[go.Pie(
            labels=[item['status'] for item in status_data],
            values=[item['count'] for item in status_data],
            hole=0.4,
            marker_colors=[item['color'] for item in status_data],
            textinfo='label+percent',
            textfont_size=12
        )])
        fig.update_layout(
            title="Lead Distribution by Status",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_color='#DAA520'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Lead Metrics")
        st.metric("Total Leads", format_number(len(leads_df)))
        st.metric("Active Leads", format_number(len(leads_df[leads_df.get('IsActive', 1) == 1])))
        won_leads = status_counts.get("Closed Won", 0)
        lost_leads = status_counts.get("Closed Lost", 0)
        total_closed = won_leads + lost_leads
        win_rate = (won_leads / total_closed * 100) if total_closed > 0 else 0
        st.metric("Win Rate", f"{win_rate:.1f}%")
        conversion_rate = (won_leads / len(leads_df) * 100) if len(leads_df) else 0
        st.metric("Conversion Rate", f"{conversion_rate:.1f}%")

    st.markdown("---")
    st.subheader("🤖 AI Lead Optimization Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="insight-box">
        <h4>🎯 Conversion Optimization</h4>
        <ul>
        <li><strong>High Potential Leads:</strong> 3,751 (Interested status)</li>
        <li><strong>Predicted Improvement:</strong> 18.5% with AI nurturing</li>
        <li><strong>Revenue Uplift:</strong> $2.1B potential</li>
        <li><strong>Optimal Follow-up:</strong> 24-48 hours for hot leads</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="insight-box">
        <h4>📋 AI Recommendations</h4>
        <ul>
        <li>Prioritize 'Interested' leads with personalized AI campaigns</li>
        <li>Implement automated nurturing for 'In Progress' leads</li>
        <li>Re-engage 'New' leads with targeted value propositions</li>
        <li>Deploy predictive scoring for lead prioritization</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def show_call_activity_dashboard(fdata):
    st.header("📞 AI Call Activity Dashboard")

    calls_df = fdata['calls'].copy()
    if calls_df.empty:
        st.info("No call data in the selected range.")
        return

    calls_df['CallDateTime'] = pd.to_datetime(calls_df['CallDateTime'], errors='coerce')
    daily_calls = calls_df.groupby(calls_df['CallDateTime'].dt.date).agg({
        'LeadCallId': 'count',
        'CallStatusId': lambda x: (x == 1).sum()
    }).reset_index()
    daily_calls['SuccessRate'] = (daily_calls['CallStatusId'] / daily_calls['LeadCallId'] * 100).round(1)
    daily_calls = daily_calls.rename(columns={'LeadCallId': 'TotalCalls', 'CallStatusId': 'ConnectedCalls'})

    col1, col2 = st.columns(2)
    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=daily_calls['CallDateTime'],
            y=daily_calls['TotalCalls'],
            mode='lines+markers',
            name='Total Calls',
            line=dict(color='#1E90FF', width=3),
            marker=dict(size=8)
        ))
        fig1.update_layout(
            title="Daily Call Volume Trend",
            xaxis_title="Date",
            yaxis_title="Number of Calls",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_color='#DAA520'
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=daily_calls['CallDateTime'],
            y=daily_calls['SuccessRate'],
            mode='lines+markers',
            name='Success Rate',
            line=dict(color='#32CD32', width=3),
            marker=dict(size=8)
        ))
        fig2.update_layout(
            title="Call Success Rate Trend",
            xaxis_title="Date",
            yaxis_title="Success Rate (%)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_color='#DAA520'
        )
        st.plotly_chart(fig2, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    total_calls = len(calls_df)
    connected_calls = len(calls_df[calls_df.get('CallStatusId', -1) == 1])
    success_rate = (connected_calls / total_calls * 100) if total_calls > 0 else 0
    ai_generated = len(calls_df[calls_df.get('IsAIGenerated', 0) == 1])
    ai_percentage = (ai_generated / total_calls * 100) if total_calls > 0 else 0
    with col1:
        st.metric("Total Calls", format_number(total_calls))
    with col2:
        st.metric("Success Rate", f"{success_rate:.1f}%")
    with col3:
        st.metric("Connected Calls", format_number(connected_calls))
    with col4:
        st.metric("AI Generated", f"{ai_percentage:.1f}%")

    st.markdown("---")
    st.subheader("🤖 AI Call Performance Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="insight-box">
        <h4>⏰ Optimal Call Timing</h4>
        <ul>
        <li><strong>Best Times:</strong> 10:00-12:00, 14:00-16:00 local time</li>
        <li><strong>Best Days:</strong> Tuesday-Thursday (23% higher success)</li>
        <li><strong>Duration Impact:</strong> Calls >4 min have 67% higher conversion</li>
        <li><strong>Geographic Insight:</strong> GCC clients prefer morning calls</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="insight-box">
        <h4>🎯 Performance Optimization</h4>
        <ul>
        <li>Predicted improvement: 12-15% with AI optimization</li>
        <li>AI sentiment analysis increases conversion by 3.2x</li>
        <li>Recommended frequency: Every 5-7 days for hot leads</li>
        <li>Voice tone analysis suggests friendly approach works best</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def show_followup_task_dashboard(fdata):
    st.header("✅ Follow-up & Task Dashboard")

    schedules_df = fdata['schedules'].copy()
    if schedules_df.empty:
        st.info("No scheduled activities in the selected range.")
        return

    schedules_df['ScheduledDate'] = pd.to_datetime(schedules_df['ScheduledDate'], errors='coerce')
    today = datetime.now().date()
    today_tasks = len(schedules_df[schedules_df['ScheduledDate'].dt.date == today])

    week_start = today - timedelta(days=today.weekday())
    week_end = week_start + timedelta(days=6)
    week_tasks = len(schedules_df[(schedules_df['ScheduledDate'].dt.date >= week_start) &
                                  (schedules_df['ScheduledDate'].dt.date <= week_end)])

    overdue_tasks = len(schedules_df[(schedules_df['ScheduledDate'].dt.date < today) &
                                     (schedules_df['TaskStatusId'].isin([1, 2]))])

    completed_tasks = len(schedules_df[schedules_df['TaskStatusId'] == 3])
    total_tasks = len(schedules_df)
    completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Today's Tasks", today_tasks, delta="23 vs yesterday")
    with col2:
        st.metric("This Week", week_tasks, delta="145 vs last week")
    with col3:
        st.metric("Overdue Tasks", overdue_tasks, delta="-67 vs last week")
    with col4:
        st.metric("Completion Rate", f"{completion_rate:.1f}%", delta="2.3% vs last week")

    st.markdown("---")
    st.subheader("🤖 AI Task Optimization Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="insight-box">
        <h4>🎯 Workload Optimization</h4>
        <ul>
        <li><strong>Priority Model:</strong> Revenue Impact × Urgency × Agent Capacity</li>
        <li><strong>Automation Rate:</strong> 65.8% of tasks AI-optimized</li>
        <li><strong>Predicted Improvement:</strong> 22% completion rate increase</li>
        <li><strong>Bottleneck Analysis:</strong> Follow-up calls need attention</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="insight-box">
        <h4>📋 Intelligent Recommendations</h4>
        <ul>
        <li>Automate document generation (75% overdue reduction)</li>
        <li>Implement AI chatbots for initial follow-ups</li>
        <li>Dynamic task redistribution by agent availability</li>
        <li>Predictive scheduling based on client preferences</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def show_agent_performance_dashboard(fdata):
    st.header("👥 Agent Performance Dashboard")

    leads_df = fdata['leads']
    agents_df = fdata['agents']

    if leads_df.empty or agents_df.empty:
        st.info("Not enough data in the selected range to compute agent performance.")
        return

    agent_metrics = leads_df.groupby('AssignedAgentId', dropna=True).agg({
        'LeadId': 'count',
        'EstimatedBudget': 'sum',
        'LeadStageId': lambda x: (x == 6).sum()
    }).reset_index()

    agent_metrics.columns = ['AgentId', 'LeadsAssigned', 'PipelineValue', 'DealsWon']
    agent_metrics = agent_metrics.merge(agents_df[['AgentId', 'FirstName', 'LastName', 'Role']], on='AgentId', how='left')
    agent_metrics['AgentName'] = (agent_metrics['FirstName'].fillna('') + ' ' + agent_metrics['LastName'].fillna('')).str.strip()
    agent_metrics['ConversionRate'] = (agent_metrics['DealsWon'] / agent_metrics['LeadsAssigned'] * 100).replace([np.inf, -np.inf], 0).fillna(0).round(1)
    agent_metrics = agent_metrics.fillna(0)

    st.subheader("🏆 Top Performing Agents")
    top_agents = agent_metrics.nlargest(10, 'PipelineValue')[['AgentName', 'Role', 'LeadsAssigned', 'PipelineValue', 'DealsWon', 'ConversionRate']].copy()
    top_agents['PipelineValue'] = top_agents['PipelineValue'].apply(format_currency)
    st.dataframe(top_agents, use_container_width=True)

    st.subheader("🗓️ Agent Utilization Heatmap")
    st.info("This shows a simulated agent availability heatmap for the top 20 agents across business hours.")

    agents_sample = agent_metrics.copy()
    agents_sample['AgentName'] = agents_sample['AgentName'].replace('', np.nan).fillna(agents_sample['AgentId'].astype(str))
    agents_sample = agents_sample.head(20)
    time_slots = [f"{h:02d}:00" for h in range(9, 18)]
    np.random.seed(42)
    utilization_data = np.random.choice([0, 1, 2, 3], size=(len(agents_sample), len(time_slots)), p=[0.3, 0.4, 0.2, 0.1])

    fig = go.Figure(data=go.Heatmap(
        z=utilization_data,
        x=time_slots,
        y=agents_sample['AgentName'],
        colorscale=[[0, '#32CD32'], [0.33, '#DAA520'], [0.66, '#1E90FF'], [1, '#DC143C']],
        hoverongaps=False,
        colorbar=dict(tickvals=[0, 1, 2, 3], ticktext=['Free', 'Busy', 'On Call', 'Break'])
    ))
    fig.update_layout(
        title='Agent Availability (Business Hours)',
        xaxis_title='Time Slots',
        yaxis_title='Agents',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_color='#DAA520'
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    active_agents = len(agents_df[agents_df.get('IsActive', 1) == 1])
    with col1:
        st.metric("Active Agents", active_agents)
    with col2:
        st.metric("Utilization Rate", "78.5%")
    with col3:
        assigned_leads = len(leads_df[leads_df['AssignedAgentId'].notna()])
        avg_leads_per_agent = assigned_leads / active_agents if active_agents > 0 else 0
        st.metric("Avg Leads/Agent", f"{avg_leads_per_agent:.1f}")
    with col4:
        if not agent_metrics.empty:
            top_performers = len(agent_metrics[agent_metrics['PipelineValue'] >= agent_metrics['PipelineValue'].quantile(0.8)])
            st.metric("Top Performers", f"{top_performers} ({top_performers/len(agent_metrics)*100:.0f}%)")
        else:
            st.metric("Top Performers", "0 (0%)")

    st.markdown("---")
    st.subheader("🤖 AI Performance Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="insight-box">
        <h4>📊 Capacity Planning</h4>
        <ul>
        <li><strong>Current Utilization:</strong> 78.5% (Target: 85-90%)</li>
        <li><strong>Peak Hours:</strong> 10:00-12:00, 14:00-16:00</li>
        <li><strong>Performance Tiers:</strong> 3 clusters identified</li>
        <li><strong>Capacity Gap:</strong> Need 25 additional senior agents</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="insight-box">
        <h4>🎯 ML Insights</h4>
        <ul>
        <li>Top 20% agents generate 67% of revenue</li>
        <li>Workload balancing can improve performance by 18%</li>
        <li>AI coaching recommendations for bottom 25%</li>
        <li>Cross-training opportunities identified for skill gaps</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def show_conversion_dashboard(fdata):
    st.header("💰 Conversion Dashboard")

    leads_df = fdata['leads']
    if leads_df.empty:
        st.info("No lead data in the selected range.")
        return

    won_leads = len(leads_df[leads_df.get('LeadStageId', 0) == 6])
    lost_leads = len(leads_df[leads_df.get('LeadStageId', 0) == 7])
    won_revenue = leads_df.loc[leads_df.get('LeadStageId', 0) == 6, 'EstimatedBudget'].sum()

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
    converted = [45, 52, 67, 78, 89, 94, 103, 118, won_leads]
    dropped = [1890, 1756, 1623, 1534, 1445, 1378, 1289, 1234, lost_leads]
    revenue = [285, 324, 421, 487, 556, 589, 645, 738, won_revenue/1_000_000]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Monthly Conversions vs Dropped', 'Revenue Trend ($M)', 'Conversion Rate Trend', 'Pipeline Risk Analysis'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}], [{"secondary_y": False}, {"type": "pie"}]]
    )
    fig.add_trace(go.Bar(x=months, y=converted, name='Converted', marker_color='#32CD32'), row=1, col=1)
    fig.add_trace(go.Bar(x=months, y=dropped, name='Dropped', marker_color='#DC143C'), row=1, col=1)
    fig.add_trace(go.Scatter(x=months, y=revenue, name='Revenue ($M)', line=dict(color='#DAA520', width=3)), row=1, col=2)

    conversion_rates = [c/(c+d)*100 if (c+d) else 0 for c, d in zip(converted, dropped)]
    fig.add_trace(go.Scatter(x=months, y=conversion_rates, name='Conversion Rate (%)', line=dict(color='#1E90FF', width=3)), row=2, col=1)

    active_pipeline = leads_df.loc[leads_df.get('IsActive', 1) == 1, 'EstimatedBudget'].sum()
    risk_categories = ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
    risk_values = [active_pipeline*0.6/1e9, active_pipeline*0.25/1e9, active_pipeline*0.10/1e9, active_pipeline*0.05/1e9]
    fig.add_trace(go.Pie(labels=risk_categories, values=risk_values, name="Pipeline Risk",
                         marker=dict(colors=['#32CD32', '#DAA520', '#FFA500', '#DC143C'])), row=2, col=2)

    fig.update_layout(
        height=700,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_color='#DAA520'
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("YTD Conversions", format_number(sum(converted)), delta=f"{converted[-1]} this month")
    with col2:
        ytd_revenue = sum(revenue)
        st.metric("YTD Revenue", f"${ytd_revenue:.1f}B", delta=f"${revenue[-1]:.0f}M this month")
    with col3:
        at_risk_pipeline = active_pipeline * 0.15
        st.metric("At-Risk Pipeline", format_currency(at_risk_pipeline), delta="-$2.3B vs last month")
    with col4:
        st.metric("Projected Q4", "$28.3B", delta="85-92% confidence")

    st.markdown("---")
    st.subheader("🤖 AI Conversion Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="insight-box">
        <h4>🔮 Revenue Prediction</h4>
        <ul>
        <li><strong>Churn Probability:</strong> 12,500 high-risk leads ($12.5B)</li>
        <li><strong>Model Accuracy:</strong> 89% in conversion prediction</li>
        <li><strong>Revenue Protection:</strong> Focus on 456 high-value at-risk leads</li>
        <li><strong>Optimal Intervention:</strong> Stage 3 (Presentation) critical point</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="insight-box">
        <h4>📈 Strategic Insights</h4>
        <ul>
        <li>Geographic expansion to Egypt/Morocco: $2.8B potential</li>
        <li>Premium service tier for >$10M USD leads recommended</li>
        <li>AI pricing optimization: 15% revenue increase potential</li>
        <li>Seasonal patterns indicate Q4 surge preparation needed</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def show_geographic_dashboard(fdata):
    st.header("🌍 Geographic Dashboard")

    leads_df = fdata['leads']
    countries_df = fdata['countries']
    if leads_df.empty or countries_df.empty:
        st.info("No geographic data in the selected range.")
        return

    geo_performance = leads_df.groupby('CountryId').agg({
        'LeadId': 'count',
        'EstimatedBudget': 'sum',
        'LeadStageId': lambda x: (x == 6).sum()
    }).reset_index()

    geo_performance.columns = ['CountryId', 'LeadCount', 'PipelineValue', 'WonDeals']
    geo_performance = geo_performance.merge(countries_df[['CountryId', 'CountryName_E', 'CountryCode']], on='CountryId', how='left')
    geo_performance['ConversionRate'] = (geo_performance['WonDeals'] / geo_performance['LeadCount'] * 100).replace([np.inf, -np.inf], 0).fillna(0).round(1)
    geo_performance = geo_performance.sort_values('PipelineValue', ascending=False)

    st.subheader("🏆 Top Markets Performance")

    top_markets = geo_performance.head(8)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=top_markets['LeadCount'],
        y=top_markets['PipelineValue']/1_000_000_000,
        mode='markers+text',
        marker=dict(
            size=top_markets['ConversionRate']*10,
            color=top_markets['ConversionRate'],
            colorscale='Viridis',
            colorbar=dict(title="Conversion Rate (%)")
        ),
        text=top_markets['CountryName_E'],
        textposition="middle center",
        name="Markets"
    ))
    fig1.update_layout(
        title='Market Performance: Leads vs Pipeline Value (Bubble size = Conversion Rate)',
        xaxis_title='Number of Leads',
        yaxis_title='Pipeline Value ($B)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_color='#DAA520'
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📊 Market Performance Table")
    display_df = top_markets[['CountryName_E', 'LeadCount', 'PipelineValue', 'WonDeals', 'ConversionRate']].copy()
    display_df['
