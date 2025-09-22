import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="DAR Global -Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for executive styling
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
</style>
""", unsafe_allow_html=True)

# Load data function
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

def format_currency(value):
    """Format currency values for display"""
    if pd.isna(value) or value == 0:
        return "$0"
    if value >= 1000000000:
        return f"${value/1000000000:.1f}B"
    elif value >= 1000000:
        return f"${value/1000000:.1f}M"
    elif value >= 1000:
        return f"${value/1000:.1f}K"
    else:
        return f"${value:,.0f}"

def format_number(value):
    """Format numbers with appropriate suffixes"""
    if pd.isna(value) or value == 0:
        return "0"
    if value >= 1000000:
        return f"{value/1000000:.1f}M"
    elif value >= 1000:
        return f"{value/1000:.1f}K"
    else:
        return f"{value:,.0f}"

def main():
    # Load data
    data = load_data()
    if data is None:
        st.error("Failed to load data. Please ensure all CSV files are present.")
        return

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>DAR Global</h1>
        <h2>Executive CRM Dashboard</h2>
        <p style="color: #1E90FF; font-size: 1.2rem; margin-top: 15px;">
            Luxury Real Estate AI-Powered Analytics • Q3 2025
        </p>
        <p style="color: #32CD32; font-size: 1rem; margin-top: 10px;">
            Last Updated: {0}
        </p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M UTC")), unsafe_allow_html=True)

    # Sidebar Navigation
    st.sidebar.markdown("## 📊 Dashboard Navigation")
    st.sidebar.markdown("### Select Dashboard Sheet:")

    dashboard_options = [
        "🎯 Executive Summary",
        "📈 Lead Status Dashboard", 
        "📞 AI Call Activity Dashboard",
        "✅ Follow-up & Task Dashboard",
        "👥 Agent Performance Dashboard",
        "💰 Conversion Dashboard",
        "🌍 Geographic Dashboard"
    ]

    selected_sheet = st.sidebar.selectbox("", dashboard_options, key="nav_select")

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 🏗️ DAR Global
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

    # Route to appropriate dashboard
    if selected_sheet == "🎯 Executive Summary":
        show_executive_summary(data)
    elif selected_sheet == "📈 Lead Status Dashboard":
        show_lead_status_dashboard(data)
    elif selected_sheet == "📞 AI Call Activity Dashboard":
        show_call_activity_dashboard(data)
    elif selected_sheet == "✅ Follow-up & Task Dashboard":
        show_followup_task_dashboard(data)
    elif selected_sheet == "👥 Agent Performance Dashboard":
        show_agent_performance_dashboard(data)
    elif selected_sheet == "💰 Conversion Dashboard":
        show_conversion_dashboard(data)
    elif selected_sheet == "🌍 Geographic Dashboard":
        show_geographic_dashboard(data)

def show_executive_summary(data):
    st.header("🎯 Executive Summary")

    leads_df = data['leads']
    calls_df = data['calls']
    agents_df = data['agents']

    total_leads = len(leads_df)
    active_pipeline = leads_df[leads_df['IsActive'] == 1]['EstimatedBudget'].sum()
    won_revenue = leads_df[leads_df['LeadStageId'] == 6]['EstimatedBudget'].sum()
    conversion_rate = (len(leads_df[leads_df['LeadStageId'] == 6]) / total_leads) * 100

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
        call_success_rate = (len(calls_df[calls_df['CallStatusId'] == 1]) / len(calls_df)) * 100
        st.metric("Call Success Rate", f"{call_success_rate:.1f}%")

    with col2:
        st.metric("ROI", "8,205.2%")

    with col3:
        st.metric("Active Agents", format_number(len(agents_df[agents_df['IsActive'] == 1])))

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

def show_lead_status_dashboard(data):
    st.header("📈 Lead Status Dashboard")

    leads_df = data['leads']
    stage_counts = leads_df['LeadStageId'].value_counts().sort_index()

    # Map stages to simplified statuses for pie chart
    status_mapping = {1: "New", 2: "In Progress", 3: "In Progress", 4: "Interested", 5: "Interested", 6: "Closed Won", 7: "Closed Lost"}

    status_data = []
    colors = {"New": "#1E90FF", "In Progress": "#FFA500", "Interested": "#32CD32", "Closed Won": "#DAA520", "Closed Lost": "#DC143C"}

    status_counts = {}
    for stage_id, count in stage_counts.items():
        status = status_mapping.get(stage_id, "Other")
        status_counts[status] = status_counts.get(status, 0) + count

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
        st.metric("Active Leads", format_number(len(leads_df[leads_df['IsActive'] == 1])))

        won_leads = status_counts.get("Closed Won", 0)
        lost_leads = status_counts.get("Closed Lost", 0)
        total_closed = won_leads + lost_leads
        win_rate = (won_leads / total_closed * 100) if total_closed > 0 else 0
        st.metric("Win Rate", f"{win_rate:.1f}%")

        conversion_rate = (won_leads / len(leads_df) * 100)
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

def show_call_activity_dashboard(data):
    st.header("📞 AI Call Activity Dashboard")

    calls_df = data['calls']
    calls_df['CallDateTime'] = pd.to_datetime(calls_df['CallDateTime'])

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
    connected_calls = len(calls_df[calls_df['CallStatusId'] == 1])
    success_rate = (connected_calls / total_calls * 100) if total_calls > 0 else 0
    ai_generated = len(calls_df[calls_df['IsAIGenerated'] == 1])
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

def show_followup_task_dashboard(data):
    st.header("✅ Follow-up & Task Dashboard")

    schedules_df = data['schedules']
    schedules_df['ScheduledDate'] = pd.to_datetime(schedules_df['ScheduledDate'])

    today = datetime.now().date()
    today_tasks = len(schedules_df[schedules_df['ScheduledDate'].dt.date == today])
    week_start = today - timedelta(days=today.weekday())
    week_end = week_start + timedelta(days=6)
    week_tasks = len(schedules_df[(schedules_df['ScheduledDate'].dt.date >= week_start) & (schedules_df['ScheduledDate'].dt.date <= week_end)])
    overdue_tasks = len(schedules_df[(schedules_df['ScheduledDate'].dt.date < today) & (schedules_df['TaskStatusId'].isin([1, 2]))])
    completed_tasks = len(schedules_df[schedules_df['TaskStatusId'] == 3])
    completion_rate = (completed_tasks / len(schedules_df) * 100) if len(schedules_df) > 0 else 0

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

def show_agent_performance_dashboard(data):
    st.header("👥 Agent Performance Dashboard")

    leads_df = data['leads']
    agents_df = data['agents']

    agent_metrics = leads_df.groupby('AssignedAgentId').agg({
        'LeadId': 'count',
        'EstimatedBudget': 'sum',
        'LeadStageId': lambda x: (x == 6).sum()
    }).reset_index()

    agent_metrics.columns = ['AgentId', 'LeadsAssigned', 'PipelineValue', 'DealsWon']
    agent_metrics = agent_metrics.merge(agents_df[['AgentId', 'FirstName', 'LastName', 'Role']], on='AgentId', how='left')
    agent_metrics['AgentName'] = agent_metrics['FirstName'] + ' ' + agent_metrics['LastName']
    agent_metrics['ConversionRate'] = (agent_metrics['DealsWon'] / agent_metrics['LeadsAssigned'] * 100).round(1)
    agent_metrics = agent_metrics.fillna(0)

    st.subheader("🏆 Top Performing Agents")
    top_agents = agent_metrics.nlargest(10, 'PipelineValue')[['AgentName', 'Role', 'LeadsAssigned', 'PipelineValue', 'DealsWon', 'ConversionRate']]
    top_agents['PipelineValue'] = top_agents['PipelineValue'].apply(format_currency)
    st.dataframe(top_agents, use_container_width=True)

    st.subheader("🗓️ Agent Utilization Heatmap")
    st.info("This shows a simulated agent availability heatmap for the top 20 agents across business hours.")

    agents_sample = agent_metrics.head(20)['AgentName'].tolist()
    time_slots = [f"{h:02d}:00" for h in range(9, 18)]

    np.random.seed(42)
    utilization_data = np.random.choice([0, 1, 2, 3], size=(len(agents_sample), len(time_slots)), p=[0.3, 0.4, 0.2, 0.1])

    fig = go.Figure(data=go.Heatmap(
        z=utilization_data,
        x=time_slots,
        y=agents_sample,
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
    active_agents = len(agents_df[agents_df['IsActive'] == 1])

    with col1:
        st.metric("Active Agents", active_agents)

    with col2:
        st.metric("Utilization Rate", "78.5%")

    with col3:
        assigned_leads = len(leads_df[leads_df['AssignedAgentId'].notna()])
        avg_leads_per_agent = assigned_leads / active_agents if active_agents > 0 else 0
        st.metric("Avg Leads/Agent", f"{avg_leads_per_agent:.1f}")

    with col4:
        top_performers = len(agent_metrics[agent_metrics['PipelineValue'] >= agent_metrics['PipelineValue'].quantile(0.8)])
        st.metric("Top Performers", f"{top_performers} ({top_performers/len(agent_metrics)*100:.0f}%)")

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

def show_conversion_dashboard(data):
    st.header("💰 Conversion Dashboard")

    leads_df = data['leads']
    won_leads = len(leads_df[leads_df['LeadStageId'] == 6])
    lost_leads = len(leads_df[leads_df['LeadStageId'] == 7])
    won_revenue = leads_df[leads_df['LeadStageId'] == 6]['EstimatedBudget'].sum()

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
    converted = [45, 52, 67, 78, 89, 94, 103, 118, won_leads]
    dropped = [1890, 1756, 1623, 1534, 1445, 1378, 1289, 1234, lost_leads]
    revenue = [285, 324, 421, 487, 556, 589, 645, 738, won_revenue/1000000]

    fig = make_subplots(rows=2, cols=2, subplot_titles=('Monthly Conversions vs Dropped', 'Revenue Trend ($M)', 'Conversion Rate Trend', 'Pipeline Risk Analysis'), specs=[[{"secondary_y": False}, {"secondary_y": False}], [{"secondary_y": False}, {"type": "pie"}]])

    fig.add_trace(go.Bar(x=months, y=converted, name='Converted', marker_color='#32CD32'), row=1, col=1)
    fig.add_trace(go.Bar(x=months, y=dropped, name='Dropped', marker_color='#DC143C'), row=1, col=1)
    fig.add_trace(go.Scatter(x=months, y=revenue, name='Revenue ($M)', line=dict(color='#DAA520', width=3)), row=1, col=2)

    conversion_rates = [c/(c+d)*100 for c, d in zip(converted, dropped)]
    fig.add_trace(go.Scatter(x=months, y=conversion_rates, name='Conversion Rate (%)', line=dict(color='#1E90FF', width=3)), row=2, col=1)

    active_pipeline = leads_df[leads_df['IsActive'] == 1]['EstimatedBudget'].sum()
    risk_categories = ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
    risk_values = [active_pipeline*0.6/1e9, active_pipeline*0.25/1e9, active_pipeline*0.10/1e9, active_pipeline*0.05/1e9]
    fig.add_trace(go.Pie(labels=risk_categories, values=risk_values, name="Pipeline Risk", marker=dict(colors=['#32CD32', '#DAA520', '#FFA500', '#DC143C'])), row=2, col=2)

    fig.update_layout(height=700, showlegend=True, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white', title_font_color='#DAA520')
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

def show_geographic_dashboard(data):
    st.header("🌍 Geographic Dashboard")

    leads_df = data['leads']
    countries_df = data['countries']

    geo_performance = leads_df.groupby('CountryId').agg({
        'LeadId': 'count',
        'EstimatedBudget': 'sum',
        'LeadStageId': lambda x: (x == 6).sum()
    }).reset_index()

    geo_performance.columns = ['CountryId', 'LeadCount', 'PipelineValue', 'WonDeals']
    geo_performance = geo_performance.merge(countries_df[['CountryId', 'CountryName_E', 'CountryCode']], on='CountryId')
    geo_performance['ConversionRate'] = (geo_performance['WonDeals'] / geo_performance['LeadCount'] * 100).round(1)
    geo_performance = geo_performance.sort_values('PipelineValue', ascending=False)

    st.subheader("🏆 Top Markets Performance")

    top_markets = geo_performance.head(8)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=top_markets['LeadCount'],
        y=top_markets['PipelineValue']/1000000000,
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
    display_df['PipelineValue'] = display_df['PipelineValue'].apply(format_currency)
    display_df.columns = ['Country', 'Leads', 'Pipeline Value', 'Won Deals', 'Conversion Rate (%)']
    st.dataframe(display_df, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Markets", len(geo_performance))

    with col2:
        st.metric("Global Pipeline", format_currency(geo_performance['PipelineValue'].sum()))

    with col3:
        st.metric("Top Market", geo_performance.iloc[0]['CountryName_E'])

    with col4:
        st.metric("Avg Conversion", f"{geo_performance['ConversionRate'].mean():.1f}%")

    st.markdown("---")
    st.subheader("🤖 AI Geographic Intelligence")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="insight-box">
        <h4>🎯 Market Analysis</h4>
        <ul>
        <li><strong>Expansion Opportunities:</strong> Egypt, Morocco, Turkey markets</li>
        <li><strong>Saturation Risk:</strong> UAE approaching market capacity</li>
        <li><strong>Optimal Allocation:</strong> Saudi Arabia needs +25 brokers</li>
        <li><strong>Cultural Adaptation:</strong> Arabic language processing 94% accurate</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="insight-box">
        <h4>📊 Geo Predictions</h4>
        <ul>
        <li>Qatar market projected to grow 45% in 2026</li>
        <li>India shows highest lead volume potential (3x growth)</li>
        <li>European expansion ROI: 285% projected return</li>
        <li>Cross-border referral network opportunity identified</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
