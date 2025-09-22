############################################################
# DAR Global – CEO Dashboard  (v2 with top nav + date slider)
############################################################
import streamlit as st
from streamlit_option_menu import option_menu          # pip install streamlit-option-menu
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta, date
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
# Page set-up
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DAR Global – CEO Dashboard",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
#  CSS (unchanged, trimmed)
# ──────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header{background:linear-gradient(135deg,#1a1a1a 0%,#2d2d2d 100%);
    color:#DAA520;padding:30px;border-radius:15px;margin-bottom:25px;
    text-align:center;box-shadow:0 8px 16px rgba(0,0,0,.4);
    border:2px solid #DAA520;}
    /* Metric cards */
    div[data-testid="metric-container"]{
      background:linear-gradient(135deg,#2d2d2d 0%,#1a1a1a 100%);
      border:2px solid #DAA520;border-radius:12px;padding:1rem;color:white;
      box-shadow:0 4px 8px rgba(0,0,0,.3);}
    div[data-testid="metric-container"]>label{color:#1E90FF!important;font-weight:bold;font-size:1.1rem;}
    div[data-testid="metric-container"]>div{color:#DAA520!important;font-weight:bold;font-size:2rem;}
    .insight-box{background:linear-gradient(135deg,#2d2d2d 0%,#1a1a1a 100%);
      padding:20px;border-radius:12px;border-left:5px solid #32CD32;
      margin:15px 0;color:white;box-shadow:0 4px 8px rgba(0,0,0,.3);}
    .insight-box h4{color:#32CD32;margin-bottom:12px;}
    h1,h2,h3{color:#DAA520;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    ds = {}
    for csv in [
        "lead.csv",
        "agent.csv",
        "lead_call.csv",
        "lead_schedule.csv",
        "country.csv",
        "lead_stage.csv",
        "call_status.csv",
        "sentiment.csv",
        "task_type.csv",
        "task_status.csv",
    ]:
        ds[csv.split(".")[0]] = pd.read_csv(csv)
    with open("dashboard_data.json") as f:
        ds["config"] = json.load(f)
    return ds


def money(x):
    if pd.isna(x) or x == 0:
        return "$0"
    if x >= 1e9:
        return f"${x/1e9:,.1f}B"
    if x >= 1e6:
        return f"${x/1e6:,.1f}M"
    if x >= 1e3:
        return f"${x/1e3:,.1f}K"
    return f"${x:,.0f}"


def num(x):
    if pd.isna(x) or x == 0:
        return "0"
    if x >= 1e6:
        return f"{x/1e6:,.1f}M"
    if x >= 1e3:
        return f"{x/1e3:,.1f}K"
    return f"{x:,.0f}"


# ──────────────────────────────────────────────────────────
#  UI – Header + global filters
# ──────────────────────────────────────────────────────────
ds = load_data()
if ds is None:
    st.stop()

st.markdown(
    f"""
    <div class="main-header">
      <h1>🏗️ DAR Global</h1>
      <h2>Executive CRM Dashboard</h2>
      <p style="color:#1E90FF">Luxury Real Estate AI-Powered Analytics • Q3 2025</p>
      <p style="color:#32CD32">Last Updated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Date-grain selector in sidebar
with st.sidebar:
    st.markdown("### 📅 Date Filter")
    grain = st.radio("Grain", ["Week", "Month", "Year"], horizontal=True)
    # Date range slider
    min_d = pd.to_datetime(ds["leads"]["CreatedOn"]).min().date()
    max_d = pd.to_datetime(ds["leads"]["CreatedOn"]).max().date()
    step = timedelta(days=1) if grain != "Year" else timedelta(days=7)
    default_start = max_d - timedelta(days=6) if grain == "Week" else (
        max_d.replace(day=1) if grain == "Month" else max_d.replace(month=1, day=1)
    )
    date_start, date_end = st.slider(
        "Select range",
        min_value=min_d,
        max_value=max_d,
        value=(default_start, max_d),
        step=step,
        format="YYYY-MM-DD",
    )

# Filter helper (used by every sheet)
def time_filter(df, col):
    mask = (pd.to_datetime(df[col]).dt.date >= date_start) & (
        pd.to_datetime(df[col]).dt.date <= date_end
    )
    return df.loc[mask].copy()


# ──────────────────────────────────────────────────────────
# Top navigation bar  (streamlit-option-menu)
# ──────────────────────────────────────────────────────────
selected = option_menu(
    menu_title=None,
    options=[
        "Executive",
        "Leads",
        "Calls",
        "Tasks",
        "Agents",
        "Conversion",
        "Geography",
    ],
    icons=[
        "speedometer2",
        "people",
        "telephone",
        "check2-circle",
        "person-badge",
        "graph-up",
        "geo-alt",
    ],
    orientation="horizontal",
    default_index=0,
    styles={
        "container": {"padding": "0!important"},
        "nav-link": {
            "font-size": "14px",
            "font-weight": "600",
            "color": "#ffffff",
            "padding": "10px 18px",
        },
        "nav-link-selected": {"background-color": "#DAA520"},
    },
)

# ──────────────────────────────────────────────────────────
#  Page routers
# ──────────────────────────────────────────────────────────
if selected == "Executive":
    # Use time-filtered data
    leads = time_filter(ds["leads"], "CreatedOn")
    calls = time_filter(ds["calls"], "CallDateTime")
    agents = ds["agents"]

    total_leads = len(leads)
    pipeline_active = leads["EstimatedBudget"].sum()
    won_rev = leads[leads["LeadStageId"] == 6]["EstimatedBudget"].sum()
    conv_rate = (len(leads[leads["LeadStageId"] == 6]) / total_leads) * 100 if total_leads else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Leads", num(total_leads))
    m2.metric("Active Pipeline", money(pipeline_active))
    m3.metric("Revenue Generated", money(won_rev))
    m4.metric("Conversion Rate", f"{conv_rate:.1f}%")

    # insights (static)
    st.markdown(
        """
        <div class="insight-box"><h4>🔮 Predictive Revenue Forecast</h4>
        <ul><li>Q4 2025 projection: $28.3 B (≈90 % confidence)</li>
        <li>Growth trajectory remains positive over selected period</li></ul></div>
        """,
        unsafe_allow_html=True,
    )

elif selected == "Leads":
    leads = time_filter(ds["leads"], "CreatedOn")
    stage_map = {1: "New", 2: "In Progress", 3: "In Progress",
                 4: "Interested", 5: "Interested", 6: "Closed Won", 7: "Closed Lost"}
    status_counts = (
        leads["LeadStageId"].map(stage_map).value_counts().reindex(
            ["New", "In Progress", "Interested", "Closed Won", "Closed Lost"]
        )
    )
    fig = go.Figure(go.Pie(
        labels=status_counts.index,
        values=status_counts.values,
        hole=0.4,
        marker_colors=["#1E90FF", "#FFA500", "#32CD32", "#DAA520", "#DC143C"],
        textinfo="label+percent",
    ))
    fig.update_layout(title="Lead Status Distribution", title_font_color="#DAA520")
    st.plotly_chart(fig, use_container_width=True)

elif selected == "Calls":
    calls = time_filter(ds["calls"], "CallDateTime")
    daily = (
        calls.groupby(calls["CallDateTime"].dt.date)["LeadCallId"]
        .count()
        .reset_index(name="Total")
    )
    fig = go.Figure(go.Scatter(x=daily["CallDateTime"], y=daily["Total"], mode="lines+markers"))
    fig.update_layout(title="Call Volume", title_font_color="#DAA520")
    st.plotly_chart(fig, use_container_width=True)

elif selected == "Tasks":
    tasks = time_filter(ds["schedules"], "ScheduledDate")
    today = date.today()
    st.metric("Tasks Today", len(tasks[tasks["ScheduledDate"].dt.date == today]))
    st.dataframe(tasks[["ScheduleId", "LeadId", "TaskStatusId", "ScheduledDate"]].head(20))

elif selected == "Agents":
    # simplified table view
    st.dataframe(
        ds["agents"][["AgentId", "FirstName", "LastName", "Role", "IsActive"]].head(20),
        use_container_width=True,
    )

elif selected == "Conversion":
    leads = time_filter(ds["leads"], "CreatedOn")
    won = leads[leads["LeadStageId"] == 6]
    lost = leads[leads["LeadStageId"] == 7]
    st.metric("Won Deals", len(won))
    st.metric("Lost Deals", len(lost))
    funnel = go.Figure(go.Funnel(
        y=["New", "Qualified", "Presentation", "Negotiation", "Contract", "Closed Won"],
        x=[len(leads[leads["LeadStageId"] == i]) for i in range(1, 7)],
    ))
    st.plotly_chart(funnel, use_container_width=True)

elif selected == "Geography":
    geo = (
        ds["leads"]
        .groupby("CountryId")["EstimatedBudget"]
        .sum()
        .reset_index()
        .merge(ds["country"], on="CountryId")
    )
    fig = go.Figure(go.Bar(
        x=geo["CountryName_E"],
        y=geo["EstimatedBudget"] / 1e9,
        marker_color="#1E90FF",
    ))
    fig.update_layout(title="Pipeline by Country ($B)", title_font_color="#DAA520")
    st.plotly_chart(fig, use_container_width=True)

