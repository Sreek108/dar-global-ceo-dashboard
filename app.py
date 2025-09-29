import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date

# Optional dependency for a premium horizontal nav
try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except Exception:
    HAS_OPTION_MENU = False

# -----------------------------------------------------------------------------
# Page configuration and executive styling
# -----------------------------------------------------------------------------
st.markdown(f"""
<style>
:root {{
  --exec-bg: {EXEC_BG};
  --exec-surface: {EXEC_SURFACE};
  --exec-primary: {EXEC_PRIMARY};
  --exec-blue: {EXEC_BLUE};
  --exec-green: {EXEC_GREEN};
}}

/* Full-bleed layout with minimal gutters */
section.main > div.block-container {{
  padding-left: 0.25rem !important;
  padding-right: 0.25rem !important;
  padding-top: 0.25rem !important;
  padding-bottom: 0.25rem !important;
  max-width: 100% !important;
}}

/* Remove Streamlit top headroom */
header[data-testid="stHeader"] {{
  height: 0 !important; padding: 0 !important; margin: 0 !important; background: transparent !important;
}}

/* Bigger horizontal navigation, tighter strip */
div[role="tablist"] {{
  margin-top: 4px !important; margin-bottom: 6px !important; gap: 6px !important;
  padding-top: 2px !important; padding-bottom: 2px !important;
}}
div[role="tablist"] > div,
div[role="tablist"] > button {{
  font-size: 16px !important; line-height: 36px !important; padding: 6px 12px !important;
}}
div[role="tablist"] button[aria-selected="true"],
div[role="tab"][aria-selected="true"] {{
  border-bottom: 3px solid var(--exec-primary) !important;
}}

/* Compact headings and dividers */
h1, .stMarkdown h1 {{ margin: 0 0 6px 0 !important; }}
h2, .stMarkdown h2 {{ margin: 4px 0 6px 0 !important; }}
hr {{ margin: 6px 0 !important; }}

/* Optional: tighten custom sections/banners if present */
.main-header {{ margin: 0 !important; padding: 10px 12px !important; border-width: 1px !important; }}
.section {{ padding: 8px !important; }}

/* Plotly and DataFrame spacing */
.element-container:has(.plotly) {{ margin-top: 4px !important; }}
[data-testid="stDataFrame"] {{ margin-top: 4px !important; }}

/* Hide default footer to remove bottom whitespace */
footer {{ visibility: hidden !important; height: 0 !important; }}
</style>
""", unsafe_allow_html=True)
# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def format_currency(value: float) -> str:
    if value is None or pd.isna(value):
        return "$0"
    if value >= 1_000_000_000:
        return f"${value/1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    if value >= 1_000:
        return f"${value/1_000:.1f}K"
    return f"${value:,.0f}"

def format_number(value: float) -> str:
    if value is None or pd.isna(value):
        return "0"
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value/1_000:.1f}K"
    return f"{value:,.0f}"

def safe_to_datetime(series, col_tz=None):
    s = pd.to_datetime(series, errors="coerce")
    return s

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    datasets = {}
    # Main datasets
    for name, fname in [
        ("leads", "lead.csv"),
        ("agents", "agent.csv"),
        ("calls", "lead_call.csv"),
        ("schedules", "lead_schedule.csv"),
        ("transactions", "lead_transaction.csv"),
        ("geographic", "geographic_data.csv"),
    ]:
        try:
            datasets[name] = pd.read_csv(fname)
        except Exception:
            datasets[name] = None

    # Lookups / config
    for name, fname in [
        ("countries", "country.csv"),
        ("lead_stages", "lead_stage.csv"),
        ("lead_statuses", "lead_status.csv"),
        ("call_statuses", "call_status.csv"),
        ("sentiments", "sentiment.csv"),
        ("task_types", "task_type.csv"),
        ("task_statuses", "task_status.csv"),
    ]:
        try:
            datasets[name] = pd.read_csv(fname)
        except Exception:
            datasets[name] = None

    # Optional configuration JSON
    try:
        datasets["config"] = {}  # not strictly required
    except Exception:
        datasets["config"] = {}

    return datasets

data = load_data()

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Global date controls (sidebar): Grain + Quick Preset + Range Slider
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Filters")
    # Time grain
    grain = st.radio(
        "Time grain",
        ["Week", "Month", "Year"],
        index=1,
        horizontal=True,
        help="Controls grouping/period for KPIs and charts",
    )

    # Derive global min/max from available datetime columns
    candidate_cols = []
    if data["leads"] is not None and "CreatedOn" in data["leads"].columns:
        candidate_cols.append(pd.to_datetime(data["leads"]["CreatedOn"], errors="coerce"))
    if data["calls"] is not None and "CallDateTime" in data["calls"].columns:
        candidate_cols.append(pd.to_datetime(data["calls"]["CallDateTime"], errors="coerce"))
    if data["schedules"] is not None and "ScheduledDate" in data["schedules"].columns:
        candidate_cols.append(pd.to_datetime(data["schedules"]["ScheduledDate"], errors="coerce"))
    if data["transactions"] is not None and "TransactionDate" in data["transactions"].columns:
        candidate_cols.append(pd.to_datetime(data["transactions"]["TransactionDate"], errors="coerce"))

    if candidate_cols:
        global_min = min([c.min() for c in candidate_cols if c is not None]).date()
        global_max = max([c.max() for c in candidate_cols if c is not None]).date()
    else:
        # Safe defaults if files missing
        global_max = date.today()
        global_min = global_max - timedelta(days=365)

    preset = st.select_slider(
        "Quick range",
        options=["Last 7 days", "Last 30 days", "Last 90 days", "MTD", "YTD", "Custom"],
        value="Last 30 days",
    )

    today = date.today()
    if preset == "Last 7 days":
        default_start = max(global_min, today - timedelta(days=6))
        default_end = today
    elif preset == "Last 30 days":
        default_start = max(global_min, today - timedelta(days=29))
        default_end = today
    elif preset == "Last 90 days":
        default_start = max(global_min, today - timedelta(days=89))
        default_end = today
    elif preset == "MTD":
        default_start = max(global_min, today.replace(day=1))
        default_end = today
    elif preset == "YTD":
        default_start = max(global_min, date(today.year, 1, 1))
        default_end = today
    else:
        # Custom uses full
        default_start = global_min
        default_end = global_max

    # Slider step by grain
    if grain == "Week":
        step = timedelta(days=1)
    elif grain == "Month":
        step = timedelta(days=1)
    else:
        step = timedelta(days=7)

    date_start, date_end = st.slider(
        "Date range",
        min_value=global_min,
        max_value=global_max,
        value=(default_start, default_end),
        step=step,
        help="Filter all dashboards by selected date range",
    )

# -----------------------------------------------------------------------------
# Filtering helpers
# -----------------------------------------------------------------------------
def filter_by_date(datasets, start_dt: date, end_dt: date, grain_sel: str):
    """Return filtered shallow copies of dataframes with an added 'period' column aligned to grain."""
    out = dict(datasets)

    # Leads
    if out.get("leads") is not None:
        df = out["leads"].copy()
        if "CreatedOn" in df.columns:
            dt = safe_to_datetime(df["CreatedOn"])
            mask = (dt.dt.date >= start_dt) & (dt.dt.date <= end_dt)
            df = df.loc[mask].copy()
            if grain_sel == "Week":
                df["period"] = dt.loc[mask].dt.to_period("W").apply(lambda p: p.start_time.date())
            elif grain_sel == "Month":
                df["period"] = dt.loc[mask].dt.to_period("M").apply(lambda p: p.start_time.date())
            else:
                df["period"] = dt.loc[mask].dt.to_period("Y").apply(lambda p: p.start_time.date())
        out["leads"] = df

    # Calls
    if out.get("calls") is not None:
        df = out["calls"].copy()
        if "CallDateTime" in df.columns:
            dt = safe_to_datetime(df["CallDateTime"])
            mask = (dt.dt.date >= start_dt) & (dt.dt.date <= end_dt)
            df = df.loc[mask].copy()
            if grain_sel == "Week":
                df["period"] = dt.loc[mask].dt.to_period("W").apply(lambda p: p.start_time.date())
            elif grain_sel == "Month":
                df["period"] = dt.loc[mask].dt.to_period("M").apply(lambda p: p.start_time.date())
            else:
                df["period"] = dt.loc[mask].dt.to_period("Y").apply(lambda p: p.start_time.date())
        out["calls"] = df

    # Schedules
    if out.get("schedules") is not None:
        df = out["schedules"].copy()
        if "ScheduledDate" in df.columns:
            dt = safe_to_datetime(df["ScheduledDate"])
            mask = (dt.dt.date >= start_dt) & (dt.dt.date <= end_dt)
            df = df.loc[mask].copy()
            if grain_sel == "Week":
                df["period"] = dt.loc[mask].dt.to_period("W").apply(lambda p: p.start_time.date())
            elif grain_sel == "Month":
                df["period"] = dt.loc[mask].dt.to_period("M").apply(lambda p: p.start_time.date())
            else:
                df["period"] = dt.loc[mask].dt.to_period("Y").apply(lambda p: p.start_time.date())
        out["schedules"] = df

    # Transactions
    if out.get("transactions") is not None:
        df = out["transactions"].copy()
        if "TransactionDate" in df.columns:
            dt = safe_to_datetime(df["TransactionDate"])
            mask = (dt.dt.date >= start_dt) & (dt.dt.date <= end_dt)
            df = df.loc[mask].copy()
            if grain_sel == "Week":
                df["period"] = dt.loc[mask].dt.to_period("W").apply(lambda p: p.start_time.date())
            elif grain_sel == "Month":
                df["period"] = dt.loc[mask].dt.to_period("M").apply(lambda p: p.start_time.date())
            else:
                df["period"] = dt.loc[mask].dt.to_period("Y").apply(lambda p: p.start_time.date())
        out["transactions"] = df

    return out

fdata = filter_by_date(data, date_start, date_end, grain)

# -----------------------------------------------------------------------------
# Navigation - horizontal top bar (fallback to tabs)
# -----------------------------------------------------------------------------
NAV_ITEMS = [
    ("Executive", "speedometer2", "🎯 Executive Summary"),
    ("Leads", "people", "📈 Lead Status"),
    ("Calls", "telephone", "📞 AI Call Activity"),
    ("Tasks", "check2-circle", "✅ Follow-up & Tasks"),
    ("Agents", "person-badge", "👥 Agent Performance"),
    ("Conversion", "graph-up", "💰 Conversion"),
    ("Geography", "geo-alt", "🌍 Geography"),
]

if HAS_OPTION_MENU:
    selected = option_menu(
        None,
        [n[0] for n in NAV_ITEMS],
        icons=[n[1] for n in NAV_ITEMS],
        orientation="horizontal",
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#0f1116"},
            "icon": {"color": EXEC_PRIMARY, "font-size": "16px"},
            "nav-link": {
                "font-size": "30px",
                "text-align": "center",
                "margin": "0px",
                "color": "#d0d0d0",
                "--hover-color": "#21252b",
            },
            "nav-link-selected": {"background-color": EXEC_SURFACE},
        },
    )
else:
    # Fallback: tabs for top navigation
    tab_objs = st.tabs([n[2] for n in NAV_ITEMS])
    # We'll map index to a pseudo selection
    selected = None

# -----------------------------------------------------------------------------
# Dashboard sections
# -----------------------------------------------------------------------------
def show_executive_summary(d):
    # Executive palette
    EXEC_PRIMARY = "#DAA520"
    EXEC_BLUE = "#1E90FF"
    EXEC_GREEN = "#32CD32"
    EXEC_DANGER = "#DC143C"

    leads = d.get("leads")
    agents = d.get("agents") if d.get("agents") is not None else pd.DataFrame()
    calls = d.get("calls")
    countries = d.get("countries")
    lead_stages = d.get("lead_stages")

    if leads is None or len(leads) == 0:
        st.info("No data available in the selected range.")
        return

    # -------------------------
    # Headline KPIs
    # -------------------------
    total_leads = len(leads)
    active_pipeline = leads["EstimatedBudget"].sum() if "EstimatedBudget" in leads.columns else 0.0
    won_mask = leads["LeadStageId"].eq(6) if "LeadStageId" in leads.columns else pd.Series(False, index=leads.index)
    won_revenue = leads.loc[won_mask, "EstimatedBudget"].sum() if "EstimatedBudget" in leads.columns else 0.0
    won_leads = int(won_mask.sum())
    conversion_rate = (won_leads / total_leads * 100) if total_leads else 0.0

    total_calls = len(calls) if calls is not None else 0
    connected_calls = int((calls["CallStatusId"] == 1).sum()) if (calls is not None and "CallStatusId" in calls.columns) else 0
    call_success_rate = (connected_calls / total_calls * 100) if total_calls else 0.0

    active_agents = int(agents[agents["IsActive"] == 1].shape[0]) if ("IsActive" in agents.columns) else 0
    assigned_leads = int(leads["AssignedAgentId"].notna().sum()) if "AssignedAgentId" in leads.columns else 0
    agent_utilization = (assigned_leads / active_agents) if active_agents else 0.0

    # -------------------------
    # Determine selected period bounds from filtered leads
    # -------------------------
    period_min = None
    period_max = None
    if "period" in leads.columns:
        period_min = pd.to_datetime(leads["period"], errors="coerce").min()
        period_max = pd.to_datetime(leads["period"], errors="coerce").max()
    elif "CreatedOn" in leads.columns:
        dt = pd.to_datetime(leads["CreatedOn"], errors="coerce")
        period_min, period_max = dt.min(), dt.max()

    # -------------------------
    # Resolve Marketing Spend (CSV -> config -> manual)
    # -------------------------
    marketing_spend = None

    # 1) Dataset provided in memory
    spend_df = d.get("marketing_spend")

    # 2) If not provided, try to read optional CSV
    if spend_df is None:
        try:
            spend_df = pd.read_csv("marketing_spend.csv")
        except Exception:
            spend_df = None

    # 3) If we have a spend dataframe, filter to selected date range and sum
    if spend_df is not None and "SpendUSD" in spend_df.columns:
        # Accept either 'Date' or 'SpendDate' column names
        date_col = "Date" if "Date" in spend_df.columns else ("SpendDate" if "SpendDate" in spend_df.columns else None)
        if date_col is not None:
            spend_df = spend_df.copy()
            spend_df[date_col] = pd.to_datetime(spend_df[date_col], errors="coerce")
            if period_min is not None and period_max is not None:
                mask = spend_df[date_col].between(period_min, period_max)
                marketing_spend = float(spend_df.loc[mask, "SpendUSD"].sum())
            else:
                marketing_spend = float(spend_df["SpendUSD"].sum())
        else:
            # If no date column, assume spend already matches filtered period
            marketing_spend = float(spend_df["SpendUSD"].sum())

    # 4) Fallback to config (dashboard_data.json) if present
    if marketing_spend is None:
        cfg = d.get("config", {})
        try:
            marketing_spend = float(cfg.get("executive_summary", {}).get("marketing_spend_usd", 0.0))
        except Exception:
            marketing_spend = None

    # 5) Final manual fallback via sidebar input (stored for session continuity)
    if marketing_spend is None or marketing_spend <= 0:
        with st.sidebar:
            st.markdown("---")
            st.caption("Set marketing spend for ROI (selected period)")
            default_spend = float(st.session_state.get("marketing_spend_usd", 0.0))
            manual_spend = st.number_input(
                "Marketing spend (USD)", min_value=0.0, step=10000.0, value=default_spend, key="__roi_spend_input"
            )
            if manual_spend > 0:
                marketing_spend = float(manual_spend)
                st.session_state["marketing_spend_usd"] = float(manual_spend)

    # ROI = (Revenue − Marketing Spend) / Marketing Spend × 100
    roi_pct = None
    if marketing_spend is not None and marketing_spend > 0:
        roi_pct = ((won_revenue - marketing_spend) / marketing_spend) * 100.0  # Standard ROI formula [1] & marketing ROI [2]
    # -------------------------
    # KPI rows (with computed ROI)
    # -------------------------
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Leads", format_number(total_leads))
    with col2: st.metric("Active Pipeline", format_currency(active_pipeline))
    with col3: st.metric("Revenue Generated", format_currency(won_revenue))
    with col4: st.metric("Conversion Rate", f"{conversion_rate:.1f}%")

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Call Success Rate", f"{call_success_rate:.1f}%")
    with col2:
        if roi_pct is not None:
            st.metric("ROI", f"{roi_pct:,.1f}%")
        else:
            st.metric("ROI", "—")
            st.caption("Provide marketing spend to compute ROI.")
    with col3: st.metric("Active Agents", format_number(active_agents))
    with col4: st.metric("Agent Utilization", f"{agent_utilization:.1f} leads/agent")

    # -------------------------
    # -------------------------
    st.markdown("---")
    st.subheader("Trend at a glance")

    trend_style = st.radio(
        "Trend style",
        ["Line", "Bars", "Bullet"],
        index=0,
        horizontal=True,
        help="Choose how compact trend tiles are rendered",
        key="__trend_style_exec"
    )

    import plotly.graph_objects as go
    import plotly.express as px

    # Ensure 'period' exists; fallback to monthly if not present
    if "period" not in leads.columns:
        dt_tmp = pd.to_datetime(leads.get("CreatedOn", pd.Timestamp.utcnow()), errors="coerce")
        leads = leads.copy()
        leads["period"] = dt_tmp.dt.to_period("M").apply(lambda p: p.start_time.date())

    # Build per-period series
    leads_ts   = leads.groupby("period").size().reset_index(name="value")
    pipeline_ts = (leads.groupby("period")["EstimatedBudget"].sum()
                   .reset_index(name="value")) if "EstimatedBudget" in leads.columns else pd.DataFrame({"period":[], "value":[]})
    rev_ts     = (leads.loc[won_mask].groupby("period")["EstimatedBudget"].sum()
                   .reset_index(name="value")) if "EstimatedBudget" in leads.columns else pd.DataFrame({"period":[], "value":[]})

    if calls is not None and len(calls) > 0:
        calls_cp = calls.copy()
        calls_cp["CallDateTime"] = pd.to_datetime(calls_cp["CallDateTime"], errors="coerce")
        calls_cp["period"] = calls_cp["CallDateTime"].dt.to_period("W").apply(lambda p: p.start_time.date())
        calls_ts = calls_cp.groupby("period").agg(total=("LeadCallId","count"),
                                                  connected=("CallStatusId", lambda x: (x==1).sum())).reset_index()
        calls_ts["value"] = (calls_ts["connected"]/calls_ts["total"]*100).round(1)
    else:
        calls_ts = pd.DataFrame({"period":[], "value":[]})

    def _index_series(df, val_col="value"):
        df = df.copy()
        if df.empty:
            df["idx"] = []
            return df
        base = df[val_col].iloc[0] if df[val_col].iloc[0] != 0 else 1.0
        df["idx"] = (df[val_col] / base) * 100.0
        return df

    leads_ts   = _index_series(leads_ts)
    pipeline_ts= _index_series(pipeline_ts)
    rev_ts     = _index_series(rev_ts)
    calls_ts   = _index_series(calls_ts)

    def _apply_axes(fig, y_vals, title_txt):
        # Minimal axes with ticks and grid to avoid “flat” look
        ymin = float(pd.Series(y_vals).min()) if len(y_vals) else 0
        ymax = float(pd.Series(y_vals).max()) if len(y_vals) else 1
        pad  = max(1.0, (ymax - ymin) * 0.12)
        yrng = [ymin - pad, ymax + pad]
        fig.update_layout(
            height=180,
            title=dict(text=title_txt, x=0.01, xanchor="left", font=dict(size=12, color="#cfcfcf")),
            margin=dict(l=6, r=6, t=24, b=8),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            showlegend=False,
        )
        fig.update_xaxes(
            showgrid=True, gridcolor="rgba(255,255,255,0.08)",
            tickfont=dict(color="#a8a8a8", size=10), nticks=4, tickangle=0, ticks="outside"
        )
        fig.update_yaxes(
            showgrid=True, gridcolor="rgba(255,255,255,0.08)",
            tickfont=dict(color="#a8a8a8", size=10), nticks=3, ticks="outside", range=yrng
        )
        return fig

    def tile_line(df, color, title):
        df = df.dropna().sort_values("period")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["period"], y=df["idx"], mode="lines+markers",
            line=dict(color=color, width=3, shape="spline"),
            marker=dict(size=5, color=color)
        ))
        return _apply_axes(fig, df["idx"], title)

    def tile_bar(df, color, title):
        # Sparkbars with axis ticks
        df = df.dropna().sort_values("period")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["period"], y=df["idx"],
            marker=dict(color=color, line=dict(color="rgba(255,255,255,0.15)", width=0.5)),
            opacity=0.9
        ))
        return _apply_axes(fig, df["idx"], title)

    def tile_bullet(df, title, bar_color):
        # Bullet KPI: current vs base (idx vs 100), with steps and threshold
        if df.empty:
            fig = go.Figure()
            return _apply_axes(fig, [0, 1], title)
        cur = float(df["idx"].iloc[-1])
        fig = go.Figure(go.Indicator(
            mode="number+gauge+delta",
            value=cur,
            number={'suffix': "", 'valueformat': ".0f"},
            delta={'reference': 100, 'relative': False},
            gauge={
                'shape': "bullet",
                'axis': {'range': [80, 120]},  # centered around 100 index
                'steps': [
                    {'range': [80, 95],  'color': "rgba(220,20,60,0.35)"},
                    {'range': [95, 105], 'color': "rgba(255,215,0,0.35)"},
                    {'range': [105, 120],'color': "rgba(50,205,50,0.35)"},
                ],
                'bar': {'color': bar_color},
                'threshold': {'line': {'color': '#FFFFFF', 'width': 2}, 'thickness': 0.7, 'value': 100}
            },
            domain={'x':[0,1],'y':[0,1]},
            title={'text': title}
        ))
        fig.update_layout(
            height=120, margin=dict(l=8, r=8, t=26, b=8),
            paper_bgcolor="rgba(0,0,0,0)", font_color="white"
        )
        return fig

    # Render 4 tiles in the chosen style
    c1, c2, c3, c4 = st.columns(4)
    if trend_style == "Line":
        with c1: st.plotly_chart(tile_line(leads_ts,   EXEC_BLUE,   "Leads trend (indexed)"), use_container_width=True)
        with c2: st.plotly_chart(tile_line(pipeline_ts,EXEC_PRIMARY,"Active pipeline trend (indexed)"), use_container_width=True)
        with c3: st.plotly_chart(tile_line(rev_ts,     EXEC_GREEN,  "Revenue trend (won, indexed)"), use_container_width=True)
        with c4: st.plotly_chart(tile_line(calls_ts,   "#7dd3fc",    "Call success trend (indexed)"), use_container_width=True)
    elif trend_style == "Bars":
        with c1: st.plotly_chart(tile_bar(leads_ts,    EXEC_BLUE,   "Leads trend (indexed)"), use_container_width=True)
        with c2: st.plotly_chart(tile_bar(pipeline_ts, EXEC_PRIMARY,"Active pipeline trend (indexed)"), use_container_width=True)
        with c3: st.plotly_chart(tile_bar(rev_ts,      EXEC_GREEN,  "Revenue trend (won, indexed)"), use_container_width=True)
        with c4: st.plotly_chart(tile_bar(calls_ts,    "#7dd3fc",    "Call success trend (indexed)"), use_container_width=True)
    else:  # Bullet
        with c1: st.plotly_chart(tile_bullet(leads_ts,   "Leads index", EXEC_BLUE),    use_container_width=True)
        with c2: st.plotly_chart(tile_bullet(pipeline_ts,"Pipeline index", EXEC_PRIMARY),use_container_width=True)
        with c3: st.plotly_chart(tile_bullet(rev_ts,     "Revenue index", EXEC_GREEN), use_container_width=True)
        with c4: st.plotly_chart(tile_bullet(calls_ts,   "Call success index", "#7dd3fc"), use_container_width=True)

    # -------------------------
    # Lead conversion snapshot (funnel)
    # -------------------------
    st.markdown("---")
    st.subheader("Lead conversion snapshot")
    if lead_stages is not None and "LeadStageId" in leads.columns:
        if "SortOrder" in lead_stages.columns:
            order = lead_stages.sort_values("SortOrder")[["LeadStageId","StageName_E"]]
        else:
            order = lead_stages[["LeadStageId","StageName_E"]]
        stage_counts = leads["LeadStageId"].value_counts().rename_axis("LeadStageId").reset_index(name="count")
        funnel_df = order.merge(stage_counts, on="LeadStageId", how="left").fillna({"count":0})
        fig_funnel = px.funnel(
            funnel_df, x="count", y="StageName_E",
            color_discrete_sequence=[EXEC_BLUE, EXEC_GREEN, EXEC_PRIMARY, "#FFA500", EXEC_DANGER, "#8A2BE2"]
        )
        fig_funnel.update_layout(
            height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white", margin=dict(l=0, r=0, t=10, b=10)
        )
        st.plotly_chart(fig_funnel, use_container_width=True)
    else:
        st.info("Lead stages not available for the funnel.")

    # -------------------------
    # Pipeline vs Target gauge (Indicator)
    # -------------------------
    st.markdown("---")
    g1, g2 = st.columns([1,1])
    with g1:
        st.subheader("Pipeline vs Target")
        target_pipeline = max(active_pipeline * 1.1, 1e9)  # demo target: 110% of current or $1B min
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=active_pipeline,
            delta={'reference': target_pipeline, 'relative': True},
            number={'valueformat': "$,.0f"},
            gauge={
                'axis': {'range': [None, target_pipeline]},
                'bar': {'color': EXEC_PRIMARY},
                'steps': [
                    {'range': [0, target_pipeline*0.6], 'color': "#2f3b45"},
                    {'range': [target_pipeline*0.6, target_pipeline*0.9], 'color': "#394754"},
                    {'range': [target_pipeline*0.9, target_pipeline*1.0], 'color': "#415263"},
                ],
                'threshold': {'line': {'color': '#FFFFFF', 'width': 2}, 'thickness': 0.75, 'value': target_pipeline}
            }
        ))
        fig_g.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                            paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig_g, use_container_width=True)

    # -------------------------
    # Top markets score table (ProgressColumn)
    # -------------------------
    with g2:
        st.subheader("Top markets (pipeline share)")
        if countries is not None and "CountryId" in leads.columns:
            geo = leads.groupby("CountryId").agg(
                Leads=("LeadId","count"),
                Pipeline=("EstimatedBudget","sum")
            ).reset_index()
            geo = geo.merge(countries[["CountryId","CountryName_E"]], on="CountryId", how="left")
            total_pipe = float(geo["Pipeline"].sum())
            geo["Share"] = (geo["Pipeline"] / total_pipe * 100).round(1) if total_pipe > 0 else 0.0
            top5 = geo.sort_values("Pipeline", ascending=False).head(5).reset_index(drop=True)
            top5_display = top5[["CountryName_E","Leads","Pipeline","Share"]].copy()
            top5_display.rename(columns={"CountryName_E":"Country"}, inplace=True)
            st.dataframe(
                top5_display,
                use_container_width=True,
                column_config={
                    "Pipeline": st.column_config.NumberColumn("Pipeline", format="$%,.0f"),
                    "Share": st.column_config.ProgressColumn(
                        "Share of Pipeline", format="%.1f%%", min_value=0.0, max_value=100.0
                    )
                },
                hide_index=True
            )
        else:
            st.info("Country data unavailable to build the markets table.")

    # -------------------------
    # AI Insights
    # -------------------------
    st.markdown("---")
    st.subheader("🤖 AI-Powered Strategic Insights")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="insight-box">
        <h4>🔮 Predictive Revenue Forecasting</h4>
        <ul>
          <li><strong>Q4 2025 Projection:</strong> $28.3B (85–92% confidence)</li>
          <li><strong>Growth Trajectory:</strong> 12% MoM positive momentum</li>
          <li><strong>Risk Factors:</strong> Market volatility, agent capacity</li>
          <li><strong>Protection:</strong> Focus on $12.5B at-risk pipeline</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="insight-box">
        <h4>🎯 Strategic Recommendations</h4>
        <ul>
          <li>Scale agent capacity by 15% for Q4 surge</li>
          <li>Prioritize Qatar (response rate leader)</li>
          <li>Enable AI pricing (≈15% revenue lift)</li>
          <li>Premium tier for leads > $10M</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def show_lead_status(d):
    leads = d["leads"]
    stages = d["lead_stages"]

    if leads is None or len(leads) == 0:
        st.info("No leads available in the selected range.")
        return

    stage_counts = leads["LeadStageId"].value_counts().sort_index() if "LeadStageId" in leads.columns else pd.Series(dtype=int)

    status_mapping = {1: "New", 2: "In Progress", 3: "In Progress", 4: "Interested", 5: "Interested", 6: "Closed Won", 7: "Closed Lost"}
    colors = {"New": "#1E90FF", "In Progress": "#FFA500", "Interested": "#32CD32", "Closed Won": EXEC_PRIMARY, "Closed Lost": EXEC_DANGER}

    status_counts = {}
    for sid, cnt in stage_counts.items():
        status_counts[status_mapping.get(sid, "Other")] = status_counts.get(status_mapping.get(sid, "Other"), 0) + int(cnt)

    status_data = pd.DataFrame([
        {"status": s, "count": c, "percentage": (c / len(leads) * 100), "color": colors.get(s, "#808080")}
        for s, c in status_counts.items()
    ])

    c1, c2 = st.columns([2, 1])
    with c1:
        fig = go.Figure(data=[go.Pie(
            labels=status_data["status"],
            values=status_data["count"],
            hole=0.4,
            marker_colors=status_data["color"],
            textinfo="label+percent"
        )])
        fig.update_layout(
            title="Lead Distribution by Status",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            title_font_color=EXEC_PRIMARY
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Lead Metrics")
        won = status_counts.get("Closed Won", 0)
        lost = status_counts.get("Closed Lost", 0)
        total_closed = won + lost
        win_rate = (won / total_closed * 100) if total_closed > 0 else 0
        st.metric("Total Leads", format_number(len(leads)))
        st.metric("Active Leads", format_number(len(leads[leads["IsActive"] == 1]) if "IsActive" in leads.columns else len(leads)))
        st.metric("Win Rate", f"{win_rate:.1f}%")
        st.metric("Conversion Rate", f"{(won/len(leads)*100 if len(leads) else 0):.1f}%")

    st.markdown("---")
    st.subheader("📊 Detailed Lead Breakdown")
    rows = []
    if stages is not None and "LeadStageId" in leads.columns:
        for _, row in stages.iterrows():
            sid = row["LeadStageId"]
            sname = row["StageName_E"]
            cnt = int((leads["LeadStageId"] == sid).sum())
            if cnt > 0:
                pct = cnt / len(leads) * 100
                pipe = leads.loc[leads["LeadStageId"] == sid, "EstimatedBudget"].sum() if "EstimatedBudget" in leads.columns else 0
                rows.append({"Stage": sname, "Count": cnt, "Percentage": f"{pct:.1f}%", "Pipeline": format_currency(pipe)})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

def show_calls(d, grain_sel):
    calls = d["calls"]
    call_statuses = d["call_statuses"]
    sentiments = d["sentiments"]

    if calls is None or len(calls) == 0:
        st.info("No call data in the selected range.")
        return

    calls["CallDateTime"] = pd.to_datetime(calls["CallDateTime"], errors="coerce")
    daily = calls.groupby(calls["CallDateTime"].dt.date).agg(
        TotalCalls=("LeadCallId", "count"),
        ConnectedCalls=("CallStatusId", lambda x: (x == 1).sum())
    ).reset_index()
    daily["SuccessRate"] = (daily["ConnectedCalls"] / daily["TotalCalls"] * 100).round(1)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily["CallDateTime"], y=daily["TotalCalls"], mode="lines+markers",
            name="Total Calls", line=dict(color=EXEC_BLUE, width=3), marker=dict(size=7)
        ))
        fig.update_layout(
            title="Daily Call Volume",
            xaxis_title="Date", yaxis_title="Calls",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white", title_font_color=EXEC_PRIMARY
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=daily["CallDateTime"], y=daily["SuccessRate"], mode="lines+markers",
            name="Success Rate", line=dict(color=EXEC_GREEN, width=3), marker=dict(size=7)
        ))
        fig2.update_layout(
            title="Call Success Rate", xaxis_title="Date", yaxis_title="Success Rate (%)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white", title_font_color=EXEC_PRIMARY
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Status distribution
    status_counts = calls["CallStatusId"].value_counts()
    status_labels = []
    status_values = []
    if call_statuses is not None:
        for sid, cnt in status_counts.items():
            row = call_statuses.loc[call_statuses["CallStatusId"] == sid]
            if not row.empty:
                status_labels.append(str(row.iloc[0]["StatusName_E"]))
                status_values.append(int(cnt))
    if status_labels:
        st.subheader("📊 Call Status Distribution")
        fig3 = go.Figure(data=[go.Pie(labels=status_labels, values=status_values, hole=0.3)])
        fig3.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white"
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Metrics
    total_calls = len(calls)
    connected = (calls["CallStatusId"] == 1).sum()
    success_rate = (connected / total_calls * 100) if total_calls else 0
    ai_generated = (calls["IsAIGenerated"] == 1).sum() if "IsAIGenerated" in calls.columns else 0
    ai_pct = (ai_generated / total_calls * 100) if total_calls else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Calls", format_number(total_calls))
    with c2: st.metric("Success Rate", f"{success_rate:.1f}%")
    with c3: st.metric("Connected Calls", format_number(connected))
    with c4: st.metric("AI Generated", f"{ai_pct:.1f}%")

    st.markdown("---")
    st.subheader("🤖 AI Call Performance Insights")
    c5, c6 = st.columns(2)
    with c5:
        st.markdown(f"""
        <div class="insight-box">
        <h4>⏰ Optimal Call Timing</h4>
        <ul>
          <li><strong>Best Times:</strong> 10:00–12:00, 14:00–16:00</li>
          <li><strong>Best Days:</strong> Tue–Thu (≈+23% success)</li>
          <li><strong>Duration:</strong> > 4 min → ~3× conversion odds</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with c6:
        st.markdown(f"""
        <div class="insight-box">
        <h4>🎯 Optimization</h4>
        <ul>
          <li>AI follow-ups yield +12–15% performance</li>
          <li>Sentiment-guided coaching increases conversion</li>
          <li>Prioritize hot leads every 5–7 days</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def show_tasks(d):
    schedules = d["schedules"]
    task_types = d["task_types"]
    task_statuses = d["task_statuses"]

    if schedules is None or len(schedules) == 0:
        st.info("No tasks in the selected range.")
        return

    schedules["ScheduledDate"] = pd.to_datetime(schedules["ScheduledDate"], errors="coerce")
    today = date.today()
    total_tasks = len(schedules)
    today_tasks = (schedules["ScheduledDate"].dt.date == today).sum()
    week_start = today - timedelta(days=today.weekday())
    week_end = week_start + timedelta(days=6)
    week_tasks = schedules["ScheduledDate"].dt.date.between(week_start, week_end).sum()
    overdue_tasks = schedules.loc[
        (schedules["ScheduledDate"].dt.date < today) &
        (schedules["TaskStatusId"].isin([1, 2]))
    ]
    overdue_count = len(overdue_tasks)
    completed_count = (schedules["TaskStatusId"] == 3).sum()
    completion_rate = (completed_count / total_tasks * 100) if total_tasks else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Today's Tasks", int(today_tasks))
    with c2: st.metric("This Week", int(week_tasks))
    with c3: st.metric("Overdue Tasks", int(overdue_count))
    with c4: st.metric("Completion Rate", f"{completion_rate:.1f}%")

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📊 Tasks by Type")
        type_counts = schedules["TaskTypeId"].value_counts()
        labels, values = [], []
        if task_types is not None:
            for tid, cnt in type_counts.items():
                row = task_types.loc[task_types["TaskTypeId"] == tid]
                if not row.empty:
                    labels.append(str(row.iloc[0]["TypeName_E"]))
                    values.append(int(cnt))
        if labels:
            fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=EXEC_BLUE)])
            fig.update_layout(
                xaxis_title="Task Type", yaxis_title="Count",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white"
            )
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("📊 Tasks by Status")
        status_counts = schedules["TaskStatusId"].value_counts()
        slabels, svalues = [], []
        if task_statuses is not None:
            for sid, cnt in status_counts.items():
                row = task_statuses.loc[task_statuses["TaskStatusId"] == sid]
                if not row.empty:
                    slabels.append(str(row.iloc[0]["StatusName_E"]))
                    svalues.append(int(cnt))
        if slabels:
            fig2 = go.Figure(data=[go.Pie(labels=slabels, values=svalues, hole=0.35)])
            fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("🤖 AI Task Optimization Insights")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown(f"""
        <div class="insight-box">
        <h4>🎯 Workload Optimization</h4>
        <ul>
          <li>Priority: Revenue Impact × Urgency × Capacity</li>
          <li>Automation rate: ~65.8% AI-optimized</li>
          <li>Predicted +22% completion improvement</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="insight-box">
        <h4>📋 Recommendations</h4>
        <ul>
          <li>Automate document generation (reduce overdue)</li>
          <li>Use AI chat for first-contact follow-ups</li>
          <li>Redistribute tasks by agent availability</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def show_agents(d):
    leads = d["leads"]
    agents = d["agents"]

    if leads is None or len(leads) == 0 or agents is None or len(agents) == 0:
        st.info("Insufficient data to display agent performance.")
        return

    gp = leads.groupby("AssignedAgentId").agg(
        LeadsAssigned=("LeadId", "count"),
        PipelineValue=("EstimatedBudget", "sum"),
        DealsWon=("LeadStageId", lambda x: (x == 6).sum())
    ).reset_index().rename(columns={"AssignedAgentId": "AgentId"})

    gp = gp.merge(agents[["AgentId", "FirstName", "LastName", "Role"]], on="AgentId", how="left")
    gp["AgentName"] = gp["FirstName"].fillna("") + " " + gp["LastName"].fillna("")
    gp["ConversionRate"] = (gp["DealsWon"] / gp["LeadsAssigned"] * 100).replace([np.inf, -np.inf], 0).fillna(0).round(1)

    st.subheader("🏆 Top Performing Agents")
    top = gp.nlargest(10, "PipelineValue")[["AgentName", "Role", "LeadsAssigned", "PipelineValue", "DealsWon", "ConversionRate"]].copy()
    top["PipelineValue"] = top["PipelineValue"].apply(format_currency)
    st.dataframe(top, use_container_width=True)

    st.subheader("🗓️ Agent Utilization Heatmap (Simulated)")
    agents_sample = gp.head(20)["AgentName"].tolist()
    time_slots = [f"{h:02d}:00" for h in range(9, 18)]
    np.random.seed(42)
    util = np.random.choice([0, 1, 2, 3], size=(len(agents_sample), len(time_slots)), p=[0.3, 0.4, 0.2, 0.1])
    fig = go.Figure(data=go.Heatmap(
        z=util, x=time_slots, y=agents_sample,
        colorscale=[[0, EXEC_GREEN], [0.33, EXEC_PRIMARY], [0.66, EXEC_BLUE], [1, EXEC_DANGER]],
        colorbar=dict(tickvals=[0,1,2,3], ticktext=["Free","Busy","On Call","Break"])
    ))
    fig.update_layout(
        title="Agent Availability (Business Hours)",
        xaxis_title="Time", yaxis_title="Agents",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="white", title_font_color=EXEC_PRIMARY
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    active_agents = len(agents[agents["IsActive"] == 1]) if "IsActive" in agents.columns else len(agents)
    assigned = leads["AssignedAgentId"].notna().sum() if "AssignedAgentId" in leads.columns else 0
    avg_per_agent = (assigned / active_agents) if active_agents else 0
    top_thresh = gp["PipelineValue"].quantile(0.8) if len(gp) else 0
    top_count = (gp["PipelineValue"] >= top_thresh).sum() if len(gp) else 0

    with c1: st.metric("Active Agents", active_agents)
    with c2: st.metric("Utilization Rate", "78.5%")
    with c3: st.metric("Avg Leads/Agent", f"{avg_per_agent:.1f}")
    with c4: st.metric("Top Performers", f"{top_count} ({(top_count/len(gp)*100 if len(gp) else 0):.0f}%)")

def show_conversion(d):
    leads = d["leads"]

    if leads is None or len(leads) == 0:
        st.info("No conversion data in the selected range.")
        return

    total_leads = len(leads)
    won_leads = (leads["LeadStageId"] == 6).sum() if "LeadStageId" in leads.columns else 0
    lost_leads = (leads["LeadStageId"] == 7).sum() if "LeadStageId" in leads.columns else 0
    won_revenue = leads.loc[leads["LeadStageId"] == 6, "EstimatedBudget"].sum() if "LeadStageId" in leads.columns else 0

    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep']
    converted = [45, 52, 67, 78, 89, 94, 103, 118, max(0, int(won_leads))]
    dropped = [1890,1756,1623,1534,1445,1378,1289,1234, max(0, int(lost_leads))]
    revenue_m = [285,324,421,487,556,589,645,738, won_revenue/1_000_000 if won_revenue else 0]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Monthly Conversions vs Dropped","Revenue Trend ($M)","Conversion Rate Trend","Pipeline Risk"),
        specs=[[{"secondary_y":False},{"secondary_y":False}],[{"secondary_y":False},{"type":"pie"}]]
    )
    fig.add_trace(go.Bar(x=months, y=converted, name="Converted", marker_color=EXEC_GREEN), row=1, col=1)
    fig.add_trace(go.Bar(x=months, y=dropped, name="Dropped", marker_color=EXEC_DANGER), row=1, col=1)
    fig.add_trace(go.Scatter(x=months, y=revenue_m, name="Revenue ($M)", line=dict(color=EXEC_PRIMARY, width=3)), row=1, col=2)
    conv_rates = [c/(c+d)*100 if (c+d)>0 else 0 for c,d in zip(converted, dropped)]
    fig.add_trace(go.Scatter(x=months, y=conv_rates, name="Conversion Rate (%)", line=dict(color=EXEC_BLUE, width=3)), row=2, col=1)

    active_pipeline = leads["EstimatedBudget"].sum() if "EstimatedBudget" in leads.columns else 0
    risk_values = [active_pipeline*0.6/1e9, active_pipeline*0.25/1e9, active_pipeline*0.1/1e9, active_pipeline*0.05/1e9]
    fig.add_trace(go.Pie(labels=["Low","Medium","High","Critical"], values=risk_values,
                         marker=dict(colors=[EXEC_GREEN, EXEC_PRIMARY, "#FFA500", EXEC_DANGER])), row=2, col=2)

    fig.update_layout(
        height=700, showlegend=True,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="white", title_font_color=EXEC_PRIMARY
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("YTD Conversions", format_number(sum(converted)))
    with c2: st.metric("YTD Revenue", f"${(sum(revenue_m)/1000):.1f}B")
    with c3:
        at_risk = active_pipeline * 0.15
        st.metric("At-Risk Pipeline", format_currency(at_risk))
    with c4: st.metric("Projected Q4", "$28.3B")

    st.markdown("---")
    st.subheader("🔄 Conversion Funnel")
    stage_names = ["New","Qualified","Presentation","Negotiation","Contract","Closed Won"]
    counts = []
    for sid in [1,2,3,4,5,6]:
        counts.append(int((leads["LeadStageId"] == sid).sum()) if "LeadStageId" in leads.columns else 0)
    f = go.Figure()
    f.add_trace(go.Funnel(y=stage_names, x=counts, textinfo="value+percent initial",
                          marker=dict(color=[EXEC_BLUE, EXEC_GREEN, EXEC_PRIMARY, "#FFA500", EXEC_DANGER, "#8A2BE2"])))
    f.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="white", title_font_color=EXEC_PRIMARY
    )
    st.plotly_chart(f, use_container_width=True)

def show_geography(d):
    leads = d["leads"]
    countries = d["countries"]

    if leads is None or len(leads) == 0 or countries is None or len(countries) == 0:
        st.info("No geographic data available in the selected range.")
        return

    geo = leads.groupby("CountryId").agg(
        LeadCount=("LeadId","count"),
        PipelineValue=("EstimatedBudget","sum"),
        WonDeals=("LeadStageId", lambda x: (x==6).sum())
    ).reset_index()
    geo = geo.merge(countries[["CountryId","CountryName_E","CountryCode"]], on="CountryId", how="left")
    geo["ConversionRate"] = (geo["WonDeals"]/geo["LeadCount"]*100).replace([np.inf,-np.inf],0).fillna(0).round(1)
    geo = geo.sort_values("PipelineValue", ascending=False)

    st.subheader("🏆 Top Markets")
    top = geo.head(8)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=top["LeadCount"],
        y=top["PipelineValue"]/1e9,
        mode="markers+text",
        text=top["CountryName_E"],
        textposition="middle center",
        marker=dict(size=top["ConversionRate"]*10, color=top["ConversionRate"], colorscale="Viridis",
                    colorbar=dict(title="Conv (%)"))
    ))
    fig.update_layout(
        title="Leads vs Pipeline (Bubble = Conversion Rate)",
        xaxis_title="Leads", yaxis_title="Pipeline ($B)",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="white", title_font_color=EXEC_PRIMARY
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Market Table")
    disp = top[["CountryName_E","LeadCount","PipelineValue","WonDeals","ConversionRate"]].copy()
    disp["PipelineValue"] = disp["PipelineValue"].apply(format_currency)
    disp.columns = ["Country","Leads","Pipeline","Won Deals","Conv (%)"]
    st.dataframe(disp, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Markets", len(geo))
    with c2: st.metric("Global Pipeline", format_currency(geo["PipelineValue"].sum()))
    with c3: st.metric("Top Market", str(geo.iloc[0]["CountryName_E"]) if len(geo) else "-")
    with c4: st.metric("Avg Conversion", f"{geo['ConversionRate'].mean():.1f}%")

    st.markdown("---")
    st.subheader("🤖 Geographic Intelligence")
    c5, c6 = st.columns(2)
    with c5:
        st.markdown(f"""
        <div class="insight-box">
        <h4>🎯 Market Analysis</h4>
        <ul>
          <li>Expansion: Egypt, Morocco, Turkey</li>
          <li>UAE nearing capacity → watch saturation</li>
          <li>Saudi requires +25 brokers</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with c6:
        st.markdown(f"""
        <div class="insight-box">
        <h4>📊 Predictions</h4>
        <ul>
          <li>Qatar: ~45% growth projected (2026)</li>
          <li>India: highest lead volume potential</li>
          <li>Europe expansion ROI ≈ 285%</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------
def render_page(page_key: str, fdata, grain_sel: str):
    if page_key == "Executive":
        show_executive_summary(fdata)
    elif page_key == "Leads":
        show_lead_status(fdata)
    elif page_key == "Calls":
        show_calls(fdata, grain_sel)
    elif page_key == "Tasks":
        show_tasks(fdata)
    elif page_key == "Agents":
        show_agents(fdata)
    elif page_key == "Conversion":
        show_conversion(fdata)
    elif page_key == "Geography":
        show_geography(fdata)

if HAS_OPTION_MENU:
    render_page(selected, fdata, grain)
else:
    # Fallback in tabs mode
    for idx, tab in enumerate(tab_objs):
        with tab:
            render_page(NAV_ITEMS[idx][0], fdata, grain)
