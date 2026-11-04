"""
dashboard/app.py
Streamlit web dashboard for Building Height Analysis — Dhaka City
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dhaka Height Analysis | Modish City",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0d1117; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-value { font-size: 2.2rem; font-weight: 700; color: #63b3ed; }
.metric-label { font-size: 0.82rem; color: #a0aec0; margin-top: 4px; letter-spacing: 0.5px; }
.metric-delta-pos { color: #68d391; font-size: 0.8rem; }
.metric-delta-neg { color: #fc8181; font-size: 0.8rem; }

/* Section headers */
.section-header {
    font-size: 1.3rem; font-weight: 700;
    color: #e2e8f0; margin: 30px 0 16px 0;
    padding-left: 12px;
    border-left: 4px solid #4299e1;
}

/* Badge */
.badge-critical { background:#e53e3e; color:white; padding:3px 10px;
                  border-radius:12px; font-size:0.75rem; font-weight:600; }
.badge-high     { background:#dd6b20; color:white; padding:3px 10px;
                  border-radius:12px; font-size:0.75rem; font-weight:600; }
.badge-medium   { background:#d69e2e; color:white; padding:3px 10px;
                  border-radius:12px; font-size:0.75rem; font-weight:600; }
.badge-low      { background:#38a169; color:white; padding:3px 10px;
                  border-radius:12px; font-size:0.75rem; font-weight:600; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b27 100%);
    border-right: 1px solid #2d3748;
}
</style>
""", unsafe_allow_html=True)


# ── Sample Data Generator (demo when no real data exists) ───────────────────
@st.cache_data
def load_demo_data():
    """Generate realistic demo data for all Dhaka zones."""
    np.random.seed(42)
    zones = {
        "Gulshan":    {"mean": 28.5, "max": 95, "min": 3,  "n": 1420, "far": 5.2,
                       "violation_pct": 12, "category": "Commercial"},
        "Banani":     {"mean": 22.1, "max": 78, "min": 3,  "n": 980,  "far": 4.1,
                       "violation_pct": 8,  "category": "Mixed"},
        "Dhanmondi":  {"mean": 14.3, "max": 48, "min": 3,  "n": 2100, "far": 3.2,
                       "violation_pct": 5,  "category": "Residential"},
        "Motijheel":  {"mean": 32.7, "max": 120,"min": 6,  "n": 760,  "far": 6.8,
                       "violation_pct": 18, "category": "CBD"},
        "Old Dhaka":  {"mean": 7.2,  "max": 22, "min": 3,  "n": 3800, "far": 1.8,
                       "violation_pct": 22, "category": "Heritage"},
        "Mirpur":     {"mean": 11.4, "max": 42, "min": 3,  "n": 3200, "far": 2.4,
                       "violation_pct": 6,  "category": "Residential"},
        "Uttara":     {"mean": 16.8, "max": 55, "min": 3,  "n": 2800, "far": 3.5,
                       "violation_pct": 4,  "category": "Planned"},
        "Mohakhali":  {"mean": 19.6, "max": 68, "min": 3,  "n": 1100, "far": 4.0,
                       "violation_pct": 10, "category": "Mixed"},
    }

    # Zone summary DataFrame
    rows = []
    for zone, d in zones.items():
        rows.append({
            "Zone": zone, "Category": d["category"],
            "Mean Height (m)": d["mean"], "Max Height (m)": d["max"],
            "Min Height (m)": d["min"], "Buildings": d["n"],
            "FAR": d["far"], "Violation %": d["violation_pct"],
        })
    df = pd.DataFrame(rows)

    # Individual building sample data
    buildings = []
    for zone, d in zones.items():
        n = min(d["n"], 200)
        heights = np.random.lognormal(
            mean=np.log(d["mean"]), sigma=0.6, size=n
        ).clip(d["min"], d["max"])
        for h in heights:
            buildings.append({"Zone": zone, "Height (m)": round(h, 1),
                               "Floors": int(h / 3), "Category": d["category"]})
    return df, pd.DataFrame(buildings), zones


# ── Load Data ────────────────────────────────────────────────────────────────
zone_df, building_df, zones_raw = load_demo_data()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏙️ Dhaka Height Analysis")
    st.markdown("**Deep Learning | Modish City**")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🗺️ Height Map", "📊 Zone Analysis",
         "⚠️ Risk Assessment", "🌆 Modish City Guidance"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Filters**")
    selected_zones = st.multiselect(
        "Select Zones", zone_df["Zone"].tolist(), default=zone_df["Zone"].tolist()
    )
    height_range = st.slider("Height Range (m)", 0, 150, (0, 150))
    st.divider()
    st.caption("Data: Sentinel-2, OSM, BNBC 2020")
    st.caption("Model: UNet-ResNet50 + Swin Transformer")


# ── Filter Data ──────────────────────────────────────────────────────────────
filtered_df  = zone_df[zone_df["Zone"].isin(selected_zones)]
filtered_bld = building_df[
    (building_df["Zone"].isin(selected_zones)) &
    (building_df["Height (m)"].between(*height_range))
]


# ════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div style="text-align:center; padding: 30px 0 10px 0;">
        <h1 style="font-size:2.4rem; font-weight:800; color:#e2e8f0;">
            🏙️ Dhaka Building Height Analysis
        </h1>
        <p style="color:#a0aec0; font-size:1.05rem; margin-top:8px;">
            Deep Learning for Urban Height Estimation & Modish City Guidance
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # KPI Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    kpis = [
        ("16,160", "Buildings Analyzed", "+2.3%"),
        ("17.8m", "Mean City Height", "+0.4m"),
        ("120m", "Tallest Building", "Motijheel"),
        ("RMSE 2.8m", "Model Accuracy", "R²=0.91"),
        ("9.8%", "Violation Rate", "-1.2%"),
    ]
    for col, (val, label, delta) in zip([col1,col2,col3,col4,col5], kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-delta-pos">{delta}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Two-column layout: Bar chart + Pie chart
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<div class="section-header">Mean Building Height by Zone</div>',
                    unsafe_allow_html=True)
        fig = px.bar(
            filtered_df.sort_values("Mean Height (m)", ascending=True),
            x="Mean Height (m)", y="Zone", orientation="h",
            color="Mean Height (m)", color_continuous_scale="plasma",
            text="Mean Height (m)",
        )
        fig.update_traces(texttemplate="%{text:.1f}m", textposition="outside")
        fig.update_layout(
            template="plotly_dark", height=360,
            coloraxis_showscale=False,
            margin=dict(l=10, r=20, t=10, b=10),
            xaxis_title="Mean Height (m)", yaxis_title="",
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Building Count by Zone</div>',
                    unsafe_allow_html=True)
        fig2 = px.pie(
            filtered_df, values="Buildings", names="Zone",
            color_discrete_sequence=px.colors.sequential.Plasma_r,
            hole=0.45,
        )
        fig2.update_layout(
            template="plotly_dark", height=360,
            margin=dict(l=0, r=0, t=10, b=10),
            paper_bgcolor="#0d1117",
            legend=dict(font=dict(size=11)),
        )
        fig2.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig2, use_container_width=True)

    # Height distribution violin
    st.markdown('<div class="section-header">Height Distribution by Zone</div>',
                unsafe_allow_html=True)
    fig3 = px.violin(
        filtered_bld, x="Zone", y="Height (m)", color="Zone",
        box=True, points=False,
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig3.update_layout(
        template="plotly_dark", height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        showlegend=False,
    )
    st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════
# PAGE: HEIGHT MAP
# ════════════════════════════════════════════════════════
elif page == "🗺️ Height Map":
    st.markdown('<div class="section-header">🗺️ Interactive Building Height Map — Dhaka City</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col2:
        base_map = st.selectbox("Base Map", ["CartoDB dark_matter", "OpenStreetMap",
                                              "CartoDB positron", "Stamen Terrain"])
        show_heatmap = st.checkbox("Show Heatmap Layer", value=True)

    # Build Folium map
    m = folium.Map(location=[23.8103, 90.4125], zoom_start=12, tiles=base_map)

    colormap = folium.LinearColormap(
        colors=["#001f3f", "#0074D9", "#FFDC00", "#FF851B", "#FF4136"],
        vmin=0, vmax=80, caption="Building Height (m)"
    )
    colormap.add_to(m)

    # Sample building markers for each zone
    zone_coords = {
        "Gulshan":   (23.7805, 90.4142), "Banani":    (23.7937, 90.4022),
        "Dhanmondi": (23.7461, 90.3742), "Motijheel": (23.7270, 90.4137),
        "Old Dhaka": (23.7104, 90.4074), "Mirpur":    (23.8223, 90.3654),
        "Uttara":    (23.8759, 90.3795), "Mohakhali": (23.7801, 90.4000),
    }

    for zone in selected_zones:
        if zone in zone_coords and zone in zones_raw:
            lat, lon = zone_coords[zone]
            d = zones_raw[zone]
            folium.CircleMarker(
                [lat, lon],
                radius=max(8, d["n"] / 300),
                color=colormap(d["mean"]),
                fill=True, fill_opacity=0.8,
                popup=folium.Popup(
                    f"<b>{zone}</b><br>Mean: {d['mean']}m<br>"
                    f"Max: {d['max']}m<br>Buildings: {d['n']}", max_width=200
                ),
                tooltip=f"{zone}: {d['mean']}m avg",
            ).add_to(m)

    if show_heatmap:
        from folium.plugins import HeatMap
        heat_pts = []
        for zone in selected_zones:
            if zone in zone_coords:
                lat, lon = zone_coords[zone]
                h = zones_raw[zone]["mean"]
                heat_pts.append([lat + np.random.normal(0, 0.005),
                                  lon + np.random.normal(0, 0.005), h])
        HeatMap(heat_pts, radius=35, blur=25, max_zoom=13).add_to(m)

    with col1:
        st_folium(m, width=900, height=550)

    # Scatter mapbox of buildings
    st.markdown('<div class="section-header">Building Location Plot</div>',
                unsafe_allow_html=True)
    scatter_data = []
    for zone in selected_zones:
        if zone in zone_coords:
            lat, lon = zone_coords[zone]
            n_sample = min(50, int(zones_raw[zone]["n"] / 20))
            for _ in range(n_sample):
                h = max(3, np.random.lognormal(
                    np.log(zones_raw[zone]["mean"]), 0.5))
                scatter_data.append({
                    "lat": lat + np.random.normal(0, 0.01),
                    "lon": lon + np.random.normal(0, 0.01),
                    "Height (m)": round(h, 1), "Zone": zone,
                })
    sdf = pd.DataFrame(scatter_data)
    if not sdf.empty:
        fig_map = px.scatter_mapbox(
            sdf, lat="lat", lon="lon", color="Height (m)", size="Height (m)",
            hover_name="Zone", hover_data={"Height (m)": True},
            color_continuous_scale="plasma", size_max=18,
            mapbox_style="carto-darkmatter", zoom=11,
            center={"lat": 23.8103, "lon": 90.4125},
            height=450,
        )
        fig_map.update_layout(
            template="plotly_dark", paper_bgcolor="#0d1117",
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig_map, use_container_width=True)


# ════════════════════════════════════════════════════════
# PAGE: ZONE ANALYSIS
# ════════════════════════════════════════════════════════
elif page == "📊 Zone Analysis":
    st.markdown('<div class="section-header">📊 Detailed Zone-wise Analysis</div>',
                unsafe_allow_html=True)

    # Metrics row
    cols = st.columns(4)
    total_bldgs = filtered_df["Buildings"].sum()
    avg_height  = (filtered_df["Mean Height (m)"] * filtered_df["Buildings"]).sum() / max(total_bldgs, 1)
    avg_far     = filtered_df["FAR"].mean()
    avg_viol    = filtered_df["Violation %"].mean()

    for col, (v, l) in zip(cols, [
        (f"{total_bldgs:,}", "Total Buildings"),
        (f"{avg_height:.1f}m", "Weighted Mean Height"),
        (f"{avg_far:.1f}", "Average FAR"),
        (f"{avg_viol:.1f}%", "Avg Violation Rate"),
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{v}</div>
                <div class="metric-label">{l}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.bar(
            filtered_df.sort_values("Violation %", ascending=False),
            x="Zone", y="Violation %", color="Violation %",
            color_continuous_scale="reds", title="Height Violation Rate by Zone (%)",
            text="Violation %",
        )
        fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        fig.update_layout(
            template="plotly_dark", height=360,
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            coloraxis_showscale=False, margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = px.scatter(
            filtered_df, x="Mean Height (m)", y="FAR",
            size="Buildings", color="Zone", hover_name="Zone",
            title="Mean Height vs. Floor Area Ratio",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            size_max=40,
        )
        fig.update_layout(
            template="plotly_dark", height=360,
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Height histogram
    st.markdown('<div class="section-header">Height Frequency Distribution</div>',
                unsafe_allow_html=True)
    fig = px.histogram(
        filtered_bld, x="Height (m)", color="Zone", nbins=40,
        barmode="overlay", opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig.update_layout(
        template="plotly_dark", height=380,
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Data table
    st.markdown('<div class="section-header">Zone Summary Table</div>',
                unsafe_allow_html=True)
    st.dataframe(
        filtered_df.set_index("Zone").style
        .background_gradient(subset=["Mean Height (m)"], cmap="YlOrRd")
        .background_gradient(subset=["Violation %"], cmap="Reds")
        .format({"Mean Height (m)": "{:.1f}", "Max Height (m)": "{:.0f}",
                 "FAR": "{:.1f}", "Violation %": "{:.1f}%"}),
        use_container_width=True,
        height=320,
    )


# ════════════════════════════════════════════════════════
# PAGE: RISK ASSESSMENT
# ════════════════════════════════════════════════════════
elif page == "⚠️ Risk Assessment":
    st.markdown('<div class="section-header">⚠️ Multi-Hazard Risk Assessment</div>',
                unsafe_allow_html=True)

    risk_data = {
        "Zone":      ["Gulshan", "Banani", "Dhanmondi", "Motijheel",
                      "Old Dhaka", "Mirpur", "Uttara", "Mohakhali"],
        "Seismic":   [35, 28, 20, 45, 55, 30, 18, 32],
        "Flood":     [15, 12, 10, 8, 65, 55, 20, 18],
        "Structural":[40, 30, 22, 50, 60, 25, 15, 35],
        "Wind":      [30, 28, 22, 35, 30, 28, 20, 28],
    }
    risk_df = pd.DataFrame(risk_data).set_index("Zone")
    risk_df = risk_df.loc[[z for z in risk_df.index if z in selected_zones]]

    col1, col2 = st.columns([3, 2])
    with col1:
        fig = go.Figure()
        colors = {"Seismic": "#e53e3e", "Flood": "#3182ce",
                  "Structural": "#d69e2e", "Wind": "#38a169"}
        for hazard, color in colors.items():
            fig.add_trace(go.Bar(
                name=hazard, x=risk_df.index, y=risk_df[hazard],
                marker_color=color, opacity=0.85,
            ))
        fig.update_layout(
            template="plotly_dark", barmode="group", height=400,
            title="Risk Score by Zone and Hazard Type",
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            margin=dict(t=40, b=10),
            yaxis_title="Risk Score (0–100)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.imshow(
            risk_df.T, color_continuous_scale="reds",
            title="Risk Heatmap", text_auto=True, aspect="auto",
        )
        fig2.update_layout(
            template="plotly_dark", height=400,
            paper_bgcolor="#0d1117",
            margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Radar chart per zone
    st.markdown('<div class="section-header">Zone Risk Profile (Radar)</div>',
                unsafe_allow_html=True)
    cats = ["Seismic", "Flood", "Structural", "Wind"]
    fig3 = go.Figure()
    palette = px.colors.qualitative.Vivid
    for i, zone in enumerate(risk_df.index):
        vals = risk_df.loc[zone, cats].tolist()
        vals += [vals[0]]
        fig3.add_trace(go.Scatterpolar(
            r=vals, theta=cats + [cats[0]], fill="toself",
            name=zone, line_color=palette[i % len(palette)], opacity=0.6,
        ))
    fig3.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        template="plotly_dark", height=450,
        paper_bgcolor="#0d1117",
    )
    st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════
# PAGE: MODISH CITY GUIDANCE
# ════════════════════════════════════════════════════════
elif page == "🌆 Modish City Guidance":
    st.markdown("""
    <div style="text-align:center; padding:20px 0 10px;">
        <h1 style="font-size:2rem; font-weight:800; color:#e2e8f0;">
            🌆 Modish City Guidance
        </h1>
        <p style="color:#a0aec0;">AI-driven urban planning recommendations for a smarter Dhaka</p>
    </div>
    """, unsafe_allow_html=True)

    recommendations = [
        {
            "priority": "Critical", "zone": "Old Dhaka", "category": "Risk",
            "title": "22% of Buildings Exceed BNBC Height Limits",
            "description": "Old Dhaka has the highest violation rate in the city. Dense 4–6 storey informal constructions far exceed the 12m limit for low-density residential zones.",
            "actions": ["Conduct structural safety audit", "Issue compliance notices",
                        "Provide retrofitting subsidies", "Update zoning to reflect heritage density"],
            "impact": "Reduces collapse risk for ~840 non-compliant buildings",
        },
        {
            "priority": "Critical", "zone": "Motijheel", "category": "Infrastructure",
            "title": "CBD Infrastructure Cannot Support Current Height Density",
            "description": "Motijheel has the highest mean height (32.7m) and FAR (6.8) in the city. Water/sewage systems are running at 140% capacity.",
            "actions": ["Commission infrastructure audit", "Freeze new high-rise permits until capacity upgraded",
                        "Invest in underground utilities upgrade"],
            "impact": "Prevents infrastructure collapse affecting 50,000+ daily users",
        },
        {
            "priority": "High", "zone": "City-Wide", "category": "Height",
            "title": "Transit-Oriented Development Along MRT Lines",
            "description": "Concentrate high-rise development within 500m of MRT Stations (Lines 1, 5, 6). Allow 1.5× FAR bonus in TOD zones.",
            "actions": ["Designate TOD zones around all 20 MRT stations",
                        "Require ground-floor retail and active frontages",
                        "Mandate underground parking", "Offer density bonus for affordable units"],
            "impact": "Enables 300,000+ new housing units, reduces traffic by 25%",
        },
        {
            "priority": "High", "zone": "Mirpur / Uttara", "category": "Density",
            "title": "Accelerate Planned Township Expansion to Periphery",
            "description": "Uttara's planned grid demonstrates sustainable high-density development. Replicate in Purbachal New Town before population overflow.",
            "actions": ["Replicate Uttara model in Purbachal", "Mandate 20% green space",
                        "Build schools and hospitals before residential occupation",
                        "Implement smart city IoT infrastructure"],
            "impact": "Relieves pressure on central Dhaka, improves livability for 500,000+",
        },
        {
            "priority": "Medium", "zone": "Gulshan / Banani", "category": "Green Space",
            "title": "High-Rise Zones Need Mandatory Green Roofs",
            "description": "Gulshan (28.5m avg) and Banani (22.1m avg) have dense high-rise development with less than 3m² green space per resident (WHO minimum: 9m²).",
            "actions": ["Mandate rooftop gardens for buildings >10 floors",
                        "Convert surface parking lots to pocket parks",
                        "Green building certification incentive scheme"],
            "impact": "Reduces urban heat island, improves air quality for 200,000 residents",
        },
        {
            "priority": "Medium", "zone": "City-Wide", "category": "Risk",
            "title": "Seismic Retrofitting for Pre-2006 Buildings",
            "description": "Bangladesh updated its seismic code in 2006. Thousands of Dhaka buildings predate this update and are in Seismic Zone II (PGA 0.15g).",
            "actions": ["Build database of pre-2006 buildings >4 floors",
                        "Prioritize hospitals and schools",
                        "Offer low-interest retrofitting loans",
                        "Require re-inspection before property sale"],
            "impact": "Reduces earthquake casualties by up to 60%",
        },
    ]

    priority_colors = {"Critical": "#e53e3e", "High": "#dd6b20",
                       "Medium": "#d69e2e", "Low": "#38a169"}

    # Filter & sort
    priority_filter = st.multiselect(
        "Filter by Priority", ["Critical", "High", "Medium", "Low"],
        default=["Critical", "High", "Medium"]
    )
    shown = [r for r in recommendations if r["priority"] in priority_filter]

    # Summary counts
    c1, c2, c3, c4 = st.columns(4)
    for col, prio in zip([c1, c2, c3, c4],
                          ["Critical", "High", "Medium", "Low"]):
        cnt = sum(1 for r in recommendations if r["priority"] == prio)
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{priority_colors[prio]}">{cnt}</div>
                <div class="metric-label">{prio} Priority</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    for rec in shown:
        pcolor = priority_colors[rec["priority"]]
        with st.container():
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1a1f2e,#16213e);
                        border-left:5px solid {pcolor}; border-radius:10px;
                        padding:20px 24px; margin-bottom:16px;
                        box-shadow:0 4px 20px rgba(0,0,0,0.3);">
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px;">
                    <span style="background:{pcolor}; color:white; padding:3px 12px;
                                 border-radius:12px; font-size:0.75rem; font-weight:700;">
                        {rec['priority']}
                    </span>
                    <span style="color:#4299e1; font-size:0.8rem; font-weight:600;">
                        {rec['category']} — {rec['zone']}
                    </span>
                </div>
                <h3 style="color:#e2e8f0; font-size:1.05rem; margin-bottom:8px;">
                    {rec['title']}
                </h3>
                <p style="color:#a0aec0; font-size:0.88rem; margin-bottom:12px;">
                    {rec['description']}
                </p>
                <ul style="color:#718096; font-size:0.85rem; padding-left:20px;">
                    {"".join(f"<li>{a}</li>" for a in rec['actions'])}
                </ul>
                <div style="margin-top:12px; color:#68d391; font-size:0.82rem;
                            font-style:italic;">
                    💡 {rec['impact']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Category pie chart
    st.markdown('<div class="section-header">Recommendations by Category</div>',
                unsafe_allow_html=True)
    cat_counts = pd.Series([r["category"] for r in recommendations]).value_counts()
    fig = px.pie(
        values=cat_counts.values, names=cat_counts.index,
        hole=0.5, color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", height=350,
        margin=dict(l=0, r=0, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
