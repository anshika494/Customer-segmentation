# =============================================================================
# Streamlit Interactive Dashboard – Customer Segmentation
# =============================================================================
# Run with:  streamlit run streamlit_dashboard.py
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
    }

    .main-header h1 {
        color: #e94560;
        font-size: 2.6rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .main-header p {
        color: #a8b2d8;
        font-size: 1.1rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e2a45, #162032);
        border: 1px solid #2d4070;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }

    .metric-card h2 {
        color: #e94560;
        font-size: 2rem;
        margin: 0;
    }

    .metric-card p {
        color: #a8b2d8;
        font-size: 0.85rem;
        margin: 0;
    }

    .segment-card {
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border-left: 5px solid;
    }

    .stSelectbox label, .stSlider label {
        color: #a8b2d8 !important;
        font-weight: 600;
    }

    .sidebar .sidebar-content {
        background: #0f3460;
    }
</style>
""", unsafe_allow_html=True)

# ── Palette (10 colours so any K up to 10 is safe) ───────────────────────────
PALETTE = [
    "#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261",
    "#8338ec", "#06d6a0", "#fb5607", "#3a86ff", "#ff006e",
]
# Named labels for the canonical 5-cluster solution.
# Clusters beyond index 4 get a generic fallback label.
CLUSTER_NAMES = [
    "High Income – High Spending 💎",
    "Low Income – High Spending 🛒",
    "Average Customers 🏪",
    "High Income – Low Spending 🎯",
    "Low Income – Low Spending 💤",
]


def get_cluster_name(cluster_id: int) -> str:
    """Return a descriptive name for cluster_id, with a safe generic fallback."""
    if cluster_id < len(CLUSTER_NAMES):
        return CLUSTER_NAMES[cluster_id]
    return f"Segment {cluster_id} 🔵"
CLUSTER_RECS = {
    0: [
        "🎖️ Launch a Premium Loyalty Programme with exclusive rewards.",
        "🛍️ Offer VIP early access and personalised recommendations.",
        "🤝 Assign dedicated personal shoppers or relationship managers.",
        "📲 Cross-sell luxury brands and premium categories.",
        "💌 Exclusive private-sale event invitations.",
    ],
    1: [
        "🏷️ Promote budget-friendly value bundles.",
        "🎟️ Introduce buy-now-pay-later (BNPL) or instalment plans.",
        "🎁 Gamified loyalty points for repeat visits.",
        "📣 Flash sales and limited-time SMS/app discount alerts.",
        "🤑 Referral bonuses to grow this segment organically.",
    ],
    2: [
        "📦 Cross-sell complementary products at checkout.",
        "🔔 Seasonal promotions aligned with paydays.",
        "📧 Personalised 'picks for you' email newsletters.",
        "⭐ Tiered membership to nudge toward higher spending.",
        "📊 A/B test discount vs. value-add offers.",
    ],
    3: [
        "🔍 Survey to identify barriers preventing spending.",
        "🌟 Upsell premium quality with clear value propositions.",
        "🎪 Experiential marketing: events, tastings, previews.",
        "💳 Co-branded credit card with cashback incentives.",
        "📱 Quality-focused retargeting ad campaigns.",
    ],
    4: [
        "💲 Deep-discount campaigns and bundle offers.",
        "🛡️ 'We miss you' retention vouchers after 30-day inactivity.",
        "📍 Geo-targeted push notifications near the mall.",
        "🤲 Community-building via social media giveaways.",
        "📉 Focus on satisfaction metrics over acquisition cost.",
    ],
}


# ── Data Loading & Caching ────────────────────────────────────────────────────
@st.cache_data
def load_and_cluster(n_clusters: int):
    """Load data, scale, cluster, and return everything needed for the dashboard."""
    df = pd.read_csv("data/Mall_Customers.csv")
    df.rename(columns={"Genre": "Gender"}, inplace=True)

    X_raw = df[["Annual Income (k$)", "Spending Score (1-100)"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    km = KMeans(n_clusters=n_clusters, init="k-means++",
                n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)

    df["Cluster"] = labels
    # Use get_cluster_name() so K > 5 never raises an IndexError
    df["Cluster Name"] = [get_cluster_name(i) for i in labels]

    centroids_raw = scaler.inverse_transform(km.cluster_centers_)
    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)

    return df, X_raw, X_scaled, km, labels, centroids_raw, sil, db, scaler


@st.cache_data
def compute_elbow(max_k: int = 11):
    df = pd.read_csv("data/Mall_Customers.csv")
    df.rename(columns={"Genre": "Gender"}, inplace=True)
    X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ks, inertias, sils, dbs = [], [], [], []
    for k in range(2, max_k):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        lbl = km.fit_predict(X_scaled)
        ks.append(k)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X_scaled, lbl))
        dbs.append(davies_bouldin_score(X_scaled, lbl))
    return ks, inertias, sils, dbs


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    n_clusters = st.slider("Number of Clusters (K)", min_value=2,
                           max_value=10, value=5, step=1)
    st.markdown("---")
    st.markdown("### 📂 Dataset Info")
    st.markdown("**Mall Customers Dataset**")
    st.markdown("200 customers | 5 features")
    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("- [GitHub Repository](https://github.com/anshika494/Customer-segmentation)")
    st.markdown("- [Dataset (Kaggle)](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)")
    st.markdown("---")
    st.markdown("Built using **Streamlit** + **Scikit-learn**")

# ── Load Data ─────────────────────────────────────────────────────────────────
df, X_raw, X_scaled, km, labels, centroids_raw, sil, db, scaler = load_and_cluster(n_clusters)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛍️ Customer Segmentation Dashboard</h1>
    <p>K-Means Clustering on Mall Customers Dataset · Annual Income & Spending Score</p>
</div>
""", unsafe_allow_html=True)

# ── Metric Cards ──────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("👥 Total Customers", len(df))
with col2:
    st.metric("🔢 Clusters (K)", n_clusters)
with col3:
    st.metric("📈 Silhouette Score", f"{sil:.3f}")
with col4:
    st.metric("📉 Davies-Bouldin", f"{db:.3f}")

st.markdown("---")

# ── Tab Layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 EDA", "🔍 Optimal K", "🗺️ Clusters", "📋 Segment Analysis", "💡 Recommendations"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Exploratory Data Analysis")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Dataset Preview**")
        st.dataframe(df.drop(columns=["Cluster", "Cluster Name"]).head(10),
                     use_container_width=True)

    with col_b:
        st.markdown("**Descriptive Statistics**")
        st.dataframe(df[["Age", "Annual Income (k$)",
                          "Spending Score (1-100)"]].describe().round(2),
                     use_container_width=True)

    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        fig = px.histogram(df, x="Age", nbins=20,
                           title="Age Distribution",
                           color_discrete_sequence=["#457b9d"])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(df, x="Annual Income (k$)", nbins=20,
                           title="Annual Income Distribution",
                           color_discrete_sequence=["#2a9d8f"])
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        fig = px.histogram(df, x="Spending Score (1-100)", nbins=20,
                           title="Spending Score Distribution",
                           color_discrete_sequence=["#e63946"])
        st.plotly_chart(fig, use_container_width=True)

    c4, c5 = st.columns(2)
    with c4:
        fig = px.pie(df, names="Gender",
                     title="Gender Distribution",
                     color_discrete_sequence=["#e63946", "#457b9d"],
                     hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with c5:
        fig = px.scatter(df,
                         x="Annual Income (k$)", y="Spending Score (1-100)",
                         color="Gender",
                         title="Annual Income vs Spending Score (by Gender)",
                         color_discrete_map={"Male": "#457b9d", "Female": "#e63946"},
                         opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.markdown("**Correlation Heatmap**")
    corr = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, title="Feature Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – OPTIMAL K
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Finding the Optimal Number of Clusters")
    st.info("Use the Elbow Method, Silhouette Score, and Davies-Bouldin Index "
            "to determine the best K. Look for the 'elbow' in inertia, the "
            "peak in silhouette score, and the minimum in Davies-Bouldin.")

    ks, inertias, sils, dbs = compute_elbow()

    c1, c2, c3 = st.columns(3)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ks, y=inertias, mode="lines+markers",
                                 marker=dict(color="#e63946", size=8),
                                 line=dict(width=2)))
        fig.add_vline(x=5, line_dash="dash", line_color="#2a9d8f",
                      annotation_text="K=5")
        fig.update_layout(title="Elbow Method – Inertia",
                          xaxis_title="K", yaxis_title="Inertia (WCSS)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ks, y=sils, mode="lines+markers",
                                 marker=dict(color="#457b9d", size=8),
                                 line=dict(width=2)))
        best_k = ks[sils.index(max(sils))]
        fig.add_vline(x=best_k, line_dash="dash", line_color="#2a9d8f",
                      annotation_text=f"K={best_k}")
        fig.update_layout(title="Silhouette Score (higher = better)",
                          xaxis_title="K", yaxis_title="Silhouette Score")
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ks, y=dbs, mode="lines+markers",
                                 marker=dict(color="#e9c46a", size=8),
                                 line=dict(width=2)))
        best_db_k = ks[dbs.index(min(dbs))]
        fig.add_vline(x=best_db_k, line_dash="dash", line_color="#2a9d8f",
                      annotation_text=f"K={best_db_k}")
        fig.update_layout(title="Davies-Bouldin (lower = better)",
                          xaxis_title="K", yaxis_title="DB Index")
        st.plotly_chart(fig, use_container_width=True)

    st.success(f"✅ **Recommended K = 5** — supported by the elbow at K=5, "
               f"peak silhouette at K={best_k}, and DB minimum at K={best_db_k}.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – CLUSTERS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Cluster Visualisation")

    c1, c2 = st.columns(2)
    with c1:
        # Before clustering
        fig = px.scatter(df, x="Annual Income (k$)", y="Spending Score (1-100)",
                         title="Before Clustering",
                         color_discrete_sequence=["#a8b2d8"],
                         opacity=0.7)
        fig.update_traces(marker=dict(size=8))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # After clustering
        fig = px.scatter(df,
                         x="Annual Income (k$)", y="Spending Score (1-100)",
                         color="Cluster Name",
                         color_discrete_sequence=PALETTE[:n_clusters],
                         title=f"After K-Means Clustering (K={n_clusters})",
                         opacity=0.8)
        # Add centroids
        centroid_df = pd.DataFrame(centroids_raw,
                                   columns=["Annual Income (k$)",
                                            "Spending Score (1-100)"])
        fig.add_trace(go.Scatter(
            x=centroid_df["Annual Income (k$)"],
            y=centroid_df["Spending Score (1-100)"],
            mode="markers",
            marker=dict(symbol="star", size=18, color="white",
                        line=dict(color="black", width=1.5)),
            name="Centroids",
        ))
        fig.update_traces(marker=dict(size=9), selector=dict(mode="markers"))
        st.plotly_chart(fig, use_container_width=True)

    # 3D scatter (Age, Income, Spending)
    st.markdown("**3D View — Age, Annual Income, Spending Score**")
    fig3d = px.scatter_3d(df,
                          x="Age", y="Annual Income (k$)",
                          z="Spending Score (1-100)",
                          color="Cluster Name",
                          color_discrete_sequence=PALETTE[:n_clusters],
                          opacity=0.75,
                          title="3D Cluster View")
    st.plotly_chart(fig3d, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – SEGMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Segment-Level Analysis")

    summary = (df.groupby("Cluster Name")
               .agg(
                   Count=("CustomerID", "count"),
                   Avg_Income=("Annual Income (k$)", "mean"),
                   Avg_Spending=("Spending Score (1-100)", "mean"),
                   Avg_Age=("Age", "mean"),
               )
               .round(1)
               .reset_index())
    summary["Share (%)"] = (summary["Count"] / len(df) * 100).round(1)

    st.dataframe(summary, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(summary, x="Cluster Name", y="Count",
                     color="Cluster Name",
                     color_discrete_sequence=PALETTE[:n_clusters],
                     title="Customers per Cluster")
        fig.update_xaxes(tickangle=25)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Avg Income (k$)", x=summary["Cluster Name"],
                             y=summary["Avg_Income"],
                             marker_color=PALETTE[1]))
        fig.add_trace(go.Bar(name="Avg Spending Score", x=summary["Cluster Name"],
                             y=summary["Avg_Spending"],
                             marker_color=PALETTE[0]))
        fig.update_layout(barmode="group",
                          title="Avg Income vs Avg Spending per Cluster")
        fig.update_xaxes(tickangle=25)
        st.plotly_chart(fig, use_container_width=True)

    # Radar chart for cluster profiles
    st.markdown("**Cluster Profile Radar Chart**")
    categories = ["Avg Income", "Avg Spending", "Avg Age", "Count Share (%)"]

    max_income = summary["Avg_Income"].max()
    max_spending = summary["Avg_Spending"].max()
    max_age = summary["Avg_Age"].max()
    max_share = summary["Share (%)"].max()

    fig = go.Figure()
    for i, row in summary.iterrows():
        vals = [
            row["Avg_Income"] / max_income * 100,
            row["Avg_Spending"] / max_spending * 100,
            row["Avg_Age"] / max_age * 100,
            row["Share (%)"] / max_share * 100,
        ]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            name=row["Cluster Name"],
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            fill="toself",
            opacity=0.4,
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                      title="Normalised Cluster Profile Radar")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("💡 Actionable Business Recommendations")
    st.markdown("Select a customer segment below to see targeted marketing strategies:")

    # Build the name list using the safe helper so K > 5 works correctly
    segment_options = [get_cluster_name(i) for i in range(n_clusters)]
    selected = st.selectbox("Choose a Segment", segment_options)
    cluster_id = segment_options.index(selected)

    # PALETTE has 10 entries; modulo makes it safe for any K
    color = PALETTE[cluster_id % len(PALETTE)]

    subset = df[df["Cluster"] == cluster_id]
    avg_income = subset["Annual Income (k$)"].mean()
    avg_spending = subset["Spending Score (1-100)"].mean()
    count = len(subset)

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1e2a45, #162032);
                border-left: 5px solid {color};
                border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;">
        <h3 style="color: {color}; margin-top: 0;">{selected}</h3>
        <p style="color: #a8b2d8;">
            👥 <strong>{count}</strong> customers &nbsp;|&nbsp;
            💰 Avg Income: <strong>${avg_income:.0f}k</strong> &nbsp;|&nbsp;
            🛒 Avg Spending: <strong>{avg_spending:.0f}/100</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    recs = CLUSTER_RECS.get(cluster_id, [])
    for rec in recs:
        st.markdown(f"- {rec}")

    st.markdown("---")
    st.info("💬 These recommendations are derived from cluster characteristics. "
            "Combine with A/B testing and customer surveys for maximum impact.")

    # Quick predict tool
    st.markdown("### 🔮 Predict Segment for a New Customer")
    col_a, col_b = st.columns(2)
    with col_a:
        new_income = st.slider("Annual Income (k$)", 10, 150, 60)
    with col_b:
        new_spending = st.slider("Spending Score (1-100)", 1, 100, 50)

    X_new = scaler.transform([[new_income, new_spending]])
    pred_cluster = km.predict(X_new)[0]
    pred_name = CLUSTER_NAMES[pred_cluster] if pred_cluster < len(CLUSTER_NAMES) else f"Cluster {pred_cluster}"
    st.success(f"🎯 Predicted Segment: **{pred_name}**")
    st.markdown("**Recommended actions for this customer:**")
    for rec in CLUSTER_RECS.get(pred_cluster, []):
        st.markdown(f"  - {rec}")
