import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import requests

# Resolve paths relative to this script (works on Streamlit Cloud + local)
_APP_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="App TrendPulse Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# SIDEBAR / NAVIGATION & THEME
# -------------------------------
logo_paths = [
    os.path.join(_APP_DIR, "..", "assets", "logo.png"),
    os.path.join(_APP_DIR, "assets", "logo.png"),
]
for path in logo_paths:
    if os.path.exists(path):
        st.sidebar.image(path, use_container_width=True)
        break

st.sidebar.markdown("### 🧭 Quick Navigation", unsafe_allow_html=True)
st.sidebar.markdown("""
<a href='#kpi-metrics' class='nav-link' style='display:block; padding: 5px 0;'>📊 KPI Overview</a>
<a href='#monetization-engagement' class='nav-link' style='display:block; padding: 5px 0;'>💰 Monetization & Activity</a>
<a href='#market-share-pipeline' class='nav-link' style='display:block; padding: 5px 0;'>📈 Market Install Pipeline</a>
<a href='#temporal-heatmap' class='nav-link' style='display:block; padding: 5px 0;'>🕒 Temporal Campaign Heatmap</a>
<a href='#smart-recommendations' class='nav-link' style='display:block; padding: 5px 0;'>🤖 Smart Recommendations</a>
<hr>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🌗 Theme Settings")
if "theme" not in st.session_state:
    st.session_state["theme"] = "Dark 🌙"

theme_choice = st.sidebar.radio("Select Dashboard UI Theme:", ["Dark 🌙", "Light ☀️"], key="theme")
is_dark = "Dark" in theme_choice

# Theme Colors Map
bg_color = "#0f172a" if is_dark else "#f8fafc"
card_bg = "#1e293b" if is_dark else "#ffffff"
text_primary = "#f8fafc" if is_dark else "#0f172a"
text_secondary = "#94a3b8" if is_dark else "#64748b"
border_color = "#334155" if is_dark else "#e2e8f0"
shadow_color = "rgba(0,0,0,0.3)" if is_dark else "rgba(0,0,0,0.05)"
hover_shadow = "rgba(0,0,0,0.5)" if is_dark else "rgba(0,0,0,0.12)"
PLOTLY_THEME = "plotly_dark" if is_dark else "plotly_white"

# -------------------------------
# DYNAMIC CSS ENGINE
# -------------------------------
css = f"""
<style>
.stApp {{ background-color: {bg_color}; transition: background-color 0.4s ease; }}

@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

.kpi-card {{
    background-color: {card_bg};
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px {shadow_color};
    border: 1px solid {border_color};
    text-align: center;
    margin-bottom: 20px;
    animation: fadeInUp 0.5s ease-out both;
    transition: transform 0.2s ease, box-shadow 0.2s ease, background-color 0.4s ease;
}}

div[data-testid="column"]:nth-child(1) .kpi-card {{ animation-delay: 0.1s; }}
div[data-testid="column"]:nth-child(2) .kpi-card {{ animation-delay: 0.2s; }}
div[data-testid="column"]:nth-child(3) .kpi-card {{ animation-delay: 0.3s; }}
div[data-testid="column"]:nth-child(4) .kpi-card {{ animation-delay: 0.4s; }}

.kpi-card:hover {{
    transform: translateY(-5px);
    box-shadow: 0px 10px 25px {hover_shadow};
}}

.kpi-title {{ color: {text_secondary}; font-size: 14px; font-weight: 700; text-transform: uppercase; margin-bottom: 5px; }}
.kpi-value {{ color: {text_primary}; font-size: 30px; font-weight: 800; }}

.stPlotlyChart {{
    background-color: {card_bg};
    border-radius: 12px;
    padding: 15px;
    border: 1px solid {border_color};
    box-shadow: 0px 4px 10px {shadow_color};
    transition: transform 0.2s ease, background-color 0.4s ease;
    animation: fadeInUp 0.6s ease-out both;
}}
.stPlotlyChart:hover {{ transform: scale(1.02); }}

h1, h2, h3, h4, p, span {{ color: {text_primary} !important; font-family: 'Segoe UI', Tahoma, Verdana, sans-serif; transition: color 0.4s ease; }}
.stTabs [data-baseweb="tab"] {{ color: {text_primary}; }}

.sync-badge {{
    background-color: #059669; color: white !important; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; display: inline-flex; align-items: center; gap: 5px;
}}

.nav-link {{
    color: #14b8a6 !important;
    font-weight: bold;
    text-decoration: none;
    transition: color 0.2s ease;
}}
.nav-link:hover {{
    color: #0d9488 !important;
    text-decoration: underline;
}}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# -------------------------------
# DATA LOADING (LIVE PULL)
# -------------------------------
@st.cache_data(ttl=2)
def load_data():
    """Fetch dataset from FastAPI first; fall back to local CSV if API is down."""
    try:
        response = requests.get("http://127.0.0.1:8000/data", timeout=5)
        if response.status_code == 200:
            df = pd.DataFrame(response.json())
            return df
    except requests.exceptions.RequestException:
        pass

    df = pd.read_csv(os.path.join(_APP_DIR, "dataset.csv"))
    df = df.dropna(subset=["App"]).reset_index(drop=True)
    return df

@st.cache_data(ttl=5)
def load_reviews():
    """Load customer review logs with timestamps."""
    try:
        reviews = pd.read_csv(os.path.join(_APP_DIR, "reviews_store.csv"))
        reviews["Timestamp"] = pd.to_datetime(reviews["Timestamp"], errors="coerce")
        return reviews
    except FileNotFoundError:
        return pd.DataFrame()

def build_temporal_data(reviews_df, app_df):
    """Derive Hour/Day activity from real review timestamps instead of random noise."""
    if reviews_df.empty or "Timestamp" not in reviews_df.columns:
        # Ultimate fallback: generate uniform distribution (clearly labeled as synthetic)
        np.random.seed(42)
        app_df = app_df.copy()
        app_df["Hour"] = np.random.randint(0, 24, len(app_df))
        app_df["Day"] = np.random.choice(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], len(app_df))
        return app_df, True  # True = synthetic

    # Extract real temporal features from review timestamps
    temp = reviews_df.copy()
    temp["Hour"] = temp["Timestamp"].dt.hour
    temp["Day"] = temp["Timestamp"].dt.strftime("%a")  # Mon, Tue, etc.
    return temp[["App", "Hour", "Day"]], False

try:
    df_raw = load_data()
    reviews_df_cache = load_reviews()
    temporal_df, is_synthetic = build_temporal_data(reviews_df_cache, df_raw)
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.stop()


# -------------------------------
# RECOMMENDATION ENGINE (API + LOCAL FALLBACK)
# -------------------------------
def fetch_recommendations(app_name):
    """Ask the API for recommendations; fall back to local cosine similarity."""
    try:
        response = requests.post(
            "http://127.0.0.1:8000/recommend",
            json={"app_name": app_name, "top_n": 5},
            timeout=20,
        )
        if response.status_code == 200:
            return response.json().get("recommendations", [])
    except requests.exceptions.RequestException:
        pass

    # Local fallback
    from sklearn.metrics.pairwise import cosine_similarity
    features = df_raw[["Rating", "Sentiment", "Engagement_Score"]].fillna(0)
    sim = cosine_similarity(features)
    idx = df_raw[df_raw["App"] == app_name].index[0]
    scores = sorted(enumerate(sim[idx]), key=lambda x: x[1], reverse=True)
    return [df_raw.iloc[i[0]]["App"] for i in scores[1:6]]


# -------------------------------
# SIDEBAR / SLICERS
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("## ⚙️ Global Filters")

genres = sorted(df_raw["Genre"].dropna().unique().tolist())
selected_genres = st.sidebar.multiselect("Select Genre:", options=genres, default=genres[:3] if len(genres) > 3 else genres)

monetizations = sorted(df_raw["Monetization"].dropna().unique().tolist())
selected_monetization = st.sidebar.multiselect("Select Monetization:", options=monetizations, default=monetizations)

engagements = sorted(df_raw["Engagement_Level"].dropna().unique().tolist())
selected_engagement = st.sidebar.multiselect("Select Engagement Level:", options=engagements, default=engagements)

df = df_raw.copy()
if selected_genres:
    df = df[df["Genre"].isin(selected_genres)]
if selected_monetization:
    df = df[df["Monetization"].isin(selected_monetization)]
if selected_engagement:
    df = df[df["Engagement_Level"].isin(selected_engagement)]

# -------------------------------
# MARKETING AI LAUNCH PREDICTOR
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 ML Launch Predictor")
st.sidebar.info("Simulate campaign metrics to predict Market Engagement.")
with st.sidebar.form("ml_predict"):
    p_genre = st.selectbox("Target Genre:", options=genres if genres else ["Unknown"])
    p_mon = st.selectbox("Monetization Target:", options=monetizations if monetizations else ["Free"])
    p_inst = st.number_input("Expected Launch Installs:", min_value=1000, value=500000, step=100000)
    p_sent = st.slider("Goal Public Sentiment:", -1.0, 1.0, 0.2, 0.05)

    if st.form_submit_button("Predict Success Score", use_container_width=True):
        try:
            res = requests.post("http://127.0.0.1:8000/predict", json={
                "installs": p_inst, "sentiment": p_sent, "genre": p_genre, "monetization": p_mon
            }, timeout=20)
            if res.status_code == 200:
                score = res.json().get("predicted_engagement_score", 0)
                st.sidebar.success(f"🏆 Expected Engagement: **{score}** / 10.0")
            else:
                st.sidebar.error("API Error: Check backend logs.")
        except Exception:
            st.sidebar.error("API unreachable — ensure Uvicorn is running.")

# -------------------------------
# HEADER
# -------------------------------
col_h1, col_h2 = st.columns([0.8, 0.2])
with col_h1:
    st.markdown("<h1 style='animation: fadeInUp 0.5s ease-out both;'>🚀 App TrendPulse Analytics</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {text_secondary} !important; margin-top: -15px;'>Professional App Analytics & Market Intelligence Dashboard</p>", unsafe_allow_html=True)
with col_h2:
    st.markdown("<br><div class='sync-badge'>🟢 LIVE API SYNC</div>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------
# TABS
# -------------------------------
tab1, tab2 = st.tabs(["📊 Dashboard Overview", "💬 Admin Console: Feedback Logs"])


# ---- Helper: KPI Card Renderer ----
def render_kpi(col, title, value, delta=None):
    """Render a single animated KPI card with an optional trend delta."""
    delta_html = ""
    if delta is not None:
        color = "#10b981" if delta > 0 else "#ef4444" if delta < 0 else text_secondary
        arrow = "▲" if delta > 0 else "▼" if delta < 0 else "-"
        delta_html = f"<div style='color: {color}; font-size: 14px; font-weight: bold; margin-top: 5px;'>{delta:+.3f} {arrow} (vs Baseline)</div>"

    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


with tab1:
    st.markdown("<a id='top'></a>", unsafe_allow_html=True)

    if df.empty:
        st.warning("No data available for the selected filters.")
    else:
        st.markdown("<a id='kpi-metrics'></a>", unsafe_allow_html=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Export Current View (CSV)",
            data=csv,
            file_name="TrendPulse_Filtered_Export.csv",
            mime="text/csv",
        )
        st.write("")

        # KPI CARDS + DELTAS
        col1, col2, col3, col4 = st.columns(4)

        render_kpi(col1, "📱 Total Applications", f"{len(df):,}")

        rating_delta = df["Rating"].mean() - df_raw["Rating"].mean() if not np.isnan(df["Rating"].mean()) else 0
        sentiment_delta = df["Sentiment"].mean() - df_raw["Sentiment"].mean() if not np.isnan(df["Sentiment"].mean()) else 0

        render_kpi(col2, "⭐ Average Rating", f"{df['Rating'].mean():.2f}", delta=rating_delta)
        render_kpi(col3, "❤️ Average Sentiment", f"{df['Sentiment'].mean():.2f}", delta=sentiment_delta)

        top_app = df.sort_values("Engagement_Score", ascending=False)["App"].iloc[0] if not df.empty else "N/A"
        render_kpi(col4, "🏆 Top Performer", top_app[:20] + ("..." if len(top_app) > 20 else ""))

        st.markdown("<p style='text-align: right;'><a href='#top' class='nav-link' style='font-size: 13px;'>↑ Back to Top</a></p>", unsafe_allow_html=True)

        # ---------- DONUTS ----------
        st.markdown("---")
        st.markdown("<a id='monetization-engagement'></a>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            monet = df["Monetization"].value_counts().reset_index()
            monet.columns = ["Monetization Type", "Application Count"]
            fig1 = px.pie(monet, names="Monetization Type", values="Application Count", hole=0.45, color_discrete_sequence=px.colors.sequential.Teal, title="💰 Monetization Distribution")
            fig1.update_traces(textinfo="percent+label", textfont_size=14, hovertemplate="%{label}: %{value} Apps<br>%{percent}")
            fig1.update_layout(template=PLOTLY_THEME, margin=dict(t=50, b=20, l=0, r=0), transition_duration=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            eng_pie = df["Engagement_Level"].value_counts().reset_index()
            eng_pie.columns = ["Engagement Level", "Application Count"]
            fig2 = px.pie(eng_pie, names="Engagement Level", values="Application Count", hole=0.45, color_discrete_sequence=px.colors.sequential.Mint, title="🔥 Activity Engagement Levels")
            fig2.update_traces(textinfo="percent+label", textfont_size=14, hovertemplate="%{label}: %{value} Apps<br>%{percent}")
            fig2.update_layout(template=PLOTLY_THEME, margin=dict(t=50, b=20, l=0, r=0), transition_duration=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("<p style='text-align: right;'><a href='#top' class='nav-link' style='font-size: 13px;'>↑ Back to Top</a></p>", unsafe_allow_html=True)

        # ---------- BARS ----------
        st.markdown("---")
        st.markdown("<a id='market-share-pipeline'></a>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)

        with c3:
            genre_share = df.groupby("Genre")["Market_Share"].sum().reset_index().sort_values(by="Market_Share", ascending=True)
            fig3 = px.bar(genre_share, x="Market_Share", y="Genre", orientation="h", color="Market_Share", color_continuous_scale="Teal", title="📈 Aggregate Market Share by Genre", labels={"Market_Share": "Market Share (%)", "Genre": "Application Genre"})
            fig3.update_layout(template=PLOTLY_THEME, margin=dict(t=50, b=20, l=10, r=10), transition_duration=800, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig3, use_container_width=True)

        with c4:
            fig4 = px.scatter(df, x="Installs", y="Engagement_Score", color="Segment", size="Rating", hover_name="App", log_x=True, color_discrete_sequence=px.colors.qualitative.Pastel, title="⚡ Engagement Score vs. Total Installs", labels={"Installs": "Total Installs (Log Scale)", "Engagement_Score": "Engagement Score", "Segment": "Market Segment"})
            fig4.update_layout(template=PLOTLY_THEME, margin=dict(t=50, b=20, l=10, r=10), transition_duration=800, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("<p style='text-align: right;'><a href='#top' class='nav-link' style='font-size: 13px;'>↑ Back to Top</a></p>", unsafe_allow_html=True)

        # ---------- TEMPORAL HEATMAP ----------
        st.markdown("---")
        st.markdown("<a id='temporal-heatmap'></a>", unsafe_allow_html=True)
        st.markdown("### 🕒 Temporal Campaign Analysis")
        if is_synthetic:
            st.warning("⚠️ Heatmap uses simulated data. Run the notebook scraper to populate real review timestamps.")
        else:
            st.info("Built from real user review timestamps — use this to time Ad Campaigns & Push Notifications.")

        fig_time = px.density_heatmap(
            temporal_df, x="Hour", y="Day",
            title="User Activity Density by Time & Day",
            color_continuous_scale="Viridis",
            labels={"Hour": "Time of Day (24H)", "Day": "Day of the Week"},
            category_orders={"Day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]},
        )
        fig_time.update_layout(template=PLOTLY_THEME, margin=dict(t=50, b=20, l=10, r=10), transition_duration=800, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_time, use_container_width=True)

        st.markdown("<p style='text-align: right;'><a href='#top' class='nav-link' style='font-size: 13px;'>↑ Back to Top</a></p>", unsafe_allow_html=True)

        # ---------- RECOMMENDATIONS ----------
        st.markdown("---")
        st.markdown("<a id='smart-recommendations'></a>", unsafe_allow_html=True)
        st.markdown("### 🤖 Market Insights: App Recommendations")

        rec_app = st.selectbox("Select an App to find structurally similar applications:", options=df_raw["App"])
        if st.button("Generate Smart Recommendations"):
            with st.spinner("Computing cosine similarity vectors..."):
                recs = fetch_recommendations(rec_app)
                if recs:
                    st.success("Recommendations Generated!")
                    with st.expander("🔍 See Similarity Analysis Details", expanded=True):
                        st.info(f"The model compared the Sentiment, Rating, and Engagement Score of **{rec_app}** against all tracked apps using cosine similarity to surface these 5 closest matches:")
                        cols = st.columns(max(len(recs), 1))
                        for i, r in enumerate(recs):
                            with cols[i]:
                                with st.container(border=True):
                                    st.markdown(f"🏅 **{r}**")
                else:
                    st.warning("No recommendations returned for this application.")

        st.markdown("<p style='text-align: right;'><a href='#top' class='nav-link' style='font-size: 13px;'>↑ Back to Top</a></p>", unsafe_allow_html=True)

with tab2:
    st.markdown("### ✏️ Customer Review Control Center")
    st.info("Double-click any cell to edit. Add or delete rows, then save to disk.")

    if not reviews_df_cache.empty:
        reviews_view = reviews_df_cache.copy()
        reviews_view = reviews_view.sort_values(by="Timestamp", ascending=False).reset_index(drop=True)

        edited_df = st.data_editor(
            reviews_view,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Timestamp": st.column_config.DatetimeColumn("Date & Time", format="YYYY-MM-DD HH:mm:ss", required=True),
                "User_Name": st.column_config.TextColumn("User Name", max_chars=100, required=True),
                "App": st.column_config.SelectboxColumn("Associated App", options=df_raw["App"].tolist(), required=True),
                "Rating": st.column_config.NumberColumn("Rating (1-5)", min_value=1, max_value=5, step=1, required=True),
                "Log": st.column_config.TextColumn("Review Log Text"),
            },
            hide_index=True,
            height=450,
        )

        c_left, c_right = st.columns([0.2, 0.8])
        with c_left:
            if st.button("💾 Save Changes to Disk", use_container_width=True, type="primary"):
                try:
                    edited_df.to_csv(os.path.join(_APP_DIR, "reviews_store.csv"), index=False)
                    st.success("Reviews saved successfully.")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Save failed: {e}")
    else:
        st.warning("No customer reviews available.")
