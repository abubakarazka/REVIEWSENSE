import streamlit as st
import pandas as pd
import joblib
import os

import plotly.graph_objects as go
import plotly.express as px
import base64
from collections import Counter

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="ReviewSense",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# CUSTOM CSS (UI TOKOPEDIA / SHOPEE)
# =========================
st.markdown("""
<style>
:root {
  --bg1: #f3f4f6; /* light gray */
  --bg2: #e5e7eb; /* slightly darker light gray */
  --accent1: #ff6b35; /* warm orange/red from logo */
  --accent2: #ffb703; /* golden yellow accent */
  --accent3: #3fb1ff; /* blue accent (magnifier) */
}
body {
    background: linear-gradient(135deg, var(--bg1), var(--bg2));
} 
.card {
    background: linear-gradient(180deg, rgba(255,107,53,0.10), rgba(255,183,3,0.06));
    color: #0f172a; /* dark text for contrast */
    padding: 22px;
    border-radius: 16px;
    border: 1px solid rgba(255,107,53,0.12);
    box-shadow: 0 8px 28px rgba(255,107,53,0.08);
    margin-bottom: 20px;
    transition: transform .12s ease, box-shadow .12s ease;
}
.card:hover { transform: translateY(-4px); box-shadow: 0 12px 48px rgba(255,107,53,0.12); }
.metric-title {
    font-size: 14px;
    color: #374151; /* darker gray for light warm card */
    margin-bottom: 8px;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: var(--accent1);
}
.section-title {
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 10px;
    background: linear-gradient(90deg, var(--accent1), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stMarkdown h2, .stMarkdown h3 { color: var(--accent1); }
.subtitle { color: rgba(85, 43, 22, 0.8); }
.stMarkdown h2, .stMarkdown h3 { color: var(--accent1); }
.metric-positive .metric-value { color: #2a9d8f; }
.metric-neutral .metric-value { color: #f59e0b; }
.metric-negative .metric-value { color: #ef4444; }
.metric-health .metric-value { color: var(--accent2); }
.stDataFrame thead th {
  background: linear-gradient(90deg, rgba(37,117,252,0.08), rgba(106,17,203,0.04));
}
.uploader-card {
  display:block;
  padding: 16px;
  border-radius: 12px;
  background: linear-gradient(90deg, rgba(255,107,53,0.06), rgba(255,183,3,0.04));
  border: 1px solid rgba(255,107,53,0.12);
  box-shadow: 0 6px 20px rgba(255,107,53,0.06);
  margin-bottom: 18px;
  color: #0f172a;
}

.navbar {
  display:flex;
  align-items:center;
  justify-content:space-between;
  padding:14px 18px;
  border-radius:12px;
  background: linear-gradient(90deg, var(--accent1), var(--accent2));
  border: 1px solid rgba(255,107,53,0.12);
  box-shadow: 0 8px 24px rgba(255,107,53,0.08);
  margin-bottom: 18px;
}

.navbar h2 { margin:0; font-size:20px; color:#ffffff; }
.navbar .subtitle { color: rgba(255,255,255,0.9); font-size:12px; }

</style>
""", unsafe_allow_html=True)

# =========================
# NAVBAR
# =========================
# Load logo image (if available) and prepare data URL for inline use
logo_data_url = None
try:
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "logoo.png")
    with open(logo_path, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    logo_data_url = f"data:image/png;base64,{logo_b64}"
except Exception:
    logo_data_url = None

nav_bar = st.container()
nav_cols = nav_bar.columns([3,1])
with nav_cols[0]:
    logo_html = f'<img src="{logo_data_url}" class="nav-logo" alt="ReviewSense logo" loading="lazy">' if logo_data_url else '<div style="font-size:28px">📊</div>'
    st.markdown(f"""
    <div class="navbar">
        <div style="display:flex;align-items:center;gap:12px">
            {logo_html}
            <div>
                <h2>ReviewSense</h2>
                <div class="subtitle">AI Review & Reputation Dashboard</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
# placeholder for the product filter selectbox (will be filled after data is loaded)
filter_placeholder = nav_cols[1].empty()

# Add faint background overlay using the same image (slightly transparent & blurred)
if logo_data_url:
    st.markdown(f"""
    <style> 
    body::before {{
        content: "";
        position: fixed;
        inset: 0;
        background-image: url("{logo_data_url}");
        background-repeat: no-repeat;
        background-position: center center;
        background-size: 60% auto; /* logo occupies central area */
        background-attachment: fixed;
        opacity: 0.15; /* semi-transparent so content remains readable */
        z-index: -1;
        pointer-events: none;
        filter: none;
    }}
    .nav-logo {{ height:48px; max-width:160px; object-fit:contain; border-radius:0; box-shadow:none; background:transparent; }}
    </style>
    """, unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "model", "sentiment_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "model", "vectorizer.pkl"))



# =========================
# UPLOAD CSV
# =========================
st.markdown("<div class='uploader-card'>", unsafe_allow_html=True)
st.markdown("### 📂 Upload CSV Review")
uploaded_file = st.file_uploader(
    "Pilih file CSV untuk analisis",
    type=["csv"]
)
st.markdown("</div>", unsafe_allow_html=True)

if not uploaded_file:
    st.info("Upload file CSV untuk memulai analisis")
    st.stop()

df = pd.read_csv(uploaded_file, engine="python")


# =========================
# VALIDASI KOLOM
# =========================
required_cols = ["product_name", "review_text"]
for col in required_cols:
    if col not in df.columns:
        st.error(f"Kolom `{col}` wajib ada")
        st.stop()

# =========================
# PREDICT SENTIMENT
# =========================
X = vectorizer.transform(df["review_text"].astype(str))
df["sentiment"] = model.predict(X)

# =========================
# NAVBAR FILTER
# =========================
products = ["SEMUA PRODUK"] + sorted(df["product_name"].unique())
selected_product = filter_placeholder.selectbox("Pilih Produk", products)

if selected_product == "SEMUA PRODUK":
    filtered_df = df.copy()
else:
    filtered_df = df[df["product_name"] == selected_product]

# =========================
# METRICS
# =========================
count = filtered_df["sentiment"].value_counts()
pos = count.get("positive", 0)
neu = count.get("neutral", 0)
neg = count.get("negative", 0)
total = pos + neu + neg
health = round((pos / total) * 100, 1) if total else 0

# =========================
# HEADER
# =========================
st.markdown(f"""
<div class="card">
    <div class="section-title">️ Dashboard Reputasi Produk</div>
    <p style="color:gray">
    Produk aktif: <b>{selected_product}</b>
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# KPI CARDS
# =========================
c1, c2, c3, c4 = st.columns(4)

def metric_card(col, title, value, emoji, css_class=""):
    with col:
        st.markdown(f"""
        <div class="card {css_class}">
            <div class="metric-title">{emoji} {title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

metric_card(c1, "Positive Review", pos, "😊", "metric-positive")
metric_card(c2, "Neutral Review", neu, "😐", "metric-neutral")
metric_card(c3, "Negative Review", neg, "😡", "metric-negative")
metric_card(c4, "Health Score", f"{health}%", "❤️", "metric-health")

# =========================
# CHART + INSIGHT
# =========================
left, right = st.columns([2,1])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    # Pie chart of sentiment proportions
    sentiment_counts = filtered_df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["sentiment", "count"]

    fig = px.pie(
        sentiment_counts,
        names="sentiment",
        values="count",
        color="sentiment",
        color_discrete_map={"positive":"#ff9100","neutral":"#0c1b72","negative":"#f8cb4d"},
        hole=0.4,
        labels={"count":"Jumlah", "sentiment":"Sentimen"}
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        title="Distribusi Sentimen",
        legend_title_text='Sentimen'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🤖 Insight AI")

    neg_reviews = filtered_df[filtered_df["sentiment"] == "negative"]["review_text"]

    if len(neg_reviews) == 0:
        st.success("🎉 Tidak ada masalah signifikan")
    else:
        words = " ".join(neg_reviews).lower().split()
        common = Counter(words).most_common(5)
        issues = ", ".join([w[0] for w in common])
        st.warning(f"Masalah utama: **{issues}**")

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# DETAIL REVIEW
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 📝 Detail Review")

st.dataframe(
    filtered_df[["product_name", "review_text", "sentiment"]],
    use_container_width=True,
    height=350
)

st.markdown("</div>", unsafe_allow_html=True)
