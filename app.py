import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


DATA_PATH = Path(__file__).with_name("netflix_data.csv")

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(
    page_title="Netflix Intelligence Dashboard",
    layout="wide"fgyi7ut7
)

# ---------------------------------
# Custom Styling
# ---------------------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
}
h1, h2, h3 {
    color: #ffffff;
}
.block-container {
    padding-top: 2rem;
}
div[data-testid="metric-container"] {
    background-color: #1f2937;
    padding: 15px;
    border-radius: 15px;
    border: 1px solid #374151;
    text-align: center;
}
.stSelectbox, .stMultiSelect, .stSlider {
    background-color: #111827;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Load Data
# ---------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["release_date"] = pd.to_datetime(
        df["release_date"],
        format="%d-%b-%y",
        errors="coerce",
    )
    df["release_year"] = df["release_date"].dt.year
    return df

df = load_data()

# ---------------------------------
# Recommendation System
# ---------------------------------
@st.cache_data
def build_model(data):
    combined_text = (
        data["genre"].fillna("") + " " +
        data["summary"].fillna("")
    )
    tfidf = TfidfVectorizer(stop_words="english")
    return tfidf.fit_transform(combined_text)

tfidf_matrix = build_model(df)

def recommend(title, df, matrix, top_n=5):
    idx = df[df["title"] == title].index[0]
    scores = list(enumerate(linear_kernel(matrix[idx], matrix).flatten()))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return df["title"].iloc[[i[0] for i in scores]].values

# ---------------------------------
# HEADER
# ---------------------------------
st.markdown("<h1 style='text-align:center;'>🎬 Netflix Content Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:lightgray;'>Advanced Analytics + NLP Recommendation Engine</p>", unsafe_allow_html=True)
st.divider()

# ---------------------------------
# SIDEBAR
# ---------------------------------
st.sidebar.header("🎛 Filter Content")

content_type = st.sidebar.multiselect(
    "Content Type",
    df["series_or_movie"].dropna().unique(),
    default=df["series_or_movie"].dropna().unique()
)

min_year = int(df["release_year"].min())
max_year = int(df["release_year"].max())
default_year_range = (max(min_year, 2015), min(max_year, 2021))

year_range = st.sidebar.slider(
    "Release Year",
    min_year,
    max_year,
    default_year_range
)

# Genre Filter
all_genres = set()
df["genre"].dropna().apply(lambda x: all_genres.update([g.strip() for g in x.split(",")]))

selected_genres = st.sidebar.multiselect(
    "Genres",
    sorted(all_genres)
)

filtered_df = df[
    (df["series_or_movie"].isin(content_type)) &
    (df["release_year"].between(year_range[0], year_range[1]))
]

if selected_genres:
    filtered_df = filtered_df[
        filtered_df["genre"].apply(
            lambda x: bool(set(g.strip() for g in x.split(",")) & set(selected_genres))
            if pd.notnull(x) else False
        )
    ]

# ---------------------------------
# KPI SECTION
# ---------------------------------
st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Titles", len(filtered_df))
col2.metric("Movies", len(filtered_df[filtered_df["series_or_movie"] == "Movie"]))
col3.metric("Series", len(filtered_df[filtered_df["series_or_movie"] == "Series"]))
avg_imdb = filtered_df["imdb_score"].mean()
col4.metric("Avg IMDb", round(avg_imdb, 2) if pd.notna(avg_imdb) else "N/A")

st.divider()

# ---------------------------------
# ROW 1: IMDb + Trend
# ---------------------------------
colA, colB = st.columns(2)

with colA:
    st.subheader("⭐ IMDb Score Distribution")
    fig_imdb = px.histogram(
        filtered_df,
        x="imdb_score",
        nbins=20,
        color_discrete_sequence=["#E50914"]
    )
    fig_imdb.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=30, b=10)
    )
    st.plotly_chart(fig_imdb, use_container_width=True)

with colB:
    st.subheader("📈 Content Release Trend")
    trend = filtered_df["release_year"].value_counts().sort_index().reset_index()
    trend.columns = ["Year", "Count"]

    fig_trend = px.line(
        trend,
        x="Year",
        y="Count",
        markers=True,
        color_discrete_sequence=["#00C9A7"]
    )
    fig_trend.update_layout(template="plotly_dark")
    st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

# ---------------------------------
# ROW 2: Top Countries
# ---------------------------------
st.subheader("🌍 Top 10 Countries")

country_series = (
    filtered_df["country_availability"]
    .dropna()
    .str.split(",")
    .explode()
    .str.strip()
)

top_countries = country_series.value_counts().head(10).reset_index()
top_countries.columns = ["Country", "Count"]

fig_country = px.bar(
    top_countries,
    x="Country",
    y="Count",
    color="Count",
    color_continuous_scale="Viridis"
)

fig_country.update_layout(template="plotly_dark")
st.plotly_chart(fig_country, use_container_width=True)

st.divider()

# ---------------------------------
# RECOMMENDATION SECTION
# ---------------------------------
st.subheader("🤖 AI Recommendation Engine")

colR1, colR2 = st.columns([2,1])

with colR1:
    selected_title = st.selectbox(
        "Select a Title",
        df["title"].sort_values()
    )

with colR2:
    recommend_btn = st.button("Generate Recommendations")

if recommend_btn:
    recs = recommend(selected_title, df, tfidf_matrix)

    st.markdown("### 🎯 Recommended For You")
    for r in recs:
        st.markdown(f"• **{r}**")
