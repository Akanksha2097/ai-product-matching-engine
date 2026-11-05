import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.models.matching import MatchingEngine
from src.preprocessing.normalization import normalize_catalog
from src.review.store import ReviewStore

ROOT = Path(__file__).parents[1]
st.set_page_config(page_title="MatchWise AI", page_icon="✦", layout="wide")
source = st.sidebar.selectbox("Catalog", ["Real food products (Open Food Facts)", "Synthetic variant examples"])
catalog_file = "openfoodfacts.csv" if source.startswith("Real") else "catalog.csv"
catalog = normalize_catalog(pd.read_csv(ROOT / "data/demo" / catalog_file))
engine = MatchingEngine(catalog)

st.markdown("""<style>.stApp{background:#f7fafc}.hero{padding:1.4rem 1.8rem;background:#0b1f33;color:white;border-radius:14px}.accent{color:#2dd4bf}</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>MatchWise <span class="accent">AI</span></h1><p>Intelligent Product Matching & Catalog Entity Resolution</p></div>', unsafe_allow_html=True)
page = st.sidebar.radio("Navigate", ["Overview", "Product Matcher", "Batch Matching", "Human Review", "Model Evaluation", "Technical Architecture"])
st.sidebar.caption("Food catalog: Open Food Facts contributors · ODbL 1.0. The synthetic catalog is separate.")

if page == "Overview":
    st.subheader("Make catalog duplication measurable")
    a, b, c = st.columns(3)
    a.metric("Reference products", len(catalog)); b.metric("Evaluated approaches", 4); c.metric("FAISS recall@10", "97.86%")
    st.write("MatchWise combines conservative normalization, blocked TF-IDF retrieval, semantic embeddings, and review-oriented decisions. The real benchmark and food demo are separate datasets.")
    st.info("Benchmark metrics are measured on Abt–Buy. They do not establish accuracy on the food demo catalog.")
    st.code("Query → normalize → retrieve candidates → score → auto-match / review / reject", language="text")
elif page == "Product Matcher":
    default_query = str(catalog.iloc[0].title) if source.startswith("Real") else "Hewlett Packard 67 XL High Yield Black Ink"
    query = st.text_input("Product description", default_query)
    model = st.selectbox("Scoring approach", ["hybrid", "fuzzy", "tfidf"])
    if query:
        results = engine.match(query, model=model)
        detail = pd.DataFrame([r.__dict__ for r in results]).merge(
            catalog[["product_id", "title", "brand_norm", "quantity_norm"]], on="product_id"
        )
        st.dataframe(detail, use_container_width=True)
        if results: st.write("Decision explanation:", "; ".join(results[0].reasons))
elif page == "Batch Matching":
    uploaded = st.file_uploader("Upload CSV with a query_title column", type="csv")
    frame = pd.read_csv(uploaded) if uploaded else pd.DataFrame({"query_title": ["HP 67 XL black ink cartridge", "Sony XM4 headphones"]})
    if "query_title" not in frame: st.error("CSV must contain query_title"); st.stop()
    output = []
    for query in frame.query_title.astype(str):
        top = engine.match(query, k=1)[0]
        output.append({"query_title": query, "product_id": top.product_id, "score": top.score, "decision": top.decision})
    result = pd.DataFrame(output); st.dataframe(result, use_container_width=True); st.download_button("Download matches", result.to_csv(index=False), "matchwise_results.csv", "text/csv")
elif page == "Human Review":
    st.subheader("Ambiguous match queue")
    query = st.text_input("Review query", str(catalog.iloc[0].title))
    result = engine.match(query, k=3)[0]
    st.write(result.__dict__)
    choice = st.radio("Reviewer action", ["accept", "reject", "needs_evidence"])
    note = st.text_input("Correction or note")
    store = ReviewStore(ROOT / "data/reviews.sqlite")
    if st.button("Record review"):
        store.record(query, result.product_id, choice, note)
        st.success("Review saved locally.")
    history = store.history()
    st.dataframe(history, use_container_width=True)
    st.download_button("Export review history", history.to_csv(index=False), "review_history.csv", "text/csv")
    st.caption("Hosted Streamlit filesystems can reset; export reviews before closing a hosted session.")
elif page == "Model Evaluation":
    eval_path = ROOT / "data/demo/benchmark_evaluation.json"
    if eval_path.exists():
        metrics = json.loads(eval_path.read_text())
        comparison = pd.DataFrame([{"model": name, **values["test"]} for name, values in metrics["comparison"].items()])
        st.plotly_chart(px.bar(comparison, x="model", y=["precision", "recall", "f1"], barmode="group", title="Abt–Buy held-out test metrics"), use_container_width=True)
        st.dataframe(comparison.drop(columns="confusion_matrix"), use_container_width=True)
        st.write("FAISS candidate retrieval:", metrics["retrieval"])
        selected = st.selectbox("Confusion matrix", comparison.model.tolist())
        matrix = comparison.set_index("model").loc[selected, "confusion_matrix"]
        st.plotly_chart(px.imshow(matrix, text_auto=True, x=["predicted no", "predicted yes"], y=["actual no", "actual yes"], title=selected), use_container_width=True)
        st.caption("Thresholds target ≥90% validation precision; test precision can be lower. TF-IDF outperformed the hybrid at this operating point. The demo catalog has no benchmark labels.")
    else: st.warning("Run scripts/evaluate_benchmark.py to generate measured benchmark output.")
elif page == "Technical Architecture":
    st.subheader("Architecture")
    st.code("Public catalog → validation/normalization → Parquet\n                         ↘ TF-IDF blocking → hybrid scoring → FastAPI / Streamlit\nLabels → evaluation → thresholds → human review", language="text")
    st.write("Sources: Open Food Facts (ODbL 1.0) for the food demo; DeepMatcher Abt–Buy for measured evaluation. No linkage is assumed between them.")
    st.link_button("GitHub repository", "https://github.com/Akanksha2097/ai-product-matching-engine")
