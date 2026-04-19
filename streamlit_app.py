import streamlit as st
import joblib, re

# --- STOPWORDS ---
STOPWORDS = set([
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up',
    'down','in','out','on','off','over','under','again','further','then',
    'once','here','there','when','where','why','how','all','both','each',
    'few','more','most','other','some','such','no','nor','not','only','own',
    'same','so','than','too','very','s','t','can','will','just','don',
    'should','now','d','ll','m','o','re','ve','y','ain','aren','couldn',
    'didn','doesn','hadn','hasn','haven','isn','mightn','mustn','needn',
    'shan','shouldn','wasn','weren','won','wouldn','also','would','could',
    'shall','may','might','must','need','dare','used','br','film','movie'
])

# --- PREPROCESSING ---
def remove_html_tags(text): return re.sub(r'<[^>]+>', ' ', text)
def remove_urls(text): return re.sub(r'http\S+|www\.\S+', '', text)
def remove_special_characters(text): return re.sub(r'[^a-z\s]', '', text)
def tokenize(text): return text.split()
def remove_stopwords(tokens): return [t for t in tokens if t not in STOPWORDS]

def simple_stem(word):
    suffixes = ['ing','tion','ness','ment','able','ible','ful','less',
                'ous','ive','er','est','ed','ly','s']
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word

def apply_stemming(tokens): return [simple_stem(t) for t in tokens]

def preprocess(text):
    text = text.lower()
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_special_characters(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = apply_stemming(tokens)
    tokens = [t for t in tokens if len(t) > 2]
    return ' '.join(tokens)

# --- LOAD MODEL (cached — loads only once per session) ---
@st.cache_resource
def load_model():
    model = joblib.load("LR_model.pkl")
    vectorizer = joblib.load("LR_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# --- INFERENCE ---
def analyze_single(review_text):
    cleaned = preprocess(review_text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    return {
        "sentiment": "POSITIVE" if pred == 1 else "NEGATIVE",
        "confidence": round(float(max(prob)), 4),
        "label": int(pred)
    }

def analyze_bulk(reviews_list):
    reviews = [r.strip() for r in reviews_list if r.strip()]
    cleaned_texts = [preprocess(r) for r in reviews]
    vecs = vectorizer.transform(cleaned_texts)
    preds = model.predict(vecs)
    probs = model.predict_proba(vecs)

    predictions = [
        {
            "text": reviews[i],
            "sentiment": "POSITIVE" if preds[i] == 1 else "NEGATIVE",
            "confidence": round(float(max(probs[i])), 4),
            "label": int(preds[i])
        }
        for i in range(len(reviews))
    ]

    pos_count = sum(1 for p in predictions if p["sentiment"] == "POSITIVE")
    neg_count = len(predictions) - pos_count
    avg_conf = round(sum(p["confidence"] for p in predictions) / len(predictions), 4)
    return {
        "predictions": predictions,
        "summary": {
            "total": len(predictions),
            "positive_count": pos_count,
            "negative_count": neg_count,
            "positive_percentage": round((pos_count / len(predictions)) * 100, 1),
            "avg_confidence": avg_conf
        }
    }

# ===================== UI =====================

st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6c757d;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .result-positive-high {
        background: #1a5c2a;
        color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .result-positive-mid {
        background: #2d6a3f;
        color: #d4edda;
        border-left: 5px solid #5cb85c;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .result-positive-low {
        background: #3d7a4f;
        color: #e8f5e9;
        border-left: 5px solid #8bc34a;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .result-negative-high {
        background: #5c1a1a;
        color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .result-negative-mid {
        background: #6a2d2d;
        color: #f8d7da;
        border-left: 5px solid #e05c5c;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .result-negative-low {
        background: #7a3d3d;
        color: #fce4e4;
        border-left: 5px solid #f08080;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .review-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
    }
    .badge-pos-high  { background:#28a745; color:white; padding:2px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }
    .badge-pos-mid   { background:#5cb85c; color:white; padding:2px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }
    .badge-pos-low   { background:#8bc34a; color:white; padding:2px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }
    .badge-neg-high  { background:#dc3545; color:white; padding:2px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }
    .badge-neg-mid   { background:#e05c5c; color:white; padding:2px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }
    .badge-neg-low   { background:#f08080; color:white; padding:2px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }
    .summary-box-pos {
        background: linear-gradient(135deg, #1a5c2a, #2d8a45);
        color: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .summary-box-neg {
        background: linear-gradient(135deg, #5c1a1a, #8a2d2d);
        color: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .summary-box-mixed {
        background: linear-gradient(135deg, #4a3800, #7a6000);
        color: #fff8dc;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .stProgress > div > div > div { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


def get_style_class(sentiment, confidence):
    if sentiment == "POSITIVE":
        if confidence >= 80: return "result-positive-high", "badge-pos-high"
        elif confidence >= 60: return "result-positive-mid", "badge-pos-mid"
        else: return "result-positive-low", "badge-pos-low"
    else:
        if confidence >= 80: return "result-negative-high", "badge-neg-high"
        elif confidence >= 60: return "result-negative-mid", "badge-neg-mid"
        else: return "result-negative-low", "badge-neg-low"


def get_progress_color(sentiment, confidence):
    if sentiment == "POSITIVE":
        if confidence >= 80: return "#28a745"
        elif confidence >= 60: return "#5cb85c"
        else: return "#8bc34a"
    else:
        if confidence >= 80: return "#dc3545"
        elif confidence >= 60: return "#e05c5c"
        else: return "#f08080"


st.markdown('<div class="main-title">🎬 IMDb Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by Logistic Regression + TF-IDF</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Single Review", "Bulk Analysis"])

with tab1:
    st.markdown("#### Paste a movie review")
    single_review = st.text_area(
        label="Review input",
        label_visibility="collapsed",
        placeholder="Type or paste a movie review here...",
        height=130,
        key="single_input"
    )

    if st.button("Analyze", key="single_btn", use_container_width=False):
        if not single_review.strip():
            st.warning("Please enter a review first.")
        else:
            with st.spinner("Analyzing..."):
                result = analyze_single(single_review.strip())

            sentiment = result.get("sentiment", "UNKNOWN")
            confidence = round(result.get("confidence", 0) * 100, 1)
            result_class, _ = get_style_class(sentiment, confidence)
            emoji = "😊" if sentiment == "POSITIVE" else "😞"
            color = get_progress_color(sentiment, confidence)

            st.markdown(f"""
            <div class="{result_class}">
                <div style="font-size:1.4rem; font-weight:700;">{emoji} {sentiment}</div>
                <div style="font-size:0.9rem; margin-top:4px; opacity:0.85;">Confidence: {confidence}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="margin-top:12px;">
                <div style="height:10px; background:#e9ecef; border-radius:10px; overflow:hidden;">
                    <div style="height:100%; width:{confidence}%; background:{color}; border-radius:10px; transition:width 0.4s;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


with tab2:
    st.markdown("#### Add reviews one by one or paste multiple lines")

    if "bulk_reviews" not in st.session_state:
        st.session_state.bulk_reviews = []

    col1, col2 = st.columns([4, 1])
    with col1:
        new_review = st.text_area(
            "Add review",
            label_visibility="collapsed",
            placeholder="Paste a review to add to the list...",
            height=90,
            key="bulk_add_input"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ Add", use_container_width=True):
            if new_review.strip():
                st.session_state.bulk_reviews.append(new_review.strip())
                st.rerun()

    st.markdown("**Or paste multiple reviews (one per line):**")
    multi_paste = st.text_area(
        "Multi-line paste",
        label_visibility="collapsed",
        placeholder="Review 1\nReview 2\nReview 3\n...",
        height=90,
        key="multi_paste"
    )
    if st.button("Add All Lines", use_container_width=False):
        lines = [l.strip() for l in multi_paste.strip().split("\n") if l.strip()]
        if lines:
            st.session_state.bulk_reviews.extend(lines)
            st.rerun()

    if st.session_state.bulk_reviews:
        st.markdown(f"**{len(st.session_state.bulk_reviews)} review(s) queued:**")
        for i, rev in enumerate(st.session_state.bulk_reviews):
            col_r, col_del = st.columns([9, 1])
            with col_r:
                st.markdown(f"""
                <div class="review-card">
                    <small style="color:#6c757d;">#{i+1}</small><br>
                    <span style="font-size:0.9rem;">{rev[:150]}{"..." if len(rev) > 150 else ""}</span>
                </div>
                """, unsafe_allow_html=True)
            with col_del:
                if st.button("🗑", key=f"del_{i}"):
                    st.session_state.bulk_reviews.pop(i)
                    st.rerun()

        col_analyze, col_clear = st.columns([3, 1])
        with col_analyze:
            analyze_bulk_btn = st.button("Analyze All Reviews →", use_container_width=True, type="primary")
        with col_clear:
            if st.button("Clear All", use_container_width=True):
                st.session_state.bulk_reviews = []
                if "bulk_results" in st.session_state:
                    del st.session_state.bulk_results
                st.rerun()

        if analyze_bulk_btn:
            with st.spinner(f"Analyzing {len(st.session_state.bulk_reviews)} reviews..."):
                result = analyze_bulk(st.session_state.bulk_reviews)
            if "error" in result:
                st.error(result["error"])
            else:
                st.session_state.bulk_results = result

    if "bulk_results" in st.session_state:
        results = st.session_state.bulk_results
        predictions = results.get("predictions", [])
        summary = results.get("summary", {})

        pos_count = summary.get("positive_count", 0)
        neg_count = summary.get("negative_count", 0)
        total = summary.get("total", len(predictions))
        pos_pct = round((pos_count / total) * 100) if total > 0 else 0
        avg_conf = round(summary.get("avg_confidence", 0) * 100, 1)

        st.markdown("---")
        st.markdown("### Results")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total", total)
        m2.metric("Positive", pos_count, delta=f"{pos_pct}%")
        m3.metric("Negative", neg_count, delta=f"-{100-pos_pct}%", delta_color="inverse")
        m4.metric("Avg Confidence", f"{avg_conf}%")

        st.markdown("**Positive vs Negative split:**")
        st.progress(pos_pct / 100)
        st.caption(f"{pos_pct}% Positive  |  {100-pos_pct}% Negative")

        if pos_pct >= 70:
            verdict_class = "summary-box-pos"
            verdict_text = "Mostly Positive Reception"
            verdict_sub = f"{pos_pct}% of reviews are positive — audiences generally enjoyed this."
        elif pos_pct <= 30:
            verdict_class = "summary-box-neg"
            verdict_text = "Mostly Negative Reception"
            verdict_sub = f"{100-pos_pct}% of reviews are negative — audiences were largely disappointed."
        else:
            verdict_class = "summary-box-mixed"
            verdict_text = "Mixed Reception"
            verdict_sub = f"Divided audience — {pos_pct}% positive, {100-pos_pct}% negative."

        st.markdown(f"""
        <div class="{verdict_class}">
            <div style="font-size:1.3rem; font-weight:700;">{verdict_text}</div>
            <div style="font-size:0.9rem; margin-top:6px; opacity:0.9;">{verdict_sub}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>**Individual review results:**", unsafe_allow_html=True)
        for i, pred in enumerate(predictions):
            sentiment = pred.get("sentiment", "UNKNOWN")
            confidence = round(pred.get("confidence", 0) * 100, 1)
            review_text = pred.get("text", st.session_state.bulk_reviews[i] if i < len(st.session_state.bulk_reviews) else "")
            _, badge_class = get_style_class(sentiment, confidence)
            color = get_progress_color(sentiment, confidence)
            emoji = "😊" if sentiment == "POSITIVE" else "😞"

            with st.expander(f"Review #{i+1} — {emoji} {sentiment} ({confidence}%)"):
                st.markdown(f"""
                <div style="margin-bottom:8px;">
                    <span class="{badge_class}">{sentiment}</span>
                    <span style="font-size:0.85rem; color:#6c757d; margin-left:10px;">Confidence: {confidence}%</span>
                </div>
                <div style="height:8px; background:#e9ecef; border-radius:8px; overflow:hidden; margin-bottom:10px;">
                    <div style="height:100%; width:{confidence}%; background:{color}; border-radius:8px;"></div>
                </div>
                <div style="font-size:0.9rem; color:#333; line-height:1.6;">{review_text}</div>
                """, unsafe_allow_html=True)
    elif not st.session_state.bulk_reviews:
        st.info("Add some reviews above to get started.")