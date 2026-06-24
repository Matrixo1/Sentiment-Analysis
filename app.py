import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup

# Add parent directory to path to ensure proper imports when running streamlit
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import clean_text, HTML_TAG_RE, NON_ALPHA_RE
from data_loader import load_imdb_dataset

# Set Page Config
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design & Rich Aesthetics
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;500;700&display=swap');
    
    /* Global font styles */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Custom main title styling with gradient */
    .main-title {
        background: linear-gradient(135deg, #FF3366 0%, #FF9933 50%, #9933FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 5px;
        text-align: center;
    }
    .subtitle {
        color: #8892b0;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 40px;
        font-weight: 300;
    }
    
    /* Glassmorphism Card Style */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        margin-bottom: 20px;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.15);
    }
    
    /* Glow Badges */
    .badge-positive {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 4px 14px 0 rgba(16, 185, 129, 0.4);
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .badge-negative {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 4px 14px 0 rgba(239, 68, 68, 0.4);
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sidebar styling */
    .stSidebar {
        background-color: #0b0c10;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Highlight tokens */
    .token-pos {
        background-color: rgba(16, 185, 129, 0.25);
        color: #34D399;
        border-bottom: 2px solid #10B981;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        display: inline-block;
        margin: 2px;
    }
    .token-neg {
        background-color: rgba(239, 68, 68, 0.25);
        color: #F87171;
        border-bottom: 2px solid #EF4444;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        display: inline-block;
        margin: 2px;
    }
    .token-neutral {
        background-color: rgba(255, 255, 255, 0.05);
        color: #E2E8F0;
        padding: 2px 6px;
        border-radius: 4px;
        display: inline-block;
        margin: 2px;
    }
    
    /* Text explanation flow steps */
    .step-box {
        border-left: 4px solid #FF9933;
        padding-left: 15px;
        margin-bottom: 15px;
        background: rgba(255, 255, 255, 0.01);
        padding-top: 5px;
        padding-bottom: 5px;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(current_dir, "artifacts")):
    ARTIFACTS_DIR = os.path.join(current_dir, "artifacts")
else:
    ARTIFACTS_DIR = os.path.abspath(os.path.join(current_dir, "../artifacts"))

# Cache model and vectorizer loading
@st.cache_resource
def load_model_assets():
    model_path = os.path.join(ARTIFACTS_DIR, "best_model.joblib")
    vectorizer_path = os.path.join(ARTIFACTS_DIR, "vectorizer.joblib")
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    return None, None

@st.cache_data
def load_metrics():
    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None

@st.cache_data
def get_dataset_stats():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(current_dir, "data/IMDB_Dataset.csv")):
        csv_path = os.path.join(current_dir, "data/IMDB_Dataset.csv")
    else:
        csv_path = os.path.abspath(os.path.join(current_dir, "../data/IMDB_Dataset.csv"))
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            total_rows = len(df)
            sentiments = df["sentiment"].value_counts().to_dict()
            word_counts = df["review"].str.split().str.len()
            return {
                "total": total_rows,
                "pos": sentiments.get("positive", 0),
                "neg": sentiments.get("negative", 0),
                "avg_words": word_counts.mean(),
                "max_words": word_counts.max(),
                "min_words": word_counts.min()
            }
        except Exception:
            return None
    return None

# Load model assets
model, vectorizer = load_model_assets()
metrics = load_metrics()
dataset_stats = get_dataset_stats()

# Sidebar Setup
st.sidebar.markdown("<h2 style='text-align: center;'>🎬 Pipeline Controls</h2>", unsafe_allow_html=True)
st.sidebar.write("---")

if model and vectorizer:
    # Model Metadata
    metadata_path = os.path.join(ARTIFACTS_DIR, "model_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        st.sidebar.success(f"⚡ Loaded active model: **{meta['model_name']}**")
        st.sidebar.info(f"🎯 Validation Accuracy: **{meta['accuracy']*100:.2f}%**")
    else:
        st.sidebar.success("⚡ Loaded model and vectorizer successfully.")
else:
    st.sidebar.warning("⚠️ No pre-trained model found in artifacts. Please run the training pipeline first.")
    
# Sample review templates for user ease of testing
st.sidebar.markdown("### 💡 Sample Reviews")
sample_pos = "Absolutely wonderful movie! The performances were brilliant and the direction was top-notch. I highly recommend watching this masterpiece."
sample_neg = "This was a complete waste of time. The plot was thin, the characters were annoying, and the ending made absolutely no sense. Avoid at all costs."

if st.sidebar.button("Load Positive Sample"):
    st.session_state["review_input"] = sample_pos
if st.sidebar.button("Load Negative Sample"):
    st.session_state["review_input"] = sample_neg

# Main Title Header
st.markdown("<h1 class='main-title'>Movie Review Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>A high-fidelity sentiment analysis pipeline built with Scikit-Learn & Natural Language Processing</div>", unsafe_allow_html=True)

# Main Navigation Tabs
tab_analyze, tab_compare, tab_explanation = st.tabs([
    "🌟 Analyze Sentiment",
    "📊 Model Comparisons",
    "⚙️ Under the Hood & Preprocessing"
])

# --- TAB 1: ANALYZE SENTIMENT ---
with tab_analyze:
    st.markdown("### ✍️ Test Movie Review Sentiment")
    
    # Input Area
    review_text = st.text_area(
        "Paste or type a movie review below:",
        value=st.session_state.get("review_input", ""),
        height=150,
        placeholder="Enter your review here..."
    )
    
    col1, col2 = st.columns([3, 1])
    
    analyze_clicked = st.button("🔮 Analyze Sentiment", type="primary")
    
    if review_text or analyze_clicked:
        if not model or not vectorizer:
            st.error("Please run the training pipeline or ensure the models are exported to `/artifacts`.")
        elif not review_text.strip():
            st.warning("Please type a review to analyze.")
        else:
            # 1. Pipeline execution
            cleaned = clean_text(review_text)
            features = vectorizer.transform([cleaned])
            
            # Predict
            pred = model.predict(features)[0]
            
            # Confidence/Probability estimation
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[0]
                confidence = probs[pred]
            elif hasattr(model, "decision_function"):
                dec = model.decision_function(features)[0]
                # Convert decision function score to pseudo-probability via sigmoid
                confidence = 1 / (1 + np.exp(-abs(dec)))
            else:
                confidence = 1.0 # Fallback
                
            sentiment_str = "Positive" if pred == 1 else "Negative"
            badge_class = "badge-positive" if pred == 1 else "badge-negative"
            
            # Display Results Card
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            col_res1, col_res2 = st.columns([1, 1])
            
            with col_res1:
                st.markdown(f"<h4>Prediction Sentiment:</h4> <span class='{badge_class}'>{sentiment_str}</span>", unsafe_allow_html=True)
                st.write("")
                st.markdown(f"<h4>Confidence Level:</h4> **{confidence*100:.2f}%**", unsafe_allow_html=True)
                st.progress(float(confidence))
                
            with col_res2:
                # Highlight words in the text based on model coefficients to show "Why"
                st.markdown("<h4>Feature Importance Highlights:</h4>", unsafe_allow_html=True)
                
                # Retrieve coefficients from Logistic Regression or SVM
                has_coef = hasattr(model, "coef_")
                if has_coef:
                    # Get coefficients and mapping
                    coef = model.coef_[0]
                    vocab = vectorizer.vocabulary_
                    inv_vocab = {v: k for k, v in vocab.items()}
                    
                    tokens = cleaned.split()
                    highlighted_html = []
                    
                    for token in tokens:
                        if token in vocab:
                            idx = vocab[token]
                            weight = coef[idx]
                            
                            if weight > 0.2: # Strong positive
                                highlighted_html.append(f"<span class='token-pos'>{token}</span>")
                            elif weight < -0.2: # Strong negative
                                highlighted_html.append(f"<span class='token-neg'>{token}</span>")
                            else:
                                highlighted_html.append(f"<span class='token-neutral'>{token}</span>")
                        else:
                            highlighted_html.append(f"<span class='token-neutral'>{token}</span>")
                            
                    st.markdown("<div style='line-height: 2.2;'>" + " ".join(highlighted_html) + "</div>", unsafe_allow_html=True)
                else:
                    st.write("Coefficient details not available for this model type.")
            
            st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 2: MODEL COMPARISONS ---
with tab_compare:
    st.markdown("### 📊 Pipeline Benchmark Results")
    
    if metrics:
        # Create metrics comparison DataFrame
        metrics_df = pd.DataFrame(metrics).T * 100
        
        # Display summary table
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4>Accuracy & Quality Comparison (percentage)</h4>", unsafe_allow_html=True)
        st.dataframe(metrics_df.style.format("{:.2f}%").background_gradient(cmap="Blues", axis=0))
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Plotly Bar Chart Comparison
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4>Metric Chart Comparison</h4>", unsafe_allow_html=True)
        st.bar_chart(metrics_df[["accuracy", "f1_score", "roc_auc"]])
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display ROC Curve & Confusion Matrix images from artifacts
        col_img1, col_img2 = st.columns([1, 1])
        
        with col_img1:
            roc_path = os.path.join(ARTIFACTS_DIR, "roc_curve_comparison.png")
            if os.path.exists(roc_path):
                st.markdown("<h4>ROC-AUC Curves</h4>", unsafe_allow_html=True)
                st.image(roc_path, use_column_width=True)
            else:
                st.info("ROC Curve image not available yet. Complete training to generate.")
                
        with col_img2:
            st.markdown("<h4>Confusion Matrices</h4>", unsafe_allow_html=True)
            # Find all files ending in _cm.png
            cm_files = [f for f in os.listdir(ARTIFACTS_DIR) if f.endswith("_cm.png")]
            if cm_files:
                selected_cm = st.selectbox("Select Model Confusion Matrix:", cm_files)
                st.image(os.path.join(ARTIFACTS_DIR, selected_cm), use_column_width=True)
            else:
                st.info("Confusion matrix images not found.")
    else:
        st.info("No benchmark metrics found. Run the training script first to populate metrics.")

# --- TAB 3: UNDER THE HOOD & PREPROCESSING ---
with tab_explanation:
    st.markdown("### 📊 Dataset Overview & Statistics")
    if dataset_stats:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Total Reviews", f"{dataset_stats['total']:,}")
        col_s2.metric("Class Balance", f"🟢 Positive: {dataset_stats['pos']:,} / 🔴 Negative: {dataset_stats['neg']:,}")
        col_s3.metric("Avg Review Length", f"{dataset_stats['avg_words']:.1f} words")
        col_s4.metric("Review Length Range", f"{dataset_stats['min_words']} - {dataset_stats['max_words']} words")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Dataset statistics not available (run pipeline to cache the dataset).")

    st.markdown("### 🛠️ NLP Preprocessing Flow Visualization")
    st.write("Understand the sequence of operations applied to convert a raw review into mathematical inputs for classification.")
    
    sample_explain_text = "<h3>Fantastic!</h3> I enjoyed it. 10/10. Not boring at all!"
    
    st.markdown("#### Try the Preprocessing pipeline on this custom text:")
    custom_explain_text = st.text_input("Test string:", value=sample_explain_text)
    
    # Run step-by-step
    # Step 1: Raw HTML Clean
    try:
        html_cleaned = BeautifulSoup(custom_explain_text, "html.parser").get_text()
    except Exception:
        html_cleaned = HTML_TAG_RE.sub(" ", custom_explain_text)
        
    # Step 2: Lowercased & Non-alpha remove
    lowercase_cleaned = html_cleaned.lower()
    alpha_cleaned = NON_ALPHA_RE.sub(" ", lowercase_cleaned)
    
    # Step 3: Tokenized
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(alpha_cleaned)
    
    # Step 4: Stopwords removed
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words("english"))
    no_stopwords = [t for t in tokens if t not in stop_words]
    
    # Step 5: Lemmatized
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(t) for t in no_stopwords]
    
    # Render Steps in UI
    st.markdown("<div class='step-box'>", unsafe_allow_html=True)
    st.markdown("💡 **Step 1: HTML Tag Stripping**")
    st.code(html_cleaned)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='step-box'>", unsafe_allow_html=True)
    st.markdown("💡 **Step 2: Lowercasing & Non-Alphabetic Character Removal**")
    st.code(alpha_cleaned)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='step-box'>", unsafe_allow_html=True)
    st.markdown("💡 **Step 3: Tokenization**")
    st.write(tokens)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='step-box'>", unsafe_allow_html=True)
    st.markdown("💡 **Step 4: Stopwords Removal**")
    st.write(no_stopwords)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='step-box'>", unsafe_allow_html=True)
    st.markdown("💡 **Step 5: Lemmatization**")
    st.write(lemmas)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='step-box'>", unsafe_allow_html=True)
    st.markdown("💡 **Final Preprocessed Text (Ready for Vectorizer)**")
    st.code(" ".join(lemmas))
    st.markdown("</div>", unsafe_allow_html=True)
