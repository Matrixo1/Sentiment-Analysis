import re
import string
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging

logger = logging.getLogger(__name__)

# Initialize NLTK resources
def download_nltk_resources():
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4"
    }
    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info(f"Downloading NLTK resource: {name}...")
            nltk.download(name, quiet=True)

# Ensure resources are downloaded
download_nltk_resources()

# Load stop words and lemmatizer
STOP_WORDS = set(stopwords.words("english"))
# Custom additions to stopwords if necessary (e.g. high frequency noise words)
# We can keep it standard
LEMMATIZER = WordNetLemmatizer()

# Precompile regex for speed
HTML_TAG_RE = re.compile(r"<[^>]+>")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")

def clean_text(text):
    """
    Cleans raw review text through standard NLP pipeline:
    1. Remove HTML tags
    2. Lowercase text
    3. Remove punctuation and numbers
    4. Tokenize
    5. Remove stopwords
    6. Lemmatize words
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Remove HTML tags
    # BeautifulSoup is safer, but regex is faster. We use BeautifulSoup for robust parsing,
    # falling back to regex if BeautifulSoup fails.
    try:
        text = BeautifulSoup(text, "html.parser").get_text()
    except Exception:
        text = HTML_TAG_RE.sub(" ", text)
    
    # 2. Lowercase text
    text = text.lower()
    
    # 3. Remove punctuation and numbers
    text = NON_ALPHA_RE.sub(" ", text)
    
    # 4. Tokenize
    tokens = word_tokenize(text)
    
    # 5. Remove stopwords & 6. Lemmatize
    cleaned_tokens = [LEMMATIZER.lemmatize(token) for token in tokens if token not in STOP_WORDS]
    
    # Reconstruct text
    return " ".join(cleaned_tokens)

def preprocess_series(series, show_progress=False):
    """
    Preprocess a pandas Series of text reviews.
    """
    if show_progress:
        from tqdm import tqdm
        tqdm.pandas()
        return series.progress_apply(clean_text)
    else:
        return series.apply(clean_text)

if __name__ == "__main__":
    test_review = "<h3>Excellent Movie!</h3> I loved it 10/10. The actors were amazing, and the story was beautiful. Worth watching again!"
    print("Original:", test_review)
    print("Cleaned: ", clean_text(test_review))
