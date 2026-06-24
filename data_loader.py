import os
import urllib.request
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(current_dir, "data")):
    DATA_DIR = os.path.join(current_dir, "data")
else:
    DATA_DIR = os.path.abspath(os.path.join(current_dir, "../data"))
os.makedirs(DATA_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, "IMDB_Dataset.csv")

def load_imdb_dataset():
    """
    Loads the IMDB dataset of 50,000 movie reviews.
    1. Checks if cached locally at data/IMDB_Dataset.csv.
    2. If not cached, attempts download from a reliable github mirror.
    3. If download fails, attempts download via HuggingFace datasets library.
    4. If all else fails, falls back to NLTK movie_reviews dataset.
    
    Returns:
        train_df (pd.DataFrame): Dataframe with 'text' and 'label' columns.
        test_df (pd.DataFrame): Dataframe with 'text' and 'label' columns.
    """
    if os.path.exists(CSV_PATH):
        logger.info(f"Loading IMDB dataset from local cache: {CSV_PATH}")
        return load_csv_dataset(CSV_PATH)
        
    # Attempt 1: Download from Github Mirror
    url = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
    try:
        logger.info(f"Downloading IMDB Dataset (50,000 reviews) from {url}...")
        
        # Download with a simple progress reporter or status log
        urllib.request.urlretrieve(url, CSV_PATH)
        logger.info(f"Successfully downloaded and cached IMDB dataset to {CSV_PATH}")
        return load_csv_dataset(CSV_PATH)
        
    except Exception as e:
        logger.warning(f"Failed to download from Github Mirror: {e}")
        
        # Attempt 2: HuggingFace datasets
        try:
            logger.info("Attempting load via HuggingFace datasets...")
            from datasets import load_dataset
            dataset = load_dataset("imdb")
            train_df = pd.DataFrame(dataset["train"])
            test_df = pd.DataFrame(dataset["test"])
            
            # Cache it to CSV for future fast loads
            combined = pd.concat([train_df, test_df], ignore_index=True)
            # Revert mapping to match CSV format: label 1 -> positive, 0 -> negative
            combined["sentiment"] = combined["label"].map({1: "positive", 0: "negative"})
            combined = combined.rename(columns={"text": "review"})
            combined[["review", "sentiment"]].to_csv(CSV_PATH, index=False)
            
            logger.info(f"Loaded from HuggingFace and cached to {CSV_PATH}")
            return train_df, test_df
        except Exception as hf_err:
            logger.warning(f"HuggingFace datasets load failed: {hf_err}")
            
            # Attempt 3: NLTK Fallback
            return load_nltk_fallback()

def load_csv_dataset(csv_path):
    """
    Loads and processes the raw IMDB CSV.
    Splits into 25,000 train and 25,000 test to match the official dataset layout.
    """
    df = pd.read_csv(csv_path)
    # Map sentiment column to binary labels
    df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})
    df = df.rename(columns={"review": "text"})
    # Drop rows with nulls
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    
    # Standard IMDB dataset splits 25,000 / 25,000
    # Let's shuffle and split
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df = df_shuffled.iloc[:25000].reset_index(drop=True)
    test_df = df_shuffled.iloc[25000:].reset_index(drop=True)
    
    logger.info(f"Loaded CSV dataset. Train size: {len(train_df)}, Test size: {len(test_df)}")
    return train_df, test_df

def load_nltk_fallback():
    """
    Fallback loader using NLTK's movie reviews corpus.
    """
    logger.info("Loading NLTK movie_reviews corpus as fallback...")
    import nltk
    from nltk.corpus import movie_reviews
    
    # Download if not present
    try:
        movie_reviews.categories()
    except LookupError:
        nltk.download("movie_reviews")
    
    documents = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            text = " ".join(movie_reviews.words(fileid))
            label = 1 if category == "pos" else 0
            documents.append({"text": text, "label": label})
            
    df = pd.DataFrame(documents)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    logger.info(f"Loaded NLTK fallback dataset. Train: {len(train_df)}, Test: {len(test_df)}")
    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = load_imdb_dataset()
    print("Train dataset shape:", train_df.shape)
    print("Test dataset shape:", test_df.shape)
