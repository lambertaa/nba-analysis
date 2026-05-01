"""
NBA Draft NLP Model with MLflow Tracking

This script:
1. Auto-retrieves new All-Star data
2. Incrementally scrapes only new draft prospect data
3. Trains the NLP model with MLflow tracking for version control and experiment management

Execute this code block by block in a Jupyter notebook.
"""

# =============================================================================
# BLOCK 1: Setup and Imports
# =============================================================================

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from IPython.display import clear_output

# MLflow imports
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Sklearn imports
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy.sparse import hstack

# NLP imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import pickle

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# Add data_retrieval to path
sys.path.append(os.path.abspath("../data_retrieval"))
from realgm_retr import get_realgm_allstar_rosters

# Download NLTK data
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

# Set date string for file naming
date_string = datetime.now().strftime("%Y%m%d")

print(f"Setup complete. Date string: {date_string}")


# =============================================================================
# BLOCK 2: MLflow Configuration (SQLite Database)
# =============================================================================

# Set MLflow tracking URI to SQLite database
# IMPORTANT: Use forward slashes for SQLite URI, even on Windows
mlflow_db_path = os.path.abspath("../data/mlflow_tracking.db")
# Convert Windows backslashes to forward slashes for SQLite URI
mlflow_db_uri = mlflow_db_path.replace("\\", "/")
mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_uri}")

# Set artifact location to local directory
artifact_location = os.path.abspath("../mlruns")

# Set experiment name
experiment_name = "NBA_Draft_AllStar_Prediction"
mlflow.set_experiment(experiment_name)

print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
print(f"MLflow experiment: {experiment_name}")
print(f"Artifact location: {artifact_location}")


# =============================================================================
# BLOCK 3: Auto-Retrieve All-Star Data
# =============================================================================


def retrieve_allstar_data(start_year=1951, end_year=None):
    """
    Retrieve All-Star roster data from RealGM.
    Only retrieves new data if needed.
    """
    if end_year is None:
        end_year = datetime.now().year + 1

    allstar_file = f"../data/nba_allstar_all_{date_string}.csv"

    # Check if today's file already exists
    if os.path.exists(allstar_file):
        print(f"All-Star data for {date_string} already exists. Loading from file...")
        df = pd.read_csv(allstar_file)
        return df

    # Check for most recent file
    data_dir = "../data"
    allstar_files = [
        f
        for f in os.listdir(data_dir)
        if f.startswith("nba_allstar_all_") and f.endswith(".csv")
    ]

    if allstar_files:
        # Get most recent file
        most_recent = sorted(allstar_files)[-1]
        most_recent_path = os.path.join(data_dir, most_recent)
        existing_df = pd.read_csv(most_recent_path)
        max_existing_year = existing_df["Year"].max()

        # Only retrieve new years
        if max_existing_year >= end_year - 1:
            print(f"Using existing All-Star data from {most_recent}")
            existing_df.to_csv(allstar_file, index=False)
            return existing_df

        print(f"Retrieving All-Star data from {max_existing_year + 1} to {end_year}...")
        new_years = list(range(max_existing_year + 1, end_year))
        new_df = get_realgm_allstar_rosters(new_years)

        # Combine with existing data
        df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        print(f"Retrieving All-Star data from {start_year} to {end_year}...")
        years = list(range(start_year, end_year))
        df = get_realgm_allstar_rosters(years)

    # Save to file
    df.to_csv(allstar_file, index=False)
    print(f"All-Star data saved to {allstar_file}")

    return df


# Retrieve All-Star data
allstar_df = retrieve_allstar_data()
print(f"\nAll-Star data shape: {allstar_df.shape}")
print(f'Years covered: {allstar_df["Year"].min()} - {allstar_df["Year"].max()}')


# =============================================================================
# BLOCK 4: Incremental Draft Prospect Scraping - Get Player Links
# =============================================================================


def get_existing_player_links():
    """
    Load existing player links from the most recent file.
    """
    data_dir = "../data"
    link_files = [
        f
        for f in os.listdir(data_dir)
        if f.startswith("nbadraft_player_links_") and f.endswith(".txt")
    ]

    if not link_files:
        return set()

    most_recent = sorted(link_files)[-1]
    file_path = os.path.join(data_dir, most_recent)

    with open(file_path, "r") as file:
        links = set(line.strip() for line in file.readlines())

    print(f"Loaded {len(links)} existing player links from {most_recent}")
    return links


def scrape_player_links(max_pages=391):
    """
    Scrape player links from nbadraft.net.
    Only scrapes new links not in existing files.
    """
    base_url = "https://www.nbadraft.net/players/"
    existing_links = get_existing_player_links()
    new_links = []

    print(f"Scraping player links from nbadraft.net...")

    for page_num in range(1, max_pages + 1):
        if page_num == 1:
            page_url = base_url
        else:
            page_url = f"{base_url}page/{page_num}/"

        try:
            response = requests.get(page_url, timeout=10)

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                parent_div = soup.find("div", class_="wf-container")

                if parent_div:
                    links = parent_div.find_all("a")
                    page_new_links = 0

                    for link in links:
                        href = link.get("href", "")
                        if base_url in href and href not in existing_links:
                            new_links.append(href)
                            page_new_links += 1

                    clear_output(wait=True)
                    print(
                        f"Page {page_num}/{max_pages}: Found {page_new_links} new links (Total new: {len(new_links)})"
                    )
            else:
                print(
                    f"Failed to retrieve page {page_num}. Status code: {response.status_code}"
                )
        except Exception as e:
            print(f"Error on page {page_num}: {str(e)}")
            continue

    # Combine with existing links and save
    all_links = list(existing_links) + new_links
    all_links = list(set(all_links))  # Remove duplicates

    file_path = f"../data/nbadraft_player_links_{date_string}.txt"
    with open(file_path, "w") as file:
        for link in all_links:
            file.write(link + "\n")

    print(f"\nSaved {len(all_links)} total player links to {file_path}")
    print(f"New links found: {len(new_links)}")

    return all_links, new_links


# Scrape player links
all_player_links, new_player_links = scrape_player_links()


# =============================================================================
# BLOCK 5: Incremental Draft Prospect Scraping - Get Player Data
# =============================================================================


def get_existing_scraped_players():
    """
    Load existing scraped player data to avoid re-scraping.
    """
    data_dir = "../data"
    data_files = [
        f
        for f in os.listdir(data_dir)
        if f.startswith("nbadraft_strengths_weaknesses_") and f.endswith(".csv")
    ]

    if not data_files:
        return pd.DataFrame(), set()

    most_recent = sorted(data_files)[-1]
    file_path = os.path.join(data_dir, most_recent)
    df = pd.read_csv(file_path)

    existing_players = set(df["player"].values)
    print(f"Loaded {len(existing_players)} existing players from {most_recent}")

    return df, existing_players


def scrape_single_player(player_link):
    """
    Scrape data for a single player.
    Returns tuple: (player_data_dict or None, player_name, success_status)
    """
    player_name = player_link.rsplit("/", 2)[-2]

    try:
        response = requests.get(player_link, timeout=10)

        if response.status_code != 200:
            return None, player_name, "failed"

        soup = BeautifulSoup(response.content, "html.parser")
        weak_obj = None
        strength_obj = None

        # Find strengths and weaknesses
        for identify in ["b", "strong"]:
            for text in ["Strengths:", "Strengths: "]:
                strength_obj = soup.find(identify, text=text)
                if strength_obj is not None:
                    break
            if strength_obj is None:
                continue

            for text in ["Weaknesses:", "Weakness:", "Weaknesses: "]:
                weak_obj = soup.find(identify, text=text)
                if weak_obj is not None:
                    break

            if (strength_obj is not None) and (weak_obj is not None):
                break

        if (weak_obj is None) or (strength_obj is None):
            return None, player_name, "missing_data"

        # Extract text
        try:
            strength_text = strength_obj.find_next_sibling(text=True).strip()
            weak_text = weak_obj.find_next_sibling(text=True).strip()
        except AttributeError:
            try:
                strength_text = strength_obj.find_next_sibling().text.strip()
                weak_text = weak_obj.find_next_sibling().text.strip()
            except AttributeError:
                return None, player_name, "missing_data"

        # Extract year
        try:
            year = int(
                soup.find("div", class_="mock-year")
                .find("span", class_="label")
                .text[:4]
            )
        except AttributeError:
            year = np.nan

        # Extract overall rating
        try:
            overall = int(
                soup.find("div", class_="overall").find("span", class_="value").text
            )
        except AttributeError:
            return None, player_name, "missing_data"

        # Extract attribute scores
        player_attr_obj = soup.find("div", class_="player-attributes")
        attr_values = player_attr_obj.find_all(
            "div", class_="div-table-cell attribute-value"
        )
        attr_names = player_attr_obj.find_all(
            "div", class_="div-table-cell attribute-name"
        )
        attr_dict = {}

        try:
            for name, value in zip(attr_names, attr_values):
                attr_dict[name.text.replace(" ", "")] = int(value.text)
        except ValueError:
            return None, player_name, "missing_data"

        # Build the data dictionary
        player_data = {
            "player": player_name,
            "draft_year": year,
            "strengths": strength_text,
            "weaknesses": weak_text,
            "overall": overall,
        }
        player_data.update(attr_dict)

        return player_data, player_name, "success"

    except Exception as e:
        return None, player_name, "error"


def scrape_player_data(player_links, existing_players=None, n_workers=4):
    """
    Scrape player strengths/weaknesses data in parallel.
    Only scrapes players not in existing_players set.

    Parameters:
    -----------
    player_links : list
        List of player URLs to scrape
    existing_players : set, optional
        Set of player names already scraped (to skip)
    n_workers : int, default=4
        Number of parallel workers for scraping
        Set to 1 for sequential processing

    Returns:
    --------
    df : pd.DataFrame
        DataFrame with scraped player data
    missing_players : list
        List of player names that failed to scrape
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock

    if existing_players is None:
        existing_players = set()

    # Filter out already scraped players
    links_to_scrape = []
    skipped_count = 0

    for link in player_links:
        player_name = link.rsplit("/", 2)[-2]
        if player_name in existing_players:
            skipped_count += 1
        else:
            links_to_scrape.append(link)

    print(f"\nScraping player data:")
    print(f"  Total links: {len(player_links)}")
    print(f"  Already in database (skipped): {skipped_count}")
    print(f"  To scrape: {len(links_to_scrape)}")
    print(f"  Workers: {n_workers}")

    if len(links_to_scrape) == 0:
        print("No new players to scrape!")
        return pd.DataFrame(), []

    data = []
    missing_players = []
    lock = Lock()
    completed = 0

    # Use ThreadPoolExecutor for parallel scraping
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_link = {
            executor.submit(scrape_single_player, link): link
            for link in links_to_scrape
        }

        # Process completed tasks
        for future in as_completed(future_to_link):
            player_data, player_name, status = future.result()

            with lock:
                completed += 1

                if status == "success" and player_data is not None:
                    data.append(player_data)
                else:
                    missing_players.append(player_name)

                # Progress update
                if completed % 10 == 0 or completed == len(links_to_scrape):
                    clear_output(wait=True)
                    print(
                        f"Progress: {completed}/{len(links_to_scrape)} | "
                        f"Scraped: {len(data)} | "
                        f"Missing/Failed: {len(missing_players)}"
                    )

    print(f"\nScraping complete!")
    print(f"Successfully scraped: {len(data)} players")
    print(f"Skipped (already in database): {skipped_count} players")
    print(f"Missing/Failed: {len(missing_players)} players")

    return pd.DataFrame(data), missing_players


# Get existing scraped data
existing_df, existing_players = get_existing_scraped_players()

# Scrape only new players (with parallel processing)
# Adjust n_workers based on your needs: higher = faster but more load on the server
# Recommended: 4-8 workers for good balance
new_df, missing_players = scrape_player_data(
    all_player_links, existing_players, n_workers=8
)

# Combine with existing data
if not existing_df.empty and not new_df.empty:
    draftnet_df = pd.concat([existing_df, new_df], ignore_index=True)
elif not new_df.empty:
    draftnet_df = new_df
else:
    draftnet_df = existing_df

# Save combined data
draftnet_file = f"../data/nbadraft_strengths_weaknesses_{date_string}.csv"
draftnet_df.to_csv(draftnet_file, index=False)
print(f"\nSaved {len(draftnet_df)} total players to {draftnet_file}")


# =============================================================================
# BLOCK 6: Load and Merge Draft Data
# =============================================================================

# Load all-time draft data
alltimedraft_file = "../data/nba_draft_all_time.csv"
alltimedraft_df = pd.read_csv(alltimedraft_file)

# Merge draft data with scouting reports
keep_cols = [
    "PERSON_ID",
    "PLAYER_NAME",
    "SEASON",
    "strengths",
    "weaknesses",
    "overall",
    "Athleticism",
    "Size",
    "Defense",
    "Strength",
    "Quickness",
    "Leadership",
    "JumpShot",
    "NBAReady",
    "Rebounding",
    "Potential",
    "PostSkills",
    "Intangibles",
    "BallHandling",
    "Passing",
]

rename_dict = {"PERSON_ID": "person_id", "PLAYER_NAME": "player", "SEASON": "season"}

df = pd.merge(alltimedraft_df, draftnet_df, on="player", how="inner")[keep_cols]
df.rename(rename_dict, axis=1, inplace=True)
df = df.loc[df.season >= 2006]
df = df.dropna(subset=["strengths"])

print(f"\nMerged draft data shape: {df.shape}")
print(f'Seasons covered: {df["season"].min()} - {df["season"].max()}')


# =============================================================================
# BLOCK 7: Add All-Star Labels
# =============================================================================

# Filter All-Star data to relevant years
allstar_df = allstar_df.loc[allstar_df.Year >= 2001]

# Create All-Star indicators
all_star_indicator = []
all_star_first_year = []

for index, row in df.iterrows():
    if row["player"] in allstar_df["Player"].values:
        all_star_indicator.append(1)
        player_allstar = allstar_df.loc[allstar_df["Player"] == row["player"]]
        year_min = player_allstar.Year.min()
        allstar_player_year = year_min - row["season"]
        all_star_first_year.append(allstar_player_year)
    else:
        all_star_indicator.append(0)
        all_star_first_year.append(np.nan)

df["allstar_bool"] = all_star_indicator
df["allstar_first_year"] = all_star_first_year
df["within7"] = (df.allstar_first_year <= 7).astype(int)
df["within5"] = (df.allstar_first_year <= 5).astype(int)

# Remove duplicates
df.drop_duplicates(subset="player", inplace=True)

print(f"\nAll-Star labeling complete:")
print(f"Total players: {len(df)}")
print(f'All-Stars: {df["allstar_bool"].sum()}')
print(f'All-Stars within 5 years: {df["within5"].sum()}')
print(f'All-Stars within 7 years: {df["within7"].sum()}')

# =============================================================================
# Filter out players who haven't had enough time to become All-Stars
# =============================================================================

# Calculate current year (for determining if players have had enough time)
current_year = datetime.now().year

# Calculate years since draft
df["years_since_draft"] = current_year - df["season"]

# Define the threshold (7 years for within7 target)
YEARS_THRESHOLD = 7

print(f"\nFiltering players based on time in league:")
print(f"Current year: {current_year}")
print(f"Threshold: {YEARS_THRESHOLD} years")
print(f"Players before filtering: {len(df)}")

# Keep players who:
# 1. Have been in the league for at least YEARS_THRESHOLD years, OR
# 2. Already became an All-Star (regardless of years in league)
df_filtered = df[
    (df["years_since_draft"] >= YEARS_THRESHOLD) | (df["allstar_bool"] == 1)
].copy()

print(f"Players after filtering: {len(df_filtered)}")
print(f"Players removed (not enough time, not All-Stars): {len(df) - len(df_filtered)}")
print(f'All-Stars in filtered data: {df_filtered["allstar_bool"].sum()}')
print(f'All-Stars within 7 years in filtered data: {df_filtered["within7"].sum()}')

# Update df to the filtered version
df = df_filtered


# =============================================================================
# BLOCK 8: NLP Text Preprocessing
# =============================================================================


def preprocess_text(text):
    """
    Preprocess text for NLP:
    - Lowercasing
    - Tokenization
    - Remove punctuation
    - Remove stop words
    - Lemmatization
    """
    # Lowercasing
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Removing Punctuation
    tokens = [token for token in tokens if token not in string.punctuation]

    # Removing Stop Words
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Joining tokens back into a single text
    preprocessed_text = " ".join(tokens)

    return preprocessed_text


# Apply preprocessing
print("\nPreprocessing text data...")
df["strengths_processed"] = df["strengths"].apply(preprocess_text)
df["weaknesses_processed"] = df["weaknesses"].apply(preprocess_text)
df["combined_text"] = df["strengths_processed"] + " " + df["weaknesses_processed"]

print("Text preprocessing complete!")


# =============================================================================
# BLOCK 8B: NLP Experiment Configuration
# =============================================================================

# Configure which NLP experiment to run
# Options: 'baseline', 'separate_vectorizers', 'separate_with_trigrams'
NLP_EXPERIMENT = "separate_with_trigrams"  # Change this to switch experiments

print(f"\n{'='*80}")
print(f"NLP EXPERIMENT: {NLP_EXPERIMENT}")
print(f"{'='*80}\n")

if NLP_EXPERIMENT == "baseline":
    print("Using baseline approach: Single vectorizer, bigrams only")
    USE_SEPARATE_VECTORIZERS = False
    NGRAM_RANGE = (1, 2)
    MAX_FEATURES_PER_FIELD = 500
elif NLP_EXPERIMENT == "separate_vectorizers":
    print("Using Experiment 1: Separate vectorizers for strengths/weaknesses, bigrams")
    USE_SEPARATE_VECTORIZERS = True
    NGRAM_RANGE = (1, 2)
    MAX_FEATURES_PER_FIELD = 400  # 400 per field = 800 total text features
elif NLP_EXPERIMENT == "separate_with_trigrams":
    print("Using Experiment 2: Separate vectorizers with trigrams")
    USE_SEPARATE_VECTORIZERS = True
    NGRAM_RANGE = (1, 3)
    MAX_FEATURES_PER_FIELD = 600  # 600 per field = 1200 total text features
else:
    raise ValueError(f"Unknown NLP_EXPERIMENT: {NLP_EXPERIMENT}")


# =============================================================================
# BLOCK 9: Prepare Features and Target
# =============================================================================

# Define target variable (All-Star within 7 years)
target_col = "within7"

# Separate training data (has labels) from prediction data (2026 prospects)
# Note: 2026 prospects won't be in the merged data anyway (inner join with alltimedraft_df)
# but we explicitly handle any unlabeled data here
train_df = df[df[target_col].notna()].copy()
predict_df = df[df[target_col].isna()].copy()

print(f"\nTraining data: {len(train_df)} players")
print(f"Prediction data (unlabeled 2026 prospects): {len(predict_df)} players")

# Prepare text features based on experiment configuration
if USE_SEPARATE_VECTORIZERS:
    X_strengths = train_df["strengths_processed"]
    X_weaknesses = train_df["weaknesses_processed"]
    print(f"\nUsing separate text features:")
    print(f"  Strengths: {len(X_strengths)} samples")
    print(f"  Weaknesses: {len(X_weaknesses)} samples")
else:
    X_text = train_df["combined_text"]
    print(f"\nUsing combined text features: {len(X_text)} samples")

y = train_df[target_col]

# Numerical features
numerical_features = [
    "overall",
    "Athleticism",
    "Size",
    "Defense",
    "Strength",
    "Quickness",
    "Leadership",
    "JumpShot",
    "NBAReady",
]

# Fill NaN values in numerical features with median
for col in numerical_features:
    if col in train_df.columns:
        median_val = train_df[col].median()
        train_df[col].fillna(median_val, inplace=True)

X_numerical = train_df[numerical_features]

print(f"Numerical features: {X_numerical.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")


# =============================================================================
# BLOCK 10: Train-Test Split
# =============================================================================

# Split data based on experiment configuration
if USE_SEPARATE_VECTORIZERS:
    (
        X_str_train,
        X_str_test,
        X_weak_train,
        X_weak_test,
        X_num_train,
        X_num_test,
        y_train,
        y_test,
    ) = train_test_split(
        X_strengths,
        X_weaknesses,
        X_numerical,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"\nTrain-test split complete:")
    print(f"Training samples: {len(X_str_train)}")
    print(f"Test samples: {len(X_str_test)}")
else:
    X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = (
        train_test_split(
            X_text, X_numerical, y, test_size=0.2, random_state=42, stratify=y
        )
    )
    print(f"\nTrain-test split complete:")
    print(f"Training samples: {len(X_text_train)}")
    print(f"Test samples: {len(X_text_test)}")

print(f"Training target distribution: {y_train.value_counts().to_dict()}")
print(f"Test target distribution: {y_test.value_counts().to_dict()}")


# =============================================================================
# BLOCK 11: MLflow Training with Hyperparameter Tuning
# =============================================================================

# Start MLflow run
with mlflow.start_run(run_name=f"NBA_Draft_NLP_{date_string}") as run:

    print(f'\n{"="*80}')
    print(f"MLflow Run ID: {run.info.run_id}")
    print(f'{"="*80}\n')

    # Log experiment configuration
    mlflow.log_param("nlp_experiment", NLP_EXPERIMENT)
    mlflow.log_param("use_separate_vectorizers", USE_SEPARATE_VECTORIZERS)
    mlflow.log_param("ngram_range", str(NGRAM_RANGE))
    mlflow.log_param("max_features_per_field", MAX_FEATURES_PER_FIELD)
    mlflow.log_param("target_variable", target_col)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_split", 0.2)
    mlflow.log_param("tfidf_min_df", 2)
    mlflow.log_param("tfidf_max_df", 0.8)

    # TF-IDF Vectorization based on experiment configuration
    print("Performing TF-IDF vectorization...")

    if USE_SEPARATE_VECTORIZERS:
        # Experiment 1 & 2: Separate vectorizers for strengths and weaknesses
        print(f"Using separate vectorizers with ngram_range={NGRAM_RANGE}")

        vectorizer_strengths = TfidfVectorizer(
            max_features=MAX_FEATURES_PER_FIELD,
            ngram_range=NGRAM_RANGE,
            min_df=2,
            max_df=0.8,
            sublinear_tf=True,
        )

        vectorizer_weaknesses = TfidfVectorizer(
            max_features=MAX_FEATURES_PER_FIELD,
            ngram_range=NGRAM_RANGE,
            min_df=2,
            max_df=0.8,
            sublinear_tf=True,
        )

        # Transform strengths
        X_str_train_tfidf = vectorizer_strengths.fit_transform(X_str_train)
        X_str_test_tfidf = vectorizer_strengths.transform(X_str_test)

        # Transform weaknesses
        X_weak_train_tfidf = vectorizer_weaknesses.fit_transform(X_weak_train)
        X_weak_test_tfidf = vectorizer_weaknesses.transform(X_weak_test)

        print(f"Strengths TF-IDF shape: {X_str_train_tfidf.shape}")
        print(f"Weaknesses TF-IDF shape: {X_weak_train_tfidf.shape}")

        # Combine all features: strengths + weaknesses + numerical
        X_train_combined = hstack(
            [X_str_train_tfidf, X_weak_train_tfidf, X_num_train.values]
        )
        X_test_combined = hstack(
            [X_str_test_tfidf, X_weak_test_tfidf, X_num_test.values]
        )

        total_text_features = X_str_train_tfidf.shape[1] + X_weak_train_tfidf.shape[1]
        mlflow.log_param("total_text_features", total_text_features)
        mlflow.log_param("strengths_features", X_str_train_tfidf.shape[1])
        mlflow.log_param("weaknesses_features", X_weak_train_tfidf.shape[1])
        mlflow.log_param("train_size", len(X_str_train))
        mlflow.log_param("test_size", len(X_str_test))

    else:
        # Baseline: Single vectorizer for combined text
        print(f"Using single vectorizer with ngram_range={NGRAM_RANGE}")

        vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES_PER_FIELD,
            ngram_range=NGRAM_RANGE,
            min_df=2,
            max_df=0.8,
        )

        X_text_train_tfidf = vectorizer.fit_transform(X_text_train)
        X_text_test_tfidf = vectorizer.transform(X_text_test)

        print(f"TF-IDF shape: {X_text_train_tfidf.shape}")

        # Combine text and numerical features
        X_train_combined = hstack([X_text_train_tfidf, X_num_train.values])
        X_test_combined = hstack([X_text_test_tfidf, X_num_test.values])

        mlflow.log_param("total_text_features", X_text_train_tfidf.shape[1])
        mlflow.log_param("train_size", len(X_text_train))
        mlflow.log_param("test_size", len(X_text_test))

    print(f"Combined feature shape: {X_train_combined.shape}")
    mlflow.log_param("total_features", X_train_combined.shape[1])
    mlflow.log_param("numerical_features", len(numerical_features))

    # Hyperparameter tuning with GridSearchCV
    print("\nPerforming hyperparameter tuning...")

    # Hyperparameter grid - optimized for small datasets
    param_grid = {
        "C": [0.1, 0.5, 1.0, 5.0, 10.0],  # Regularization strength
        "penalty": ["l2"],  # L2 works best with lbfgs for small datasets
        "solver": ["lbfgs"],  # Best for small datasets, guaranteed convergence
        "max_iter": [2000],  # Increased for convergence
        "class_weight": ["balanced", None],  # Try both balanced and unbalanced
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        LogisticRegression(random_state=42),
        param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train_combined, y_train)

    # Log best parameters
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV F1 score: {grid_search.best_score_:.4f}")

    for param, value in grid_search.best_params_.items():
        mlflow.log_param(f"best_{param}", value)

    mlflow.log_metric("best_cv_f1_score", grid_search.best_score_)

    # Get best model
    best_model = grid_search.best_estimator_

    # Cross-validation metrics
    print("\nPerforming cross-validation...")
    cv_results = cross_validate(
        best_model,
        X_train_combined,
        y_train,
        cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
        return_train_score=True,
    )

    # Log CV metrics
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        train_metric = cv_results[f"train_{metric}"].mean()
        test_metric = cv_results[f"test_{metric}"].mean()

        mlflow.log_metric(f"cv_train_{metric}", train_metric)
        mlflow.log_metric(f"cv_test_{metric}", test_metric)

        print(f"CV {metric.upper()}: Train={train_metric:.4f}, Test={test_metric:.4f}")

    # Train final model on full training set
    print("\nTraining final model on full training set...")
    best_model.fit(X_train_combined, y_train)

    # Predictions on test set
    y_pred = best_model.predict(X_test_combined)
    y_pred_proba = best_model.predict_proba(X_test_combined)[:, 1]

    # Calculate test metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Log test metrics
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_f1", test_f1)
    mlflow.log_metric("test_roc_auc", test_roc_auc)

    print(f'\n{"="*80}')
    print("TEST SET PERFORMANCE:")
    print(f'{"="*80}')
    print(f"Accuracy:  {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1 Score:  {test_f1:.4f}")
    print(f"ROC AUC:   {test_roc_auc:.4f}")
    print(f'{"="*80}\n')

    # Classification report
    class_report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(class_report)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(conf_matrix)

    # Log confusion matrix
    mlflow.log_metric("true_negatives", int(conf_matrix[0, 0]))
    mlflow.log_metric("false_positives", int(conf_matrix[0, 1]))
    mlflow.log_metric("false_negatives", int(conf_matrix[1, 0]))
    mlflow.log_metric("true_positives", int(conf_matrix[1, 1]))

    # =============================================================================
    # Threshold Tuning - Find optimal decision threshold
    # =============================================================================
    print(f'\n{"="*80}')
    print("THRESHOLD TUNING:")
    print(f'{"="*80}')
    print("Finding optimal decision threshold to maximize F1 score...\n")

    # Try different thresholds
    thresholds = np.arange(0.1, 0.9, 0.05)
    f1_scores = []

    for thresh in thresholds:
        y_pred_thresh = (y_pred_proba >= thresh).astype(int)
        f1 = f1_score(y_test, y_pred_thresh)
        f1_scores.append(f1)
        print(f"Threshold: {thresh:.2f}, F1: {f1:.4f}")

    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = max(f1_scores)

    print(f"\nBest threshold: {best_threshold:.2f}")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Default threshold (0.5) F1: {test_f1:.4f}")
    print(
        f"Improvement: {((best_f1 - test_f1) / test_f1 * 100):.1f}% increase in F1 score"
    )
    print(f'{"="*80}\n')

    # Log optimal threshold metrics to MLflow
    mlflow.log_metric("optimal_threshold", best_threshold)
    mlflow.log_metric("optimal_f1", best_f1)
    mlflow.log_metric("f1_improvement_pct", (best_f1 - test_f1) / test_f1 * 100)

    # =============================================================================
    # Feature Importance Analysis
    # =============================================================================
    print(f'\n{"="*80}')
    print("FEATURE IMPORTANCE ANALYSIS:")
    print(f'{"="*80}\n')

    # Get feature names
    if USE_SEPARATE_VECTORIZERS:
        str_features = [
            "STR_" + f for f in vectorizer_strengths.get_feature_names_out()
        ]
        weak_features = [
            "WEAK_" + f for f in vectorizer_weaknesses.get_feature_names_out()
        ]
        all_feature_names = str_features + weak_features + numerical_features
    else:
        all_feature_names = (
            vectorizer.get_feature_names_out().tolist() + numerical_features
        )

    # Get coefficients from the logistic regression model
    coefficients = best_model.coef_[0]

    # Create feature importance dataframe
    feature_importance_df = pd.DataFrame(
        {"feature": all_feature_names, "coefficient": coefficients}
    )

    # Add absolute value for ranking
    feature_importance_df["abs_coefficient"] = np.abs(
        feature_importance_df["coefficient"]
    )

    # Sort by absolute coefficient
    feature_importance_df = feature_importance_df.sort_values(
        "abs_coefficient", ascending=False
    )

    # Separate into positive (All-Star indicators) and negative (Non-All-Star indicators)
    positive_features = feature_importance_df[
        feature_importance_df["coefficient"] > 0
    ].head(20)
    negative_features = feature_importance_df[
        feature_importance_df["coefficient"] < 0
    ].head(20)

    print("Top 20 Features Predicting ALL-STAR:")
    print("=" * 80)
    for idx, row in positive_features.iterrows():
        print(f"{row['feature']:50s} | Coefficient: {row['coefficient']:+.4f}")

    print(f'\n{"="*80}')
    print("Top 20 Features Predicting NON-ALL-STAR:")
    print("=" * 80)
    for idx, row in negative_features.iterrows():
        print(f"{row['feature']:50s} | Coefficient: {row['coefficient']:+.4f}")

    # =============================================================================
    # Visualization 1: Top 20 Most Important Features (by absolute value)
    # =============================================================================
    print(f'\n{"="*80}')
    print("Creating feature importance visualizations...")
    print(f'{"="*80}\n')

    top_features = feature_importance_df.head(20)

    plt.figure(figsize=(12, 8))
    colors = ["green" if x > 0 else "red" for x in top_features["coefficient"]]
    plt.barh(range(len(top_features)), top_features["coefficient"], color=colors)
    plt.yticks(range(len(top_features)), top_features["feature"])
    plt.xlabel("Coefficient (Impact on All-Star Prediction)", fontsize=12)
    plt.title(
        "Top 20 Most Important Features\n(Green = All-Star Indicator, Red = Non-All-Star Indicator)",
        fontsize=14,
        fontweight="bold",
    )
    plt.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig("feature_importance_top20.png", dpi=300, bbox_inches="tight")
    mlflow.log_artifact("feature_importance_top20.png")
    plt.close()
    print("✓ Saved: feature_importance_top20.png")

    # =============================================================================
    # Visualization 2: Separate Strengths vs Weaknesses (if applicable)
    # =============================================================================
    if USE_SEPARATE_VECTORIZERS:
        # Separate strengths and weaknesses features
        str_feature_importance = feature_importance_df[
            feature_importance_df["feature"].str.startswith("STR_")
        ].head(15)
        weak_feature_importance = feature_importance_df[
            feature_importance_df["feature"].str.startswith("WEAK_")
        ].head(15)

        # Create side-by-side comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Strengths
        colors1 = [
            "green" if x > 0 else "red" for x in str_feature_importance["coefficient"]
        ]
        ax1.barh(
            range(len(str_feature_importance)),
            str_feature_importance["coefficient"],
            color=colors1,
        )
        ax1.set_yticks(range(len(str_feature_importance)))
        ax1.set_yticklabels(
            [f.replace("STR_", "") for f in str_feature_importance["feature"]]
        )
        ax1.set_xlabel("Coefficient", fontsize=11)
        ax1.set_title("Top 15 STRENGTHS Features", fontsize=13, fontweight="bold")
        ax1.axvline(x=0, color="black", linestyle="--", linewidth=0.8)

        # Weaknesses
        colors2 = [
            "green" if x > 0 else "red" for x in weak_feature_importance["coefficient"]
        ]
        ax2.barh(
            range(len(weak_feature_importance)),
            weak_feature_importance["coefficient"],
            color=colors2,
        )
        ax2.set_yticks(range(len(weak_feature_importance)))
        ax2.set_yticklabels(
            [f.replace("WEAK_", "") for f in weak_feature_importance["feature"]]
        )
        ax2.set_xlabel("Coefficient", fontsize=11)
        ax2.set_title("Top 15 WEAKNESSES Features", fontsize=13, fontweight="bold")
        ax2.axvline(x=0, color="black", linestyle="--", linewidth=0.8)

        plt.suptitle(
            "Feature Importance: Strengths vs Weaknesses",
            fontsize=15,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            "feature_importance_strengths_vs_weaknesses.png",
            dpi=300,
            bbox_inches="tight",
        )
        mlflow.log_artifact("feature_importance_strengths_vs_weaknesses.png")
        plt.close()
        print("✓ Saved: feature_importance_strengths_vs_weaknesses.png")

    # =============================================================================
    # Visualization 3: Numerical Features Importance
    # =============================================================================
    numerical_feature_importance = feature_importance_df[
        feature_importance_df["feature"].isin(numerical_features)
    ].sort_values("coefficient", ascending=True)

    if len(numerical_feature_importance) > 0:
        plt.figure(figsize=(10, 6))
        colors = [
            "green" if x > 0 else "red"
            for x in numerical_feature_importance["coefficient"]
        ]
        plt.barh(
            range(len(numerical_feature_importance)),
            numerical_feature_importance["coefficient"],
            color=colors,
        )
        plt.yticks(
            range(len(numerical_feature_importance)),
            numerical_feature_importance["feature"],
        )
        plt.xlabel("Coefficient", fontsize=12)
        plt.title(
            "Numerical Features Importance\n(Green = All-Star Indicator, Red = Non-All-Star Indicator)",
            fontsize=14,
            fontweight="bold",
        )
        plt.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
        plt.tight_layout()
        plt.savefig("feature_importance_numerical.png", dpi=300, bbox_inches="tight")
        mlflow.log_artifact("feature_importance_numerical.png")
        plt.close()
        print("✓ Saved: feature_importance_numerical.png")

    # =============================================================================
    # Save Feature Importance CSV
    # =============================================================================
    feature_importance_df.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")
    os.remove("feature_importance.csv")
    print("✓ Saved: feature_importance.csv")

    print(f'\n{"="*80}')
    print("Feature importance analysis complete!")
    print(f'{"="*80}\n')

    # Save artifacts
    print("\nSaving artifacts...")

    # Save classification report
    with open("classification_report.txt", "w") as f:
        f.write(class_report)
    mlflow.log_artifact("classification_report.txt")
    os.remove("classification_report.txt")

    # Save confusion matrix
    with open("confusion_matrix.txt", "w") as f:
        f.write(str(conf_matrix))
    mlflow.log_artifact("confusion_matrix.txt")
    os.remove("confusion_matrix.txt")

    # Save feature names
    if USE_SEPARATE_VECTORIZERS:
        str_features = [
            "STR_" + f for f in vectorizer_strengths.get_feature_names_out()
        ]
        weak_features = [
            "WEAK_" + f for f in vectorizer_weaknesses.get_feature_names_out()
        ]
        feature_names = str_features + weak_features + numerical_features
    else:
        feature_names = vectorizer.get_feature_names_out().tolist() + numerical_features

    with open("feature_names.txt", "w") as f:
        for name in feature_names:
            f.write(f"{name}\n")
    mlflow.log_artifact("feature_names.txt")
    os.remove("feature_names.txt")

    # Save vectorizer(s)
    if USE_SEPARATE_VECTORIZERS:
        with open("tfidf_vectorizer_strengths.pkl", "wb") as f:
            pickle.dump(vectorizer_strengths, f)
        with open("tfidf_vectorizer_weaknesses.pkl", "wb") as f:
            pickle.dump(vectorizer_weaknesses, f)
        mlflow.log_artifact("tfidf_vectorizer_strengths.pkl")
        mlflow.log_artifact("tfidf_vectorizer_weaknesses.pkl")
        os.remove("tfidf_vectorizer_strengths.pkl")
        os.remove("tfidf_vectorizer_weaknesses.pkl")
    else:
        with open("tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        mlflow.log_artifact("tfidf_vectorizer.pkl")
        os.remove("tfidf_vectorizer.pkl")

    # Log model with signature
    print("Logging model to MLflow...")
    signature = infer_signature(X_train_combined, best_model.predict(X_train_combined))
    mlflow.sklearn.log_model(
        best_model,
        "model",
        signature=signature,
        registered_model_name="NBA_Draft_AllStar_Predictor",
    )

    print(f'\n{"="*80}')
    print("MLflow logging complete!")
    print(f"Run ID: {run.info.run_id}")
    print(f"Experiment ID: {run.info.experiment_id}")
    print(f'{"="*80}\n')


# =============================================================================
# BLOCK 12: Make Predictions for 2026 Draft Prospects
# =============================================================================

print("\n" + "=" * 80)
print("PREDICTING 2026 DRAFT PROSPECTS")
print("=" * 80 + "\n")

if len(predict_df) > 0:
    # Fill NaN values in numerical features
    for col in numerical_features:
        if col in predict_df.columns:
            median_val = train_df[col].median()
            predict_df[col].fillna(median_val, inplace=True)

    X_predict_numerical = predict_df[numerical_features]

    # Transform text features based on experiment configuration
    if USE_SEPARATE_VECTORIZERS:
        X_predict_strengths = predict_df["strengths_processed"]
        X_predict_weaknesses = predict_df["weaknesses_processed"]

        X_predict_str_tfidf = vectorizer_strengths.transform(X_predict_strengths)
        X_predict_weak_tfidf = vectorizer_weaknesses.transform(X_predict_weaknesses)

        # Combine features
        X_predict_combined = hstack(
            [X_predict_str_tfidf, X_predict_weak_tfidf, X_predict_numerical.values]
        )
    else:
        X_predict_text = predict_df["combined_text"]
        X_predict_text_tfidf = vectorizer.transform(X_predict_text)

        # Combine features
        X_predict_combined = hstack([X_predict_text_tfidf, X_predict_numerical.values])

    # Make predictions
    predictions = best_model.predict(X_predict_combined)
    prediction_probas = best_model.predict_proba(X_predict_combined)[:, 1]

    # Add predictions to dataframe
    predict_df["allstar_prediction"] = predictions
    predict_df["allstar_probability"] = prediction_probas

    # Sort by probability
    predict_df_sorted = predict_df.sort_values("allstar_probability", ascending=False)

    # Display top prospects
    print("Top 2026 Draft Prospects (by All-Star probability):")
    print("=" * 80)

    display_cols = ["player", "overall", "allstar_prediction", "allstar_probability"]
    top_prospects = predict_df_sorted[display_cols].head(20)

    for idx, row in top_prospects.iterrows():
        prediction_label = (
            "ALL-STAR" if row["allstar_prediction"] == 1 else "Not All-Star"
        )
        print(
            f"{row['player']:30s} | Overall: {row['overall']:2.0f} | {prediction_label:15s} | Prob: {row['allstar_probability']:.3f}"
        )

    # Save predictions
    predictions_file = f"../data/nba_draft_2026_predictions_{date_string}.csv"
    predict_df_sorted.to_csv(predictions_file, index=False)
    print(f"\nPredictions saved to: {predictions_file}")

    # Summary statistics
    print(f'\n{"="*80}')
    print("PREDICTION SUMMARY:")
    print(f'{"="*80}')
    print(f"Total 2026 prospects: {len(predict_df)}")
    print(f"Predicted All-Stars (within 7 years): {predictions.sum()}")
    print(f"Predicted Non-All-Stars: {len(predictions) - predictions.sum()}")
    print(f"Average All-Star probability: {prediction_probas.mean():.3f}")
    print(f'{"="*80}\n')
else:
    print("No unlabeled 2026 prospects found for prediction.")

print("\n" + "=" * 80)
print("SCRIPT COMPLETE!")
print("=" * 80)
print("\nTo view MLflow results, run:")
print("  mlflow ui")
print("\nThen open http://localhost:5000 in your browser.")
print("=" * 80)
