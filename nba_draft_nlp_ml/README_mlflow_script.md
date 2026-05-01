# NBA Draft NLP Model with MLflow - Usage Guide

## Overview

The `mlflow_nlp_model_training.py` script provides a complete MLflow-enabled workflow for training an NLP model to predict NBA All-Star potential from draft prospect scouting reports.

## Key Features

### 1. **Automated Data Retrieval**
- **All-Star Data**: Automatically fetches new All-Star rosters from RealGM
- **Incremental Scraping**: Only scrapes new draft prospects from nbadraft.net (saves time!)
- **Smart Caching**: Uses date-stamped files to avoid redundant downloads

### 2. **MLflow Integration**
- **SQLite Backend**: Tracking data stored in `data/mlflow_tracking.db`
- **Experiment Tracking**: All runs logged under "NBA_Draft_AllStar_Prediction"
- **Version Control**: Model versioning with MLflow Model Registry
- **Artifact Storage**: Models, vectorizers, and reports saved as artifacts

### 3. **Complete ML Pipeline**
- Text preprocessing (tokenization, lemmatization, stop word removal)
- TF-IDF vectorization with numerical features
- Hyperparameter tuning with GridSearchCV
- Cross-validation with detailed metrics
- Model training and evaluation
- Predictions for 2025 draft prospects

## How to Use

### Option 1: Run as a Script
```bash
cd nba_draft_nlp_ml
python mlflow_nlp_model_training.py
```

### Option 2: Execute Block by Block in Jupyter Notebook
1. Open a new Jupyter notebook
2. Copy and paste each block (marked with `# BLOCK X:`) into separate cells
3. Execute cells sequentially

**Blocks:**
- **Block 1**: Setup and Imports
- **Block 2**: MLflow Configuration
- **Block 3**: Auto-Retrieve All-Star Data
- **Block 4**: Scrape Player Links (Incremental)
- **Block 5**: Scrape Player Data (Incremental)
- **Block 6**: Load and Merge Draft Data
- **Block 7**: Add All-Star Labels
- **Block 8**: NLP Text Preprocessing
- **Block 9**: Prepare Features and Target
- **Block 10**: Train-Test Split
- **Block 11**: MLflow Training with Hyperparameter Tuning
- **Block 12**: Make Predictions for 2025 Draft Prospects

## Incremental Scraping

The script intelligently handles incremental updates:

### All-Star Data
- Checks for existing files with today's date
- If found, loads from file
- If not, checks for most recent file and only fetches new years
- Saves combined data with today's date

### Draft Prospect Data
- Loads existing player links from most recent file
- Only scrapes new links not in existing files
- Loads existing scraped player data
- Only scrapes players not already in database
- Combines and saves all data

**Result**: Dramatically faster execution on subsequent runs!

## Parallel Scraping

The script now supports **parallel scraping** for significantly faster data collection:

### Configuration
By default, the script uses **8 parallel workers** for scraping player data:
```python
new_df, missing_players = scrape_player_data(all_player_links, existing_players, n_workers=8)
```

### Adjusting Workers
You can adjust the number of workers based on your needs:
- **n_workers=1**: Sequential processing (slowest, but safest)
- **n_workers=4**: Conservative parallel processing (recommended for slower connections)
- **n_workers=8**: Default - good balance of speed and server load
- **n_workers=16**: Aggressive parallel processing (fastest, but may trigger rate limits)

### Performance Impact
- **Sequential (n_workers=1)**: ~1000 players in ~30-40 minutes
- **Parallel (n_workers=8)**: ~1000 players in ~5-8 minutes
- **Speedup**: Approximately **5-8x faster** with parallel processing!

### Thread Safety
The implementation uses:
- `ThreadPoolExecutor` for parallel HTTP requests
- Thread-safe locks for updating shared data structures
- Progress updates every 10 players to avoid console spam

---

## NLP Experiments

The script supports **three different NLP configurations** to optimize the model's ability to distinguish between player strengths and weaknesses:

### Configuration

Set the experiment type at the top of BLOCK 8B:
```python
NLP_EXPERIMENT = 'separate_with_trigrams'  # Change this to switch experiments
```

### Available Experiments

#### **Baseline: Single Vectorizer**
```python
NLP_EXPERIMENT = 'baseline'
```
- **Approach:** Combines strengths and weaknesses into a single text field
- **Vectorizer:** Single TF-IDF vectorizer
- **N-grams:** Bigrams (1, 2)
- **Max Features:** 500
- **Use Case:** Quick baseline for comparison

**Limitation:** Cannot distinguish if "elite shooter" appears in strengths or weaknesses

---

#### **Experiment 1: Separate Vectorizers**
```python
NLP_EXPERIMENT = 'separate_vectorizers'
```
- **Approach:** Separate TF-IDF vectorizers for strengths and weaknesses
- **Vectorizers:** Two independent vectorizers (one for strengths, one for weaknesses)
- **N-grams:** Bigrams (1, 2)
- **Max Features:** 400 per field (800 total text features)
- **Sublinear TF:** Enabled (dampens very frequent terms)

**Benefits:**
- Same phrase gets different feature columns for strengths vs weaknesses
- Model learns "elite shooter in strengths = positive" vs "elite shooter in weaknesses = negative"
- Preserves semantic context
- **Expected Impact:** +3-7% F1 score improvement over baseline

---

#### **Experiment 2: Separate Vectorizers + Trigrams** ⭐ **RECOMMENDED**
```python
NLP_EXPERIMENT = 'separate_with_trigrams'
```
- **Approach:** Separate vectorizers with expanded n-gram range
- **Vectorizers:** Two independent vectorizers
- **N-grams:** Trigrams (1, 2, 3)
- **Max Features:** 600 per field (1200 total text features)
- **Sublinear TF:** Enabled

**Benefits:**
- All benefits of Experiment 1
- Captures nuanced basketball phrases: "elite three point shooter" vs "three point shooter"
- Better context for complex scouting terminology
- **Expected Impact:** +4-10% F1 score improvement over baseline

**Example Phrases Captured:**
- "lacks elite athleticism" (trigram)
- "elite three point" (trigram)
- "defensive potential upside" (trigram)

---

### MLflow Tracking

All experiment configurations are automatically logged to MLflow:
- `nlp_experiment`: Experiment name (baseline, separate_vectorizers, separate_with_trigrams)
- `use_separate_vectorizers`: Boolean flag
- `ngram_range`: N-gram range used
- `max_features_per_field`: Max features per vectorizer
- `total_text_features`: Total number of text features
- `strengths_features`: Number of strength-specific features (if applicable)
- `weaknesses_features`: Number of weakness-specific features (if applicable)

### Comparing Experiments

To compare all three experiments:

1. Run the script with `NLP_EXPERIMENT = 'baseline'`
2. Run again with `NLP_EXPERIMENT = 'separate_vectorizers'`
3. Run again with `NLP_EXPERIMENT = 'separate_with_trigrams'`
4. Open MLflow UI: `mlflow ui`
5. Compare runs side-by-side to see which performs best

**Metrics to Compare:**
- Test F1 Score (primary metric)
- Test Accuracy
- Test Precision
- Test Recall
- Cross-validation F1 Score

---

### Feature Importance Analysis

After running experiments, you can analyze which features matter most:

**For Separate Vectorizers:**
- Features prefixed with `STR_` come from the strengths vectorizer
- Features prefixed with `WEAK_` come from the weaknesses vectorizer
- This allows you to see which strength-phrases predict All-Star success
- And which weakness-phrases are most damaging

**Example Insights:**
- "STR_elite_shooter" → High positive weight (good predictor)
- "WEAK_elite_shooter" → High negative weight (bad sign)
- "STR_defensive_potential" → Moderate positive weight
- "WEAK_lacks_athleticism" → High negative weight

## MLflow Tracking

### View Results
```bash
mlflow ui
```
Then open http://localhost:5000 in your browser

### What's Tracked
- **Parameters**: All hyperparameters, TF-IDF settings, data splits
- **Metrics**: Accuracy, precision, recall, F1, ROC-AUC (train, CV, test)
- **Artifacts**: 
  - Trained model
  - TF-IDF vectorizer
  - Classification report
  - Confusion matrix
  - Feature names

### Model Registry
Models are automatically registered as "NBA_Draft_AllStar_Predictor" for easy version management.

## Output Files

All files are saved in the `data/` directory with date stamps:

- `nba_allstar_all_YYYYMMDD.csv` - All-Star rosters
- `nbadraft_player_links_YYYYMMDD.txt` - Player links
- `nbadraft_strengths_weaknesses_YYYYMMDD.csv` - Scraped scouting reports
- `nba_draft_2025_predictions_YYYYMMDD.csv` - Predictions for 2025 prospects
- `mlflow_tracking.db` - MLflow tracking database

## Dependencies

Make sure you have these installed:
```bash
pip install mlflow pandas numpy scikit-learn scipy nltk beautifulsoup4 requests
```

## Notes

- First run will take longer as it scrapes all data
- Subsequent runs are much faster due to incremental scraping
- The script uses SQLite for MLflow tracking (no server needed)
- All data is stored locally
- The model predicts "All-Star within 5 years" as the target variable

## Troubleshooting

**Issue**: NLTK data not found
**Solution**: The script auto-downloads NLTK data, but you can manually run:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

**Issue**: MLflow UI not showing runs
**Solution**: Make sure you're in the project root directory when running `mlflow ui`

**Issue**: Scraping fails
**Solution**: Check your internet connection and that nbadraft.net is accessible

