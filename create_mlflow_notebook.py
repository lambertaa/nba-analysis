import json

# Create notebook structure
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# NBA Draft NLP Model with MLflow Tracking\n",
                "\n",
                "This notebook:\n",
                "1. Auto-retrieves new All-Star data\n",
                "2. Incrementally scrapes only new draft prospect data\n",
                "3. Trains the NLP model with MLflow tracking for version control and experiment management"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Setup and Imports"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "import os\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "from datetime import datetime\n",
                "import requests\n",
                "from bs4 import BeautifulSoup\n",
                "from IPython.display import clear_output\n",
                "\n",
                "# MLflow imports\n",
                "import mlflow\n",
                "import mlflow.sklearn\n",
                "from mlflow.models.signature import infer_signature\n",
                "\n",
                "# Sklearn imports\n",
                "from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate\n",
                "from sklearn.feature_extraction.text import TfidfVectorizer\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.metrics import classification_report, accuracy_score, confusion_matrix\n",
                "\n",
                "# NLP imports\n",
                "import nltk\n",
                "from nltk.corpus import stopwords\n",
                "from nltk.tokenize import word_tokenize\n",
                "from nltk.stem import WordNetLemmatizer\n",
                "import string\n",
                "\n",
                "# Add data_retrieval to path\n",
                "sys.path.append(os.path.abspath('../data_retrieval'))\n",
                "from realgm_retr import get_realgm_allstar_rosters\n",
                "\n",
                "# Download NLTK data\n",
                "nltk.download('stopwords', quiet=True)\n",
                "nltk.download('punkt', quiet=True)\n",
                "nltk.download('wordnet', quiet=True)\n",
                "\n",
                "# Set date string for file naming\n",
                "date_string = datetime.now().strftime('%Y%m%d')\n",
                "\n",
                "print(f'Setup complete. Date string: {date_string}')"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.9.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Save to file
with open('nba_draft_nlp_ml/mlflow_nlp_model_training.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook created successfully!")

