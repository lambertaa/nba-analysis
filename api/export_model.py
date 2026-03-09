"""
Export Model Artifacts from MLflow

This script exports the trained model and artifacts from MLflow
to the API's models directory.
"""

import os
import sys
import shutil
import pickle
import mlflow
from pathlib import Path

# Add parent directory to path to import from nba_draft_nlp_ml
sys.path.append(os.path.abspath(".."))


def export_latest_model(output_dir: str = "models"):
    """
    Export the latest model from MLflow to the API models directory.

    Args:
        output_dir: Directory to export model artifacts to
    """
    # Set MLflow tracking URI
    mlflow_db_path = os.path.abspath("../data/mlflow_tracking.db")
    mlflow_db_uri = mlflow_db_path.replace("\\", "/")
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_uri}")

    # Set experiment
    experiment_name = "NBA_Draft_AllStar_Prediction"
    mlflow.set_experiment(experiment_name)

    # Get the latest run
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if not experiment:
        print(f"Error: Experiment '{experiment_name}' not found!")
        return False

    # Get all runs sorted by start time (most recent first)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

    if runs.empty:
        print("Error: No runs found in experiment!")
        return False

    latest_run = runs.iloc[0]
    run_id = latest_run.run_id

    print(f"Exporting model from run: {run_id}")
    print(f"Run metrics:")
    print(f"  - Test F1: {latest_run.get('metrics.test_f1', 'N/A')}")
    print(f"  - Optimal F1: {latest_run.get('metrics.optimal_f1', 'N/A')}")
    print(
        f"  - Optimal Threshold: {latest_run.get('metrics.optimal_threshold', 'N/A')}"
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get artifact URI
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id)

    print(f"\nAvailable artifacts:")
    for artifact in artifacts:
        print(f"  - {artifact.path}")

    # Download model
    try:
        # Download the model directory
        print(f"\nDownloading model artifacts...")
        model_dir = client.download_artifacts(run_id, "model")

        # The model is saved as model.pkl inside the model directory
        source_model = os.path.join(model_dir, "model.pkl")
        dest_model = os.path.join(output_dir, "model.pkl")

        if os.path.exists(source_model):
            shutil.copy(source_model, dest_model)
            print(f"✓ Exported model to: {dest_model}")
        else:
            print(f"Warning: model.pkl not found in {model_dir}")
            print(f"Contents: {os.listdir(model_dir)}")
            return False

    except Exception as e:
        print(f"Error downloading model: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Download vectorizers
    try:
        artifact_path = client.download_artifacts(
            run_id, "tfidf_vectorizer_strengths.pkl"
        )
        shutil.copy(artifact_path, os.path.join(output_dir, "vectorizer_strengths.pkl"))
        print(f"✓ Exported strengths vectorizer")

        artifact_path = client.download_artifacts(
            run_id, "tfidf_vectorizer_weaknesses.pkl"
        )
        shutil.copy(
            artifact_path, os.path.join(output_dir, "vectorizer_weaknesses.pkl")
        )
        print(f"✓ Exported weaknesses vectorizer")

    except Exception as e:
        print(f"Error downloading vectorizers: {e}")
        return False

    # Download feature names
    try:
        artifact_path = client.download_artifacts(run_id, "feature_names.txt")
        shutil.copy(artifact_path, os.path.join(output_dir, "feature_names.txt"))
        print(f"✓ Exported feature names")
    except Exception as e:
        print(f"Warning: Could not download feature names: {e}")

    # Save optimal threshold
    try:
        optimal_threshold = latest_run.get("metrics.optimal_threshold", 0.65)
        threshold_file = os.path.join(output_dir, "optimal_threshold.txt")
        with open(threshold_file, "w") as f:
            f.write(str(optimal_threshold))
        print(f"✓ Exported optimal threshold: {optimal_threshold}")
    except Exception as e:
        print(f"Warning: Could not save optimal threshold: {e}")

    # Copy feature importance visualizations (optional)
    try:
        for viz_file in [
            "feature_importance_top20.png",
            "feature_importance_strengths_vs_weaknesses.png",
            "feature_importance_numerical.png",
        ]:
            try:
                artifact_path = client.download_artifacts(run_id, viz_file)
                shutil.copy(artifact_path, os.path.join(output_dir, viz_file))
                print(f"✓ Exported {viz_file}")
            except:
                pass
    except Exception as e:
        print(f"Note: Some visualizations not available")

    print(f"\n{'='*60}")
    print(f"Model export complete!")
    print(f"Artifacts saved to: {os.path.abspath(output_dir)}")
    print(f"{'='*60}")

    return True


if __name__ == "__main__":
    success = export_latest_model()
    sys.exit(0 if success else 1)
