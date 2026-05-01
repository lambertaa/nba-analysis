@echo off
REM Start MLflow UI with SQLite backend
echo Starting MLflow UI with SQLite backend...
echo.
echo Backend: sqlite:///D:/Andy/python/nba-analysis/data/mlflow_tracking.db
echo Port: 5000
echo.
mlflow ui --backend-store-uri "sqlite:///D:/Andy/python/nba-analysis/data/mlflow_tracking.db" --port 5000

