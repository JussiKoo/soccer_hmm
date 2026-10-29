from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import numpy as np
from scipy.stats import gamma as gamma_dist
from scipy.stats import vonmises, norm
import scipy.stats as stats
import pandas as pd
import arviz as az

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    return
    # -----------------------------------------


if __name__ == "__main__":
    app()
