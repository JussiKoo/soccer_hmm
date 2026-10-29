import os
from cmdstanpy import CmdStanModel

from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

def compile_model(stan_file_path):
    my_model = CmdStanModel(stan_file=stan_file_path, 
                            cpp_options={'STAN_THREADS':'true'})
    print(my_model.name)
    print(my_model.stan_file)
    print(my_model.exe_file)
    return my_model

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    stan_path: Path = MODELS_DIR / "soccer_hmm.stan"
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Compiling the model...")
    compile_model(stan_path)
    # -----------------------------------------


if __name__ == "__main__":
    app()