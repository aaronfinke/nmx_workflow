from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import pooch
from nmx_workflow.config import RAW_DATA_DIR, META_DATA_DIR
from nmx_workflow import __version__

app = typer.Typer()

MR_MANAGER = pooch.create(
    path=RAW_DATA_DIR,
    base_url="https://project.esss.dk/nextcloud/index.php/s/Db983rx3b67wxbR/",
    version=__version__ if '+' in __version__ else __version__ + '+alpha', # force Pooch to use the "main" folder to avoid re-downloading large files
    version_dev="main",
    registry=None,
)
MR_MANAGER.load_registry(META_DATA_DIR / "pooch-registry.txt")

FILELIST = (
    "nmx_by_scipp_2E12.h5"
)

@app.command()
def download_datafiles(files: list[str] = FILELIST):
    for file in tqdm(files, desc="Fetching"):
        MR_MANAGER.fetch(file, progressbar=False)


if __name__ == "__main__":
    app()
