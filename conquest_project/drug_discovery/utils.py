import os
import shutil

from drug_discovery.const import OUTPUT_DIR


def prepare_output_directory() -> None:
    """
    Creates the output directory if it does not exist already.
    If it does exist, it will be cleared.
    """
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
