import json

from loguru import logger
from ner_rgu.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


def get_data(fl_name):
    logger.info("Processing data...")
    
    with open(RAW_DATA_DIR / f"{fl_name}.txt", "r") as f:
        raw_data = f.read()
    with open(RAW_DATA_DIR / f"{fl_name}.txt", "r") as f:
        data_splitted_lines = f.readlines()

    with open(PROCESSED_DATA_DIR / f"{fl_name}.json") as f:
        labels = json.load(f)

    logger.success("Processing data complete.")
    return {
        "raw_data": raw_data,
        "data_splitted_lines": data_splitted_lines,
        "labels": labels,
    }
