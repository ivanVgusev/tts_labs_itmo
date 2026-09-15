"""Нормализация и фильтрация метаданных RUSLAN для лабораторной №1."""

import csv

import pandas as pd

from text_filter import TextFilter
from text_normalizer import TextNormalizer


INPUT_PATH = "../../data/metadata_RUSLAN_22200.csv"
OUTPUT_PATH = "../../data/metadata_RUSLAN_22200_normalized.csv"
WEIGHTS_PATH = "data/classifiers/feature_vectors/logreg_augmented.joblib"

CSV_KWARGS = {
    "sep": "|",
    "quoting": csv.QUOTE_NONE,
}


if __name__ == "__main__":
    normalizer = TextNormalizer()
    text_filter = TextFilter(WEIGHTS_PATH)

    raw = pd.read_csv(INPUT_PATH, names=["id", "raw"], **CSV_KWARGS)
    raw["nrm"] = raw["raw"].apply(normalizer.normalize)
    clean = raw[raw["nrm"].apply(text_filter.filter) == 1]

    fixed_count = (raw["raw"] != raw["nrm"]).sum()
    excluded_count = len(raw) - len(clean)

    print(f"Всего фраз: {len(raw)}")
    print(f"Исправлено: {fixed_count}")
    print(f"Исключено: {excluded_count}")
    print(f"Сохранено: {len(clean)}")

    clean[["id", "raw", "nrm"]].to_csv(
        OUTPUT_PATH,
        index=False,
        header=False,
        **CSV_KWARGS,
    )
