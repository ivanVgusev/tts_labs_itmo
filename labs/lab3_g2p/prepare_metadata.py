"""Подготовка фонемных метаданных для лабораторной работы №4."""

import csv
from pathlib import Path

import pandas as pd

from g2p import G2P, SIL, build_lexicon, load_mfa_dictionary


LAB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LAB_DIR.parents[1]
TOKENS_PATH = LAB_DIR / "data" / "tokens_phones_sil.csv"
TEXTS_PATH = PROJECT_ROOT / "data" / "metadata_RUSLAN_22200_normalized.csv"
GT_PATH = PROJECT_ROOT / "data" / "metadata_phones_gt.csv"
PREDICTED_PATH = PROJECT_ROOT / "data" / "metadata_phones.csv"
TARGET_PHONES_PATH = PROJECT_ROOT / "common" / "phones_mfa.txt"


def load_texts(path: Path = TEXTS_PATH) -> dict[str, str]:
    """Загрузить нормализованный текст по идентификатору фразы."""
    metadata = pd.read_csv(
        path,
        sep="|",
        header=None,
        names=["id", "text", "original"],
        quoting=csv.QUOTE_NONE,
    )
    return dict(metadata[["id", "text"]].itertuples(index=False, name=None))


def validate_phones(metadata: pd.DataFrame) -> None:
    """Проверить фонемы перед записью файла."""
    with open(TARGET_PHONES_PATH, encoding="utf-8") as file:
        inventory = {line.strip() for line in file if line.strip()}
    observed = {
        phone
        for transcription in metadata.phones
        for phone in transcription.split()
    }
    unknown = observed - inventory
    if unknown:
        raise ValueError(f"Фонемы вне целевого инвентаря: {sorted(unknown)}")


def save_metadata(metadata: pd.DataFrame, path: Path) -> None:
    """Записать метаданные без заголовка с разделителем ``|``."""
    validate_phones(metadata)
    metadata.to_csv(
        path,
        sep="|",
        header=False,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
    )


def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Создать эталонные и предсказанные фонемные метаданные."""
    tokens = pd.read_csv(TOKENS_PATH, sep="|", quoting=csv.QUOTE_NONE)
    texts = load_texts()
    train_lexicon = build_lexicon(tokens[(tokens["set"] == "train") & (tokens["token"] != SIL)])
    g2p = G2P(train_lexicon, load_mfa_dictionary())

    ground_truth_rows = []
    predicted_rows = []
    for utterance_id, sentence in tokens.groupby("id", sort=False):
        sentence = sentence.sort_values("token_id")
        ground_truth = " ".join(sentence.phones)
        predicted = " ".join(
            phone
            for item in g2p.transcribe(sentence.token.to_numpy())
            for phone in item
        )
        text = texts[utterance_id]
        ground_truth_rows.append((utterance_id, text, ground_truth))
        predicted_rows.append((utterance_id, text, predicted))

    columns = ["id", "text", "phones"]
    ground_truth = pd.DataFrame(ground_truth_rows, columns=columns)
    predicted = pd.DataFrame(predicted_rows, columns=columns)
    save_metadata(ground_truth, GT_PATH)
    save_metadata(predicted, PREDICTED_PATH)
    return ground_truth, predicted


if __name__ == "__main__":
    reference, hypothesis = main()
    print(f"Эталонных строк: {len(reference)}")
    print(f"Предсказанных строк: {len(hypothesis)}")
