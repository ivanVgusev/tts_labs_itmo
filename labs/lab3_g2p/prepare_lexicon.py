"""Подготовка данных для лабораторной №3.

Скрипт читает выравнивания MFA из ``data/RUSLAN_22050_align_v2`` и создаёт:

* ``data/tokens_phones.csv`` – слова и их фонемы без пауз;
* ``data/tokens_phones_sil.csv`` – та же разметка с токенами ``<SIL>``;
* ``data/lexicon.csv`` – частотный словарь произношений только по train.

Запускать из директории лабораторной:

    python prepare_lexicon.py
"""

import csv
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from praatio import textgrid
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALIGN_DIR = PROJECT_ROOT / "data" / "RUSLAN_22050_align_v2"
NORMALIZED_META = PROJECT_ROOT / "data" / "metadata_RUSLAN_22200_normalized.csv"
DATA_DIR = Path(__file__).resolve().parent / "data"
TOKENS_PATH = DATA_DIR / "tokens_phones.csv"
TOKENS_SIL_PATH = DATA_DIR / "tokens_phones_sil.csv"
LEXICON_PATH = DATA_DIR / "lexicon.csv"

SIL = "<SIL>"
SPN = "spn"
LENGTH_MARK = "ː"
TEST_EVERY = 5


def normalize_phone(label: str) -> str:
    """Привести фонему к форме, используемой целевым инвентарём."""
    return unicodedata.normalize("NFC", label).replace(LENGTH_MARK, "")


def phones_of_token(
    phone_entries: list[tuple[float, float, str]],
    start: float,
    end: float,
) -> list[str]:
    """Собрать фонемы слова по серединам фонемных интервалов.

    Границы уровней слов и фонем иногда различаются на малую величину. Поэтому
    принадлежность определяем по середине фонемы, а не по точному совпадению границ.
    Пустые фонемные интервалы здесь не добавляем: паузы берём с уровня слов и
    представляем отдельным токеном ``<SIL>``.
    """
    result = []
    for phone_start, phone_end, label in phone_entries:
        midpoint = (phone_start + phone_end) / 2
        if start <= midpoint < end and label:
            result.append(normalize_phone(label))
    return result


def read_kept_ids(path: Path) -> set[str]:
    """Прочитать идентификаторы фраз, оставшихся после лабораторной №1."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Не найдены нормализованные метаданные: {path}. "
            "Сначала выполните лабораторную №1."
        )

    with path.open(encoding="utf-8") as file:
        return {line.split("|", 1)[0] for line in file if line.strip()}


def split_name(utterance_id: str) -> str:
    """Разделить данные тем же правилом, что использовалось в лабораторной №2."""
    utterance_number = int(utterance_id.split("_", 1)[0])
    return "test" if utterance_number % TEST_EVERY == 0 else "train"


def read_utterance(path: Path) -> tuple[list[dict], bool]:
    """Прочитать одну фразу и вернуть токены с паузами и признак наличия spn."""
    utterance_id = path.stem
    grid = textgrid.openTextgrid(str(path), includeEmptyIntervals=True)
    word_tier = grid.getTier("words")
    phone_tier = grid.getTier("phones")

    phone_entries = [
        (entry.start, entry.end, entry.label)
        for entry in phone_tier.entries
    ]
    has_spn = any(normalize_phone(label) == SPN for _, _, label in phone_entries if label)

    rows = []
    for entry in word_tier.entries:
        if entry.label:
            token = unicodedata.normalize("NFC", entry.label).lower()
            phones = phones_of_token(phone_entries, entry.start, entry.end)
        else:
            token = SIL
            phones = [SIL]

        rows.append(
            {
                "id": utterance_id,
                "token": token,
                "phones": " ".join(phones),
                "set": split_name(utterance_id),
            }
        )

    return rows, has_spn


def add_token_ids(rows: list[dict]) -> list[dict]:
    """Добавить непрерывный номер токена внутри каждой фразы."""
    return [
        {
            "id": row["id"],
            "token_id": token_id,
            **{key: value for key, value in row.items() if key != "id"},
        }
        for token_id, row in enumerate(rows)
    ]


def read_alignments(
    align_dir: Path,
    kept_ids: set[str],
) -> tuple[list[dict], list[dict], dict]:
    """Прочитать корпус одним проходом и удалить фразы с ``spn`` целиком."""
    token_rows = []
    token_rows_sil = []
    spn_ids = []
    empty_transcription_ids = []
    failed_ids = []

    paths = sorted(align_dir.glob("*.TextGrid"))
    if not paths:
        raise FileNotFoundError(f"В директории нет TextGrid-файлов: {align_dir}")
    alignment_ids = {path.stem for path in paths}

    for path in tqdm(paths, desc="Чтение TextGrid"):
        if path.stem not in kept_ids:
            continue

        try:
            utterance_rows, has_spn = read_utterance(path)
        except Exception as error:
            failed_ids.append((path.stem, str(error)))
            continue

        if has_spn:
            spn_ids.append(path.stem)
            continue

        word_rows = [row for row in utterance_rows if row["token"] != SIL]
        if any(not row["phones"] for row in word_rows):
            empty_transcription_ids.append(path.stem)
            continue

        token_rows.extend(add_token_ids(word_rows))
        token_rows_sil.extend(add_token_ids(utterance_rows))

    summary = {
        "textgrid_files": len(paths),
        "kept_ids": len(kept_ids),
        "missing_alignment_ids": sorted(kept_ids - alignment_ids),
        "spn_ids": spn_ids,
        "empty_transcription_ids": empty_transcription_ids,
        "failed_ids": failed_ids,
    }
    return token_rows, token_rows_sil, summary


def build_lexicon(rows: list[dict]) -> dict[str, Counter]:
    """Посчитать варианты произношения только на обучающей части корпуса."""
    lexicon: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        if row["set"] == "train":
            lexicon[row["token"]][row["phones"]] += 1
    return lexicon


def save_rows(rows: list[dict], path: Path) -> None:
    """Сохранить токенную таблицу с разделителем ``|``."""
    columns = ["id", "token_id", "token", "phones", "set"]
    pd.DataFrame(rows, columns=columns).to_csv(
        path,
        sep="|",
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
    )


def save_lexicon(lexicon: dict[str, Counter], path: Path) -> None:
    """Сохранить словарь в формате ``token|phones|frequency``."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="|", quoting=csv.QUOTE_NONE)
        writer.writerow(["token", "phones", "frequency"])
        for token in sorted(lexicon):
            for phones, frequency in lexicon[token].most_common():
                writer.writerow([token, phones, frequency])


def main() -> None:
    """Создать обе токенные таблицы и обучающий словарь."""
    kept_ids = read_kept_ids(NORMALIZED_META)
    token_rows, token_rows_sil, summary = read_alignments(ALIGN_DIR, kept_ids)
    lexicon = build_lexicon(token_rows)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_rows(token_rows, TOKENS_PATH)
    save_rows(token_rows_sil, TOKENS_SIL_PATH)
    save_lexicon(lexicon, LEXICON_PATH)

    utterance_ids = {row["id"] for row in token_rows}
    test_ids = {row["id"] for row in token_rows if row["set"] == "test"}
    pronunciation_variants = sum(len(counter) > 1 for counter in lexicon.values())

    print(f"Фраз после фильтрации: {len(utterance_ids):,}")
    print(f"Фраз test: {len(test_ids):,}")
    print(f"Слов без пауз: {len(token_rows):,}")
    print(f"Токенов с паузами: {len(token_rows_sil):,}")
    print(f"Фраз удалено из-за spn: {len(summary['spn_ids']):,}")
    print(
        "Фраз из лабораторной №1 без TextGrid: "
        f"{len(summary['missing_alignment_ids']):,}"
    )
    print(
        "Фраз удалено из-за пустой транскрипции слова: "
        f"{len(summary['empty_transcription_ids']):,}"
    )
    print(f"Ошибок чтения TextGrid: {len(summary['failed_ids']):,}")
    print(f"Слов в train-словаре: {len(lexicon):,}")
    print(f"Слов с несколькими произношениями: {pronunciation_variants:,}")


if __name__ == "__main__":
    main()
