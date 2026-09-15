"""Классификатор нормализованного и ненормализованного текста для лабораторной №1."""

import csv
import json
import re
import string
from collections import Counter
from pathlib import Path

import emoji
import joblib
import regex
import pandas as pd
from babel.numbers import get_currency_symbol, list_currencies
from sklearn.metrics import f1_score, precision_score, recall_score


BASE_DIR = Path(__file__).resolve().parent
DEV_SET_PATH = BASE_DIR / "data/dev_sentences.csv"
SHORT_FORMS_PATH = BASE_DIR / "data/short_forms.dict"
INTERJECTIONS_PATH = BASE_DIR / "data/interjections.json"


def normalize_spaces(input_string):
    return re.sub(r"\s+", " ", input_string).strip()


short_forms_abbreviations_df = pd.read_csv(SHORT_FORMS_PATH)
short_forms_abbreviations_df = short_forms_abbreviations_df[
    ["Слово (словосочетание)", "Сокращение"]
].dropna()
short_forms_abbreviations_df["Слово (словосочетание)"] = (
    short_forms_abbreviations_df["Слово (словосочетание)"].apply(normalize_spaces)
)
short_forms_abbreviations_df["Сокращение"] = short_forms_abbreviations_df[
    "Сокращение"
].apply(normalize_spaces)


def separate_short_forms_abbreviations(df):
    short_forms = {}
    abbreviations = {}
    for _, row in df.iterrows():
        candidate = row["Сокращение"]
        if "." in candidate or "-" in candidate:
            short_forms[candidate] = row["Слово (словосочетание)"]
        else:
            abbreviations[candidate] = row["Слово (словосочетание)"]
    return short_forms, abbreviations


short_forms_dict, abbreviations_dict = separate_short_forms_abbreviations(
    short_forms_abbreviations_df
)
short_forms_list = short_forms_dict.keys()
abbreviations_list = abbreviations_dict.keys()

with INTERJECTIONS_PATH.open(encoding="utf-8") as file:
    interjections = json.load(file)

currencies_code = list_currencies()
currencies_symbols = {
    get_currency_symbol(currency, locale="en_US") for currency in currencies_code
}
currencies = currencies_code | currencies_symbols

patterns = {
    "dates": r"\b(?:0?[1-9]|[12]\d|3[01])[\./-](?:0?[1-9]|1[0-2])[\./-](?:\d{2}|\d{4})\b",
    "phone_numbers": r"(?:\+7|8)[\s().-]*(?:\d[\s().-]*){10}",
    "ordinal_numbers": r"\b\d+-?(?:й|ый|ой|я|ая|ое|ее|е|ые|ие|ых|их|ым|им|ом|ем|му|ему|ую|ю|го|ого|ей|ими|ыми|ми|м|и)\b",
    "technical": r"[*|/@+<>=\\^~_{}\[\]]",
    "punct_runs": r"(?<![.!?])(?!\?!|\?.\.|\.\.\.)[.!?]{2,}",
    "space_before_comma": r"\s+,",
    "space_after_comma": r",(?:\s{2,}|(?=\S|$))",
    "spaces_run": r"\s{2,}",
    "wrong_quot_marks": r"[“”„‟‹›„““”]",
    "unfinished_quot_marks": r"(\"(=!.+\")|\'(=!.+\')|«(=!.+»))",
}


class EntryStatistics:
    def __init__(self, input_string):
        self.input_string = input_string
        self.input_string_preprocessed = self.preprocess(self.input_string)
        self.input_string_tokenized = self.input_string.split()
        self.input_string_preprocessed_tokenized = self.input_string_preprocessed.split()

        self.len_chars = len(self.input_string)
        self.len_words = len(self.input_string_tokenized)

        self.dict_of_counters = self.build_dict_of_counters()
        self.vector = self.build_vector()

    def build_dict_of_counters(self):
        return {
            "dates": self.regex_find(patterns["dates"]),
            "phone_numbers": self.regex_find(patterns["phone_numbers"]),
            "ordinal_numbers": self.regex_find(patterns["ordinal_numbers"]),
            "technical": self.regex_find(patterns["technical"]),
            "punct_runs": self.regex_find(patterns["punct_runs"]),
            "space_before_comma": self.regex_find(patterns["space_before_comma"]),
            "space_after_comma": self.regex_find(patterns["space_after_comma"]),
            "spaces_run": self.regex_find(patterns["spaces_run"]),
            "wrong_quot_marks": self.regex_find(patterns["wrong_quot_marks"]),
            "unfinished_quot_marks": self.regex_find(patterns["unfinished_quot_marks"]),
            "numerals": self.get_numerals(),
            "currency": self.get_currency(),
            "non_cyrillic": self.get_non_cyrillic(),
            "short_forms": self.get_short_forms(),
            "special_symbols": self.get_special_symbols(),
            "abbreviations": self.get_abbreviations(),
            "interjections": self.get_interjections(),
            "emoji": self.get_emoji(),
        }

    def build_vector(self):
        return [
            sum(counter.values()) / (self.len_chars + 1)
            for counter in self.dict_of_counters.values()
        ]

    def preprocess(self, input_string):
        input_string = input_string.lower()
        return input_string.translate(str.maketrans("", "", string.punctuation))

    def regex_find(self, pattern):
        return Counter(re.findall(pattern, self.input_string))

    def get_numerals(self):
        matches = re.findall(r"-?\d*\.?\d+", self.input_string)
        matches = [float(x) if "." in x else int(x) for x in matches]
        return Counter(matches)

    def get_currency(self):
        pattern = (
            r"(?<!\w)(?:"
            + "|".join(
                re.escape(currency)
                for currency in sorted(currencies, key=len, reverse=True)
            )
            + r")(?!\w)"
        )
        return Counter(re.findall(pattern, self.input_string))

    def get_non_cyrillic(self):
        matches = [
            char
            for char in self.input_string
            if regex.fullmatch(r"[\p{L}]", char)
            and not regex.fullmatch(r"\p{Cyrillic}", char)
        ]
        return Counter(matches)

    def get_short_forms(self):
        input_string = self.input_string.lower()
        short_forms = [(short_form, len(short_form)) for short_form in short_forms_list]
        matches = []

        i = 0
        while i < len(input_string):
            found = None
            found_length = 0
            for short_form, short_form_length in short_forms:
                if input_string.startswith(short_form, i) and short_form_length > found_length:
                    before = input_string[i - 1] if i > 0 else ""
                    after_idx = i + short_form_length
                    after = input_string[after_idx] if after_idx < len(input_string) else ""
                    if not before.isalnum() and not after.isalnum():
                        found = short_form
                        found_length = short_form_length
            if found is not None:
                matches.append(found)
                i += found_length
            else:
                i += 1

        return Counter(matches)

    def get_abbreviations(self):
        def preprocess_punct(input_string):
            return input_string.translate(str.maketrans("", "", string.punctuation))

        tokens = self.input_string_tokenized
        matches = []
        for idx, token in enumerate(tokens):
            abbreviation = preprocess_punct(token)
            if abbreviation not in abbreviations_list:
                continue
            if idx == 0 and len(abbreviation) == 1:
                continue
            if (
                len(abbreviation) == 1
                and idx > 0
                and (tokens[idx - 1] == "." or tokens[idx - 1].endswith("."))
            ):
                continue
            matches.append(abbreviation)

        return Counter(matches)

    def get_special_symbols(self):
        pattern = r".,!?\"\';…:\-\u2014«»\u0301*|/@+<>=\\^~_{}\[\]"
        return Counter(
            regex.findall(rf"[^\p{{L}}\p{{N}}\s{pattern}]", self.input_string)
        )

    def get_interjections(self):
        strip_symbols = string.punctuation + "\u2026«»\u2014"
        tokens = {
            token.strip(strip_symbols).lower() for token in self.input_string_tokenized
        }
        return Counter(i for i in interjections if i.lower() in tokens)

    def get_emoji(self):
        return Counter(match["emoji"] for match in emoji.emoji_list(self.input_string))


class TextFilter:
    """Классифицирует текст аугментированной моделью на векторах признаков."""

    def __init__(self, weights_path=None):
        self.model = None

        if weights_path is None:
            return

        try:
            self.model = joblib.load(weights_path)
        except (FileNotFoundError, OSError, ValueError, EOFError):
            return

    def vectorize(self, text: str) -> list[float]:
        """Преобразует текст в 18 признаков, использованных при обучении."""

        return EntryStatistics(text).vector

    def predict(self, vector: list[float]) -> int:
        """Классифицирует готовый вектор признаков."""

        if self.model is None:
            return 1

        return int(self.model.predict([vector])[0])

    def filter(self, text: str) -> int:
        """Преобразует текст в признаки и классифицирует его за один шаг."""

        return self.predict(self.vectorize(text))


if __name__ == "__main__":
    textfilter = TextFilter()

    dev_files = pd.read_csv(
        DEV_SET_PATH, sep="|", encoding="utf-8", quoting=csv.QUOTE_NONE, header=0
    )
    dev_files["predicted"] = dev_files["text"].apply(textfilter.filter)

    prc = precision_score(dev_files["is_normalized"], dev_files["predicted"])
    rec = recall_score(dev_files["is_normalized"], dev_files["predicted"])
    f1 = f1_score(dev_files["is_normalized"], dev_files["predicted"])
    print(f"F1 Score is {f1:.4f}, Precision is {prc:.4f}, Recall is {rec:.4f}")
