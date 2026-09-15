"""Простая нормализация русского текста для лабораторной №1."""

import re
import unicodedata


class TextNormalizer:
    """Исправляет Unicode, пробелы и пунктуацию без изменения слов."""

    def __init__(self):
        self.replacements = str.maketrans(
            {
                "\u2011": "-",
                "\u2013": "\u2014",
                "„": '"',
                "“": '"',
                "”": '"',
                "‘": "'",
                "’": "'",
                "*": " ",
            }
        )

    def normalize(self, text: str) -> str:
        """Возвращает текст в NFC с исправленными пробелами и пунктуацией."""

        text = unicodedata.normalize("NFC", text)
        text = text.translate(self.replacements)

        text = re.sub(r"\.{3,}", "\u2026", text)
        text = re.sub(r"\.{2}", ".", text)
        text = re.sub(r"([!?])[.\u2026]+", r"\1", text)
        text = re.sub(r"!{2,}", "!", text)
        text = re.sub(r"\?{2,}", "?", text)

        text = re.sub(r"\s+([,.;:!?\u2026])", r"\1", text)
        text = re.sub(r"([,;:])(?=\S)", r"\1 ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return unicodedata.normalize("NFC", text)
