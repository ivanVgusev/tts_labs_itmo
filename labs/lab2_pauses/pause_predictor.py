"""Предиктор мест и длительности пауз для второй лабораторной."""

from collections.abc import Sequence
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.metrics import f1_score, mean_absolute_error, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PAUSE_PREDICTOR_DATA = Path("data/RUSLAN_pause_metadata.csv")
MAX_PAUSE_DURATION = 0.48
MIN_PAUSE_DURATION = 0.03
NO_PUNCTUATION = "<NONE>"

CATEGORICAL_FEATURES = ["punctuation", "previous_punctuation", "next_punctuation"]
NUMERIC_FEATURES = [
    "token_length",
    "previous_token_length",
    "next_token_length",
    "relative_position",
    "sentence_length",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def punctuation_type(token: str) -> str:
    """Возвращает последний значимый знак препинания токена."""

    match = re.search(r"([,;:?!\.\-\u2013\u2014\u2026])[^\w]*$", str(token))
    return match.group(1) if match else NO_PUNCTUATION


def make_features(tokens: Sequence[str]) -> pd.DataFrame:
    """Строит текстовые признаки для последовательности токенов."""

    tokens = [str(token) for token in tokens]
    sentence_length = len(tokens)
    rows = []

    for index, token in enumerate(tokens):
        previous_token = tokens[index - 1] if index > 0 else ""
        next_token = tokens[index + 1] if index + 1 < sentence_length else ""
        denominator = max(sentence_length - 1, 1)

        rows.append(
            {
                "punctuation": punctuation_type(token),
                "previous_punctuation": punctuation_type(previous_token),
                "next_punctuation": punctuation_type(next_token),
                "token_length": len(token),
                "previous_token_length": len(previous_token),
                "next_token_length": len(next_token),
                "relative_position": index / denominator,
                "sentence_length": sentence_length,
            }
        )

    return pd.DataFrame(rows, columns=FEATURES)


def filter_relevant_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Убирает финальные и сомнительные паузы из обучения и оценки."""

    punctuation = data["label_raw"].map(punctuation_type)
    relevant_pause = (
        punctuation.ne(NO_PUNCTUATION)
        & data["pause_duration"].le(MAX_PAUSE_DURATION)
    )
    keep = data["is_last_word"].eq(0) & (
        data["is_pause_after"].eq(0) | relevant_pause
    )
    return data.loc[keep].copy()


def make_model_features(data: pd.DataFrame) -> pd.DataFrame:
    """Строит признаки для всех предложений в таблице."""

    parts = []
    for _, sentence in data.groupby("id", sort=False):
        sentence_features = make_features(sentence["label_raw"].tolist())
        sentence_features.index = sentence.index
        parts.append(sentence_features)

    return pd.concat(parts).loc[data.index]


class PausePredictor:
    """Предсказывает паузы только по токенам и соседнему контексту."""

    def __init__(self) -> None:
        def make_preprocessing() -> ColumnTransformer:
            return ColumnTransformer(
                [
                    (
                        "categorical",
                        OneHotEncoder(handle_unknown="ignore"),
                        CATEGORICAL_FEATURES,
                    ),
                    ("numeric", StandardScaler(), NUMERIC_FEATURES),
                ]
            )

        self.classifier = Pipeline(
            [
                ("preprocessing", make_preprocessing()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        class_weight={0: 1.0, 1: 1.2},
                        max_iter=300,
                        solver="liblinear",
                        random_state=42,
                    ),
                ),
            ]
        )
        self.regressor = Pipeline(
            [
                ("preprocessing", make_preprocessing()),
                ("model", HuberRegressor(epsilon=1.1, max_iter=500)),
            ]
        )
        self.is_fitted = False

    def fit(self, data: pd.DataFrame) -> "PausePredictor":
        """Обучает классификатор и регрессор на подготовленной таблице."""

        training_data = filter_relevant_rows(data)
        features = make_model_features(data).loc[training_data.index]

        self.classifier.fit(features, training_data["is_pause_after"])

        pause_rows = training_data["is_pause_after"].eq(1)
        self.regressor.fit(
            features.loc[pause_rows],
            training_data.loc[pause_rows, "pause_duration"],
        )
        self.is_fitted = True
        return self

    def predict(self, tokens: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Предсказывает наличие и длительность паузы после каждого токена."""

        if not self.is_fitted:
            raise RuntimeError("Сначала обучите PausePredictor методом fit")
        if len(tokens) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        features = make_features(tokens)
        is_pause = self.classifier.predict(features).astype(int)
        pause_duration = np.clip(
            self.regressor.predict(features),
            MIN_PAUSE_DURATION,
            MAX_PAUSE_DURATION,
        )
        pause_duration *= is_pause

        is_pause[-1] = 0
        pause_duration[-1] = 0.0
        return is_pause, pause_duration

    def predict_durations(
        self, tokens: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Вставляет SIL после токенов с предсказанной паузой."""

        is_pause, pause_duration = self.predict(tokens)
        tokens_with_pauses = []
        durations_with_pauses = []

        for token, pause, duration in zip(tokens, is_pause, pause_duration):
            tokens_with_pauses.append(token)
            durations_with_pauses.append(-1.0)
            if pause:
                tokens_with_pauses.append("<SIL>")
                durations_with_pauses.append(float(duration))

        return (
            np.asarray(tokens_with_pauses),
            np.asarray(durations_with_pauses, dtype=np.float32),
        )


def add_predictions(data: pd.DataFrame, predictor: PausePredictor) -> pd.DataFrame:
    """Добавляет предсказания ко всем предложениям таблицы."""

    result = data.copy()
    features = make_model_features(result)
    result["is_pause_hat"] = predictor.classifier.predict(features).astype(int)
    result["pause_duration_hat"] = np.clip(
        predictor.regressor.predict(features),
        MIN_PAUSE_DURATION,
        MAX_PAUSE_DURATION,
    )
    result["pause_duration_hat"] *= result["is_pause_hat"]
    result.loc[result["is_last_word"].eq(1), "is_pause_hat"] = 0
    result.loc[result["is_last_word"].eq(1), "pause_duration_hat"] = 0.0
    return result


def calculate_metrics(data: pd.DataFrame) -> dict[str, float]:
    """Считает метрики на очищенной части разметки."""

    evaluation_data = filter_relevant_rows(data)
    true_positive = (
        evaluation_data["is_pause_after"].eq(1)
        & evaluation_data["is_pause_hat"].eq(1)
    )

    return {
        "precision": precision_score(
            evaluation_data["is_pause_after"],
            evaluation_data["is_pause_hat"],
            zero_division=0,
        ),
        "recall": recall_score(
            evaluation_data["is_pause_after"],
            evaluation_data["is_pause_hat"],
            zero_division=0,
        ),
        "f1": f1_score(
            evaluation_data["is_pause_after"],
            evaluation_data["is_pause_hat"],
            zero_division=0,
        ),
        "mae": mean_absolute_error(
            evaluation_data.loc[true_positive, "pause_duration"],
            evaluation_data.loc[true_positive, "pause_duration_hat"],
        ),
    }


def test_pause_predictor() -> None:
    """Обучает модель на train и печатает метрики на train и test."""

    pause_data = pd.read_csv(PAUSE_PREDICTOR_DATA, sep="|")
    predictor = PausePredictor().fit(pause_data[pause_data["set"].eq("train")])
    predicted_data = add_predictions(pause_data, predictor)

    for subset in ["train", "test"]:
        metrics = calculate_metrics(predicted_data[predicted_data["set"].eq(subset)])
        print(
            f"{subset}: "
            f"precision={metrics['precision']:.3f}, "
            f"recall={metrics['recall']:.3f}, "
            f"F1={metrics['f1']:.3f}, "
            f"MAE={metrics['mae']:.3f} с"
        )


if __name__ == "__main__":
    test_pause_predictor()
