#!/usr/bin/env python3
"""
Text Classification CLI для phishing email detection
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import joblib
import matplotlib
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import typer
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from wordcloud import WordCloud

matplotlib.use("Agg")

app = typer.Typer(help="CLI для классификации phishing emails")

# Глобальные переменные
logger = None

# Скачиваем необходимые NLTK данные
try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
except:
    pass


def setup_logging(config: Dict) -> logging.Logger:
    """Настройка логирования"""
    log_config = config.get("logging", {})

    logs_dir = Path(config["paths"]["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("text_classification")
    logger.setLevel(getattr(logging, log_config.get("level", "INFO")))

    formatter = logging.Formatter(
        log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    if log_config.get("file_enabled", True):
        file_handler = logging.FileHandler(
            logs_dir / f"text_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if log_config.get("console_enabled", True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def load_config(config_path: str) -> Dict:
    """Загрузка конфигурационного файла"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def preprocess_text(text: str, config: Dict) -> str:
    """Предобработка текста"""
    if not isinstance(text, str):
        return ""

    prep_config = config["preprocessing"]

    # Lowercase
    if prep_config.get("lowercase", True):
        text = text.lower()

    # Remove URLs
    if prep_config.get("remove_urls", True):
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove email addresses
    if prep_config.get("remove_emails", True):
        text = re.sub(r"\S+@\S+", "", text)

    # Remove HTML tags
    if prep_config.get("remove_html", True):
        text = re.sub(r"<.*?>", "", text)

    # Remove special characters
    if prep_config.get("remove_special_chars", True):
        text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove numbers
    if prep_config.get("remove_numbers", False):
        text = re.sub(r"\d+", "", text)

    # Remove extra whitespace
    text = " ".join(text.split())

    return text


@app.command()
def preprocess(
        config: str = typer.Option("config/config.yaml", help="Путь к конфигу"),
        input_file: str = typer.Option(None, help="Входной CSV файл"),
        output_file: str = typer.Option("data/preprocessed.csv", help="Выходной файл"),
):
    """Предобработка текстовых данных"""
    global logger

    cfg = load_config(config)
    logger = setup_logging(cfg)

    logger.info("=== Начало предобработки текстов ===")

    # Загрузка данных
    if not input_file:
        input_file = cfg["data"]["train_csv"]

    df = pd.read_csv(input_file)
    text_col = cfg["data"]["text_column"]

    logger.info(f"Загружено {len(df)} записей")

    # Предобработка
    logger.info("Применение preprocessing...")
    df[f"{text_col}_processed"] = df[text_col].apply(
        lambda x: preprocess_text(x, cfg)
    )

    # Сохранение
    df.to_csv(output_file, index=False)
    logger.info(f"Обработанные данные сохранены: {output_file}")


@app.command()
def eda(
        config: str = typer.Option("config/config.yaml", help="Путь к конфигу"),
        sample_size: Optional[int] = typer.Option(None, help="Размер выборки"),
):
    """Разведочный анализ текстовых данных"""
    global logger

    cfg = load_config(config)
    logger = setup_logging(cfg)

    logger.info("=== Начало EDA для текстовых данных ===")

    # Загрузка
    df = pd.read_csv(cfg["data"]["train_csv"])
    text_col = cfg["data"]["text_column"]
    label_col = cfg["data"]["label_column"]

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    logger.info(f"Анализ {len(df)} записей")

    eda_dir = Path(cfg["paths"]["eda_dir"])
    eda_dir.mkdir(parents=True, exist_ok=True)

    # Базовая статистика
    logger.info("Базовая статистика...")
    stats = {
        "total_samples": len(df),
        "class_distribution": df[label_col].value_counts().to_dict(),
        "avg_text_length": df[text_col].str.len().mean(),
        "avg_word_count": df[text_col].str.split().str.len().mean(),
    }

    with open(eda_dir / "text_statistics.yaml", "w") as f:
        yaml.dump(stats, f)

    # Распределение длин текстов
    logger.info("Анализ длины текстов...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    df[text_col].str.len().hist(bins=50, ax=axes[0])
    axes[0].set_title("Распределение длины текста (символы)")
    axes[0].set_xlabel("Длина")
    axes[0].set_ylabel("Частота")

    df[text_col].str.split().str.len().hist(bins=50, ax=axes[1])
    axes[1].set_title("Распределение количества слов")
    axes[1].set_xlabel("Количество слов")
    axes[1].set_ylabel("Частота")

    plt.tight_layout()
    plt.savefig(eda_dir / "text_length_distribution.png", dpi=300)
    plt.close()

    # Word clouds для каждого класса
    logger.info("Создание word clouds...")
    unique_labels = df[label_col].unique()

    fig, axes = plt.subplots(1, len(unique_labels), figsize=(15, 5))
    if len(unique_labels) == 1:
        axes = [axes]

    for idx, label in enumerate(unique_labels):
        text_data = " ".join(df[df[label_col] == label][text_col].astype(str))

        wordcloud = WordCloud(
            width=cfg["eda"]["wordcloud_width"],
            height=cfg["eda"]["wordcloud_height"],
            max_words=cfg["eda"]["max_words_wordcloud"],
            background_color="white"
        ).generate(text_data)

        axes[idx].imshow(wordcloud, interpolation="bilinear")
        axes[idx].set_title(f"Word Cloud - Class {label}")
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(eda_dir / "wordclouds.png", dpi=300)
    plt.close()

    logger.info(f"EDA завершен. Результаты в {eda_dir}")


@app.command()
def train(
        config: str = typer.Option("config/config.yaml", help="Путь к конфигу"),
        vectorizer: str = typer.Option("tfidf", help="tfidf или count"),
        model_type: str = typer.Option("logistic-regression", help="Тип модели"),
):
    """Обучение модели классификации"""
    global logger

    cfg = load_config(config)
    logger = setup_logging(cfg)

    logger.info(f"=== Обучение модели {model_type} с векторизацией {vectorizer} ===")

    # Загрузка данных
    df = pd.read_csv(cfg["data"]["train_csv"])
    text_col = cfg["data"]["text_column"]
    label_col = cfg["data"]["label_column"]

    # Предобработка
    logger.info("Предобработка текстов...")
    df[f"{text_col}_processed"] = df[text_col].apply(
        lambda x: preprocess_text(x, cfg)
    )

    X = df[f"{text_col}_processed"]
    y = df[label_col]

    # Векторизация
    logger.info(f"Векторизация с {vectorizer}...")
    if vectorizer == "tfidf":
        vec_params = cfg["vectorization"]["tfidf"]
        vectorizer_obj = TfidfVectorizer(**vec_params)
    elif vectorizer == "count":
        vec_params = cfg["vectorization"]["count"]
        vectorizer_obj = CountVectorizer(**vec_params)
    else:
        raise ValueError(f"Неизвестный vectorizer: {vectorizer}")

    X_vec = vectorizer_obj.fit_transform(X)
    logger.info(f"Размерность после векторизации: {X_vec.shape}")

    # Создание модели
    model_key = model_type.replace("-", "_")
    model_params = cfg["models"][model_key]

    if model_type == "logistic-regression":
        model = LogisticRegression(**model_params)
    elif model_type == "naive-bayes":
        model = MultinomialNB(**model_params)
    elif model_type == "random-forest":
        model = RandomForestClassifier(**model_params)
    elif model_type == "svm":
        model = LinearSVC(**model_params)
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")

    # Обучение
    logger.info("Обучение модели...")
    model.fit(X_vec, y)

    # Cross-validation
    logger.info("Cross-validation...")
    cv_scores = cross_val_score(
        model, X_vec, y,
        cv=cfg["training"]["cv_folds"],
        scoring="f1"
    )
    logger.info(f"CV F1-scores: {cv_scores}")
    logger.info(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Сохранение
    models_dir = Path(cfg["paths"]["models_dir"])
    vectorizers_dir = Path(cfg["paths"]["vectorizers_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    vectorizers_dir.mkdir(parents=True, exist_ok=True)

    model_filename = f"{model_type}_{vectorizer}_model.joblib"
    vectorizer_filename = f"{vectorizer}_vectorizer.joblib"

    joblib.dump(model, models_dir / model_filename)
    joblib.dump(vectorizer_obj, vectorizers_dir / vectorizer_filename)

    # Метаданные
    metadata = {
        "model_type": model_type,
        "vectorizer": vectorizer,
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
        "n_features": X_vec.shape[1],
        "n_samples": len(X),
        "timestamp": datetime.now().isoformat(),
    }

    with open(models_dir / f"{model_type}_{vectorizer}_metadata.yaml", "w") as f:
        yaml.dump(metadata, f)

    logger.info(f"Модель сохранена: {model_filename}")
    logger.info(f"Vectorizer сохранен: {vectorizer_filename}")


@app.command()
def predict(
        config: str = typer.Option("config/config.yaml", help="Путь к конфигу"),
        model_path: str = typer.Option(None, help="Путь к модели"),
        vectorizer_path: str = typer.Option(None, help="Путь к vectorizer"),
        input_file: str = typer.Option(None, help="Входной файл"),
        output_file: str = typer.Option("artifacts/predictions.csv", help="Выходной файл"),
):
    """Предсказание на новых данных"""
    global logger

    cfg = load_config(config)
    logger = setup_logging(cfg)

    logger.info("=== Предсказание на новых данных ===")

    # Загрузка модели и vectorizer
    if not model_path or not vectorizer_path:
        logger.error("Необходимо указать model_path и vectorizer_path")
        return

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Загрузка данных
    if not input_file:
        input_file = cfg["data"]["test_csv"]

    df = pd.read_csv(input_file)
    text_col = cfg["data"]["text_column"]

    # Предобработка
    df[f"{text_col}_processed"] = df[text_col].apply(
        lambda x: preprocess_text(x, cfg)
    )

    # Векторизация и предсказание
    X_vec = vectorizer.transform(df[f"{text_col}_processed"])
    predictions = model.predict(X_vec)

    # Вероятности (если поддерживается)
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_vec)
        df["probability"] = probas[:, 1]

    df["prediction"] = predictions

    # Сохранение
    df.to_csv(output_file, index=False)

    logger.info(f"Предсказания сохранены: {output_file}")
    logger.info(f"Распределение предсказаний: {pd.Series(predictions).value_counts().to_dict()}")


@app.command()
def evaluate(
        config: str = typer.Option("config/config.yaml", help="Путь к конфигу"),
        predictions_file: str = typer.Option("artifacts/predictions.csv", help="Файл с предсказаниями"),
):
    """Оценка качества модели"""
    global logger

    cfg = load_config(config)
    logger = setup_logging(cfg)

    logger.info("=== Оценка модели ===")

    # Загрузка предсказаний
    df = pd.read_csv(predictions_file)
    label_col = cfg["data"]["label_column"]

    if label_col not in df.columns:
        logger.error(f"Колонка {label_col} не найдена")
        return

    y_true = df[label_col]
    y_pred = df["prediction"]

    # Метрики
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")

    # Classification report
    logger.info("\n" + classification_report(y_true, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"\nConfusion Matrix:\n{cm}")

    # Сохранение метрик
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
    }

    with open(reports_dir / "evaluation_metrics.yaml", "w") as f:
        yaml.dump(metrics, f)

    logger.info(f"Метрики сохранены в {reports_dir}")


if __name__ == "__main__":
    app()
