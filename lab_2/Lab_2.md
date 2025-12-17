# Лабораторная работа №2: Классификация фишинговых email

## Общее описание

**Цель:** адаптировать ML-пайплайн для классификации фишинговых email на основе NLP-техник с использованием AI-ассистентов.

**Задачи:**  
1. Провести планирование NLP pipeline с AI-ассистентом и задокументировать процесс.  
2. Выполнить text-specific EDA (word clouds, n-grams, частотный анализ).  
3. Реализовать NLP preprocessing и векторизацию текстов.  
4. Обучить модели классификации (Logistic Regression, Naive Bayes, Random Forest, SVM).  
5. Провести сравнение моделей, анализ важных слов, оценить качество.  
6. Подготовить отчет о проделанной работе.

**Максимальный балл:** 100

## Структура сдаваемых материалов

```
lab2_[ФАМИЛИЯ]_text_classification.zip
├── lab2_text_classification/        # Проект
│   ├── data/                        # CSV датасета
│   ├── notebooks/                   # phishing_email_classification.ipynb
│   ├── scripts/                     # text_classification_cli.py
│   ├── config/                      # config.yaml
│   ├── artifacts/                   # модели, vectorizers, отчеты, EDA
│   ├── logs/                        # логи
│   └── README.md                    # инструкция по проекту
├── docs/                            # Документы AI-процесса
│   ├── planning_session.md          # диалог с AI про NLP
│   ├── architecture_decisions.md    # решения по preprocessing/vectorization
│   └── ai_prompts_log.md           # ключевые промпты
└── README_submission.md             # Отчет (не более 2 стр.)
```

## Детали задания

### 1. Планирование NLP Pipeline (10 баллов)
- Используйте AI-ассистента для создания плана NLP обработки (промпт в README).  
- Задокументируйте диалог в `docs/planning_session.md`.
- Обоснуйте выбор методов preprocessing и векторизации.

### 2. Text-Specific EDA (20 баллов)
- Постройте word clouds для каждого класса (phishing vs legitimate).
- Проанализируйте распределение длины текстов и количества слов.
- Топ-20 самых частых слов по классам.
- N-gram анализ (биграммы, триграммы).
- Сохраните графики в `artifacts/eda/`.

### 3. NLP Preprocessing и Векторизация (20 баллов)
- Реализуйте preprocessing: lowercase, remove URLs/emails/HTML, токенизация.
- Обработайте стоп-слова (обоснуйте выбор: удалять или нет).
- Реализуйте TF-IDF и Count Vectorizer.
- Сравните эффект разных настроек (ngram_range, max_features).

### 4. Обучение моделей (20 баллов)
- Реализуйте обучение минимум 3 алгоритмов через CLI.
- Протестируйте разные комбинации vectorizer + model.
- Сохраните модели и vectorizers в `artifacts/`.
- Cross-validation для каждой модели.

### 5. Сравнение и интерпретация (20 баллов)
- Создайте таблицу сравнения всех комбинаций (vectorizer × model).
- Выберите лучшую модель по F1-score.
- Проанализируйте feature importance (важные слова для phishing).
- Confusion matrix и classification report.
- Анализ False Positives и False Negatives.

### 6. Отчет и документация (10 баллов)
- Подготовьте `README_submission.md`:  
  - Описание NLP pipeline и результатов.
  - Сравнение TF-IDF vs Count Vectorizer.
  - Интерпретация важных слов для phishing detection.
  - Рекомендации по улучшению (precision vs recall trade-off).
  - Рефлексия по работе с AI-ассистентом.

---

## Требования к выполнению

1. **Production-стандарты кода**: модульность, логирование, CLI, конфигурация.  
2. **NLP Best Practices**: правильная обработка текстов, сохранение vectorizer вместе с моделью.
3. **AI-интеграция**: документируйте процесс планирования NLP pipeline с AI.  
4. **Интерпретируемость**: обязательно проанализируйте важные слова и ошибки.

Удачной работы!