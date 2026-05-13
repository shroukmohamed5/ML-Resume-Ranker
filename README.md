# ML Resume Ranker

## Описание

ML-модуль ранжирования резюме по вакансии. Гибридный подход: семантическое сходство (SBERT) + алгоритмический скоринг по 8 компонентам.

Модель: `intfloat/multilingual-e5-base` (мультиязычная, русский + английский).

## Установка и запуск

```bash
pip install -r requirements.txt

python main.py           # прогон по всем CSV (junior/middle/senior)
python demo_complex.py   # стресс-тест: неоднозначная вакансия + смешанные резюме
python demo_full.py      # полная демонстрация (7 частей)
python validate_real_proxy.py  # расчёт метрик (nDCG, Precision, MRR, Recall)
```
## Веб-интерфейс

Система включает два интерфейса:

- Основной интерфейс ML-ранжирования резюме
- HR Dashboard для анализа кандидатов и принятия решений

Запуск:

```bash
streamlit run app_streamlit.py
```

После запуска доступны страницы:

- Developer Panel (Main UI): `http://localhost:8501`
- HR Panel: `http://localhost:8501/hr_panel`

Система включает:

- ML-интерфейс ранжирования резюме
- HR Dashboard для анализа кандидатов
- Метрики валидации (nDCG, Precision, Recall, MRR)
- Proxy-валидацию на реальных CSV
- Explainable AI scoring
- Поддержку больших batch (~15000 резюме)

## Входные данные

**Вакансия** — JSON в формате HH API:
```json
{
  "name": "Java-разработчик",
  "description": "...",
  "key_skills": [{"name": "Java"}, {"name": "Spring Boot"}],
  "experience": {"id": "moreThan6"},
  "salary": {"from": 320000, "to": 450000, "currency": "RUR"},
  "schedule": {"name": "Полный день"},
  "area": {"name": "Москва"}
}
```

**Резюме** — CSV с HH (колонки: `resume_id`, `desired_position`, `total_experience`, `technical_skills`, `education_level`, `location_city`, `about_me`, `work_experience`, `desired_salary`, `work_schedule`, `employment_types`).

Также поддерживается JSON-формат (SA2).

## Выходной формат

```json
{
  "meta": {"model_name": "...", "seed": 42, "execution_time_sec": 1.23},
  "ranked_list": [
    {
      "resume_id": "abc123",
      "score": 0.82,
      "confidence": 0.91,
      "explanation": {
        "components": {
          "semantic": {"value": 0.65, "weight": 0.30, "contribution": 0.195},
          "skills": {"value": 0.86, "weight": 0.25, "contribution": 0.215, "matched": ["Java", "Spring"]},
          ...
        }
      },
      "warnings": []
    }
  ]
}
```

JSON-схемы — в `schemas/`.

## Компоненты скоринга и обоснование весов

Веса определены методом экспертной оценки и откалиброваны итеративно на реальных CSV (java_junior/middle/senior по ~5000 резюме). Критерий: Java-разработчики с релевантным стеком в top-10, нерелевантные роли (менеджеры, фронтенд) — в нижней части.

| Компонент | Вес | Что делает | Почему такой вес |
|-----------|-----|-----------|-----------------|
| semantic | 0.30 | Косинусное сходство SBERT-эмбеддингов | Основной сигнал, улавливает неявные связи в тексте |
| hard_skills | 0.25 | Совпадение навыков (с нормализацией и алиасами) | Самый объективный фактор — рекрутеры фильтруют по навыкам первым делом |
| experience | 0.20 | Отношение стажа к требованию | Порог компетенции: мало опыта = штраф |
| position_fit | 0.10 | Тип роли (backend/frontend/devops/...) + штраф | Фильтр: менеджер на вакансию разработчика → score ≈ 0 |
| salary_fit | 0.05 | Сравнение ЗП с бюджетом вакансии | Мягкий сигнал (многие не указывают ЗП) |
| education | 0.04 | Наличие высшего образования | Низкая предиктивная сила для IT-позиций |
| work_format | 0.03 | Совместимость графика | Часто решается на собеседовании |
| location | 0.03 | Совпадение города | В эпоху удалёнки — минимальное значение |

Три уровня значимости:
- **Ключевые** (semantic + skills + experience = 0.75) — профессиональное соответствие
- **Фильтрующий** (position_fit = 0.10) — штраф за несовпадение типа роли (мультипликативный)
- **Контекстные** (salary, education, format, location = 0.15) — тонкая настройка порядка

## Воспроизводимость

seed=42 (random, numpy, torch). Модель и версия пишутся в выходной `meta`.

## Файлы

```
core.py                 — ядро (ResumeRanker)
main.py                 — запуск ранжирования по CSV
demo_full.py            — демонстрация (7 частей)
demo_complex.py         — стресс-тест с неоднозначной вакансией
validate_real_proxy.py  — proxy-валидация (nDCG, Precision, MRR, Recall)
app_streamlit.py        — основной Streamlit-интерфейс ML ранжирования
hr_panel.py             — HR Dashboard и панель аналитики кандидатов
schemas/                — JSON-схемы входа/выхода
data/                   — вакансии (HH API JSON)
Последние файлы/        — CSV резюме + SA2 пример
```
<img width="1894" height="721" alt="image" src="https://github.com/user-attachments/assets/e6aa74aa-07f8-4e1a-ab68-417ddc9e15a2" />

<img width="1904" height="940" alt="image" src="https://github.com/user-attachments/assets/bc895e1e-c8ad-45e6-9d7e-a3e030f9ac1f" />

<img width="1880" height="822" alt="image" src="https://github.com/user-attachments/assets/8b2a6f06-f602-411f-9d0e-dd63fbc7a12f" />

<img width="1899" height="928" alt="image" src="https://github.com/user-attachments/assets/51d50936-9b9e-43a7-a2ae-d675fcdab4fd" />

<img width="1899" height="894" alt="image" src="https://github.com/user-attachments/assets/8f6aa2f4-0f3c-4cd4-af22-89713bbf0d78" />

<img width="745" height="489" alt="image" src="https://github.com/user-attachments/assets/df3296d7-9be5-48b2-a383-2a0e74f7fa8d" />

<img width="1897" height="932" alt="image" src="https://github.com/user-attachments/assets/5ae546e9-ed81-4a3d-831a-c73cdc01e049" />






