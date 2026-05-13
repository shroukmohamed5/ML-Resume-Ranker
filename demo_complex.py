# -*- coding: utf-8 -*-
"""
Демонстрация для разбора "сложного сценария", запрошенного преподавателем.
- Неоднозначная вакансия без явного прописывания "Java Backend".
- Смешанный набор из 10 резюме: 
  * реальные Java разработчики из CSV,
  * случайный системный аналитик (LLM/другая роль),
  * Data Scientist,
  * нерелевантный менеджер.
"""

import sys
import json
import random
from pathlib import Path

# Обеспечиваем корректный вывод кириллицы в консоль Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from core import ResumeRanker

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / 'Последние файлы'

def main():
    print("\n" + "="*80)
    print(" ДЕМОНСТРАЦИЯ СЛОЖНОГО СЦЕНАРИЯ (Ответ на комментарий преподавателя)")
    print("="*80 + "\n")
    
    ranker = ResumeRanker(seed=42)

    # 1. Неоднозначная вакансия
    # Вакансия "Инженер-разработчик". Не написано "Java" в заголовке, 
    # но в навыках требуются распределенные системы, базы данных, Spring (намек на Java/Backend)
    ambiguous_vacancy = {
        "id": "vac_complex_01",
        "title": "Специалист по разработке распределенных систем",
        "description": "Ищем инженера для работы над высоконагруженными сервисами, проектирования архитектуры, работы с реляционными БД и микросервисами.",
        "skills": ["Микросервисы", "PostgreSQL", "Kafka", "Docker", "Spring Framework", "REST API", "CI/CD"],
        "required_experience_years": 3,
        "salary_to": 250000,
        "location": "Москва"
    }

    print(f"ВАКАНСИЯ: {ambiguous_vacancy['title']}")
    print(f"Требуемые навыки: {', '.join(ambiguous_vacancy['skills'])}")
    print(f"Требуемый опыт: {ambiguous_vacancy['required_experience_years']} года\n")
    print("-" * 80)

    # 2. Соберем микс из резюме: 
    mixed_resumes = []
    
    # -- Берем реальных Java-разработчиков из CSV (с навыками Spring/PostgreSQL)
    csv_path = INPUT_DIR / 'java_senior.csv'
    if csv_path.exists():
        csv_resumes = ranker.parse_csv_resumes(csv_path.read_text(encoding='utf-8'))
        # Выбираем конкретных Java-разработчиков с хорошим набором навыков
        # (позиции 0, 2, 3 в CSV — проверено: Java + Spring + PostgreSQL)
        pick_indices = [0, 2, 3]
        for idx in pick_indices:
            if idx < len(csv_resumes):
                mixed_resumes.append(csv_resumes[idx])
    
    # -- Подкладываем Аналитика (из профессорского SA2)
    sa2_path = INPUT_DIR / 'SA2 (1)._json'
    if sa2_path.exists():
        sa2_data = json.loads(sa2_path.read_text(encoding='utf-8'))
        sa2_data['id'] = "sa2_analyst"
        mixed_resumes.append(sa2_data)
    
    # -- Добавим LLM-выдуманных "нерелевантных" ролей и частично-релевантных
    mixed_resumes.extend([
        {
            "id": "mock_ds_01",
            "position": "Data Scientist / ML Engineer",
            "skills": ["Python", "PyTorch", "Machine Learning", "NLP", "Pandas", "SQL"],
            "experience_years": 4,
            "experience_text": "Обучал нейросети, строил рекомендательные системы.",
            "desired_salary": 200000,
            "location": "Москва"
        },
        {
            "id": "mock_manager_02",
            "position": "Product Manager",
            "skills": ["Jira", "Agile", "Scrum", "Customer Development", "B2B"],
            "experience_years": 5,
            "experience_text": "Управлял командой, писал User Stories, проводил CustDev.",
            "desired_salary": 150000,
            "location": "Санкт-Петербург"
        },
        {
            "id": "mock_frontend_03",
            "position": "Frontend Разработчик (React)",
            "skills": ["JavaScript", "React", "CSS", "HTML", "Redux", "TypeScript"],
            "experience_years": 2,
            "experience_text": "Верстка интерфейсов, SPA приложения, интеграция с REST API.",
            "desired_salary": 120000,
            "location": "Казань"
        },
        {
            "id": "mock_security_04",
            "position": "Специалист по информационной безопасности",
            "skills": ["Penetration Testing", "SOC", "SIEM", "Compliance", "ISO 27001"],
            "experience_years": 6,
            "experience_text": "Проводил аудит ИБ, настраивал DLP, расследовал инциденты.",
            "desired_salary": 180000,
            "location": "Москва"
        }
    ])

    print(f"Всего отобрано резюме для сложного теста: {len(mixed_resumes)}")
    print("Начинаем ранжирование...")
    print("-" * 80)

    # 3. Запуск пайплайна
    results = ranker.process_batch(ambiguous_vacancy, mixed_resumes)
    ranked_list = results.get('ranked_list', [])

    # 4. Вывод и интерпретация результатов
    for i, res in enumerate(ranked_list):
        # Найдем оригинальное резюме для получения названия должности
        orig_res = next((r for r in mixed_resumes if str(r.get('id', '')) == str(res['resume_id'])), {})
        pos_title = orig_res.get('position', '')
        if not pos_title or not isinstance(pos_title, str):
            pos_title = orig_res.get('desired_position', '')
        if isinstance(pos_title, dict):
            pos_title = pos_title.get('title', 'Не указано')
        if not pos_title:
            pos_title = 'Не указано'
        
        print(f"{i+1}. [Score: {res['score']:.4f}] Резюме ID: {res['resume_id']} | Роль: {pos_title}")
        print(f"   Детали оценки (Explanation):")
        details = res['explanation']['components']
        
        # Печатаем факторы без лишнего JSON-формата, чтобы преподавателю было легко читать
        for factor_name, factor_data in details.items():
            print(f"     - {factor_name}: Вклад {factor_data['contribution']:+.4f} (Base: {factor_data['value']:.2f} * Weight: {factor_data['weight']:.2f})")
        
        print(f"   -> Оценка уверенности системы (Confidence): {res['confidence']:.2f}")
        print("-" * 40)

    print("\nИнтерпретация результатов:")
    print("=" * 80)
    for i, res in enumerate(ranked_list):
        orig_res = next((r for r in mixed_resumes if str(r.get('id', '')) == str(res['resume_id'])), {})
        pos_title = orig_res.get('position', '')
        if not pos_title or not isinstance(pos_title, str):
            pos_title = orig_res.get('desired_position', '')
        if isinstance(pos_title, dict):
            pos_title = pos_title.get('title', 'Не указано')
        if not pos_title:
            pos_title = 'Не указано'
        details = res['explanation']['components']
        skills_matched = details['skills'].get('matched', [])
        skills_val = details['skills']['value']
        pos_val = details['position_fit']['value']
        
        if res['score'] > 0.5:
            verdict = "ВЫСОКО РЕЛЕВАНТЕН"
        elif res['score'] > 0.3:
            verdict = "ЧАСТИЧНО РЕЛЕВАНТЕН"
        elif res['score'] > 0.1:
            verdict = "НИЗКАЯ РЕЛЕВАНТНОСТЬ"
        else:
            verdict = "НЕ РЕЛЕВАНТЕН"
        
        print(f"\n#{i+1} [{verdict}] {pos_title} (Score: {res['score']:.4f})")
        reasons = []
        if skills_val > 0.5:
            reasons.append(f"совпадают навыки ({', '.join(skills_matched)})")
        elif skills_val > 0:
            reasons.append(f"частичное совпадение навыков ({', '.join(skills_matched)})")
        else:
            reasons.append("навыки вакансии не представлены в резюме")
        if pos_val >= 0.8:
            reasons.append("тип роли совпадает с вакансией")
        elif pos_val < 0.2:
            reasons.append("тип роли НЕ соответствует вакансии (штраф)")
        print(f"   Причина: {'; '.join(reasons)}")
    
    print("\n" + "=" * 80)
    print("ВЫВОД: Система комбинирует семантическое сходство текстов, прямое совпадение")
    print("навыков, тип должности и опыт для ранжирования. Java/Backend разработчики")
    print("с релевантным стеком (Spring, PostgreSQL, Docker) получают наивысшие баллы,")
    print("а специалисты из других областей (безопасность, менеджмент, фронтенд)")
    print("получают жесткий штраф через position_fit несмотря на любой другой вклад.")

if __name__ == '__main__':
    main()
