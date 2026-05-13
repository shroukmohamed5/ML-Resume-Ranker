import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import logging

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sklearn").setLevel(logging.ERROR)

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from core import ResumeRanker

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
INPUT_DIR = BASE_DIR / 'Последние файлы'
REPORTS_DIR = BASE_DIR / 'reports'

CASE_CONFIG = {
    'junior': {
        'vacancy_json': DATA_DIR / 'hh_java_junior_vacancy.json',
        'resume_csv': INPUT_DIR / 'java_junior.csv',
    },
    'middle': {
        'vacancy_json': DATA_DIR / 'hh_java_middle_vacancy.json',
        'resume_csv': INPUT_DIR / 'java_middle.csv',
    },
    'senior': {
        'vacancy_json': DATA_DIR / 'hh_java_senior_vacancy.json',
        'resume_csv': INPUT_DIR / 'java_senior.csv',
    },
}


def load_json(file_path: Path) -> dict:
    return json.loads(file_path.read_text(encoding='utf-8'))


def calculate_field_coverage(resumes: list[dict]) -> dict:
    total = len(resumes) or 1
    metrics = {
        'position': sum(1 for resume in resumes if resume.get('position')),
        'experience': sum(1 for resume in resumes if resume.get('experience') or resume.get('total_experience')),
        'skills': sum(1 for resume in resumes if resume.get('skills') or resume.get('technical_skills')),
        'education': sum(1 for resume in resumes if resume.get('education') or resume.get('education_level')),
        'location': sum(1 for resume in resumes if resume.get('location') or resume.get('location_city')),
        'summary': sum(1 for resume in resumes if resume.get('summary') or resume.get('about_me') or resume.get('work_experience')),
    }
    return {
        key: round((value / total) * 100, 2)
        for key, value in metrics.items()
    }


def summarize_warnings(ranked_list: list[dict]) -> list[tuple[str, int]]:
    warning_counter = Counter()
    for item in ranked_list:
        for warning in item.get('warnings', []):
            warning_counter[warning] += 1
    return warning_counter.most_common(10)


def save_report(case_name: str, vacancy: dict, coverage: dict, result: dict) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = REPORTS_DIR / f'{case_name}_ranking_report.json'
    payload = {
        'case': case_name,
        'vacancy': vacancy,
        'data_quality': coverage,
        'result': result,
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return report_path


def print_case_report(case_name: str, vacancy: dict, resumes: list[dict], result: dict, coverage: dict, report_path: Path) -> None:
    print(f"\n=== Case: {case_name.upper()} ===")
    print(f"Vacancy: {vacancy.get('name', vacancy.get('title', 'Unknown'))}")
    print(f"Resumes loaded: {len(resumes)}")
    print(f"Execution time: {result['meta']['execution_time_sec']} sec")
    print(f"Report saved: {report_path.name}")
    print('Field coverage (%):')
    for field_name, field_value in coverage.items():
        print(f"  - {field_name}: {field_value}")

    warning_summary = summarize_warnings(result['ranked_list'])
    print(f"Batch warnings: {len(result['meta']['batch_warnings'])}")
    if warning_summary:
        print('Top warnings:')
        for warning_text, warning_count in warning_summary[:5]:
            print(f"  - {warning_count}x {warning_text}")

    print('Top 10 ranked resumes:')
    for item in result['ranked_list'][:10]:
        components = item.get('explanation', {}).get('components', {})
        position_fit = components.get('position_fit', {}).get('value', 0)
        experience = components.get('experience', {}).get('value', 0)
        skills = components.get('skills', {}).get('value', 0)
        semantic = components.get('semantic', {}).get('value', 0)
        print(
            f"  - {item['resume_id']} | score={item['score']:.4f} | "
            f"semantic={semantic:.2f} | skills={skills:.2f} | experience={experience:.2f} | position_fit={position_fit:.2f}"
        )


def run_case(ranker: ResumeRanker, case_name: str) -> None:
    config = CASE_CONFIG[case_name]
    vacancy = load_json(config['vacancy_json'])
    resumes_raw = config['resume_csv'].read_text(encoding='utf-8')
    resumes = ranker.parse_csv_resumes(resumes_raw)
    coverage = calculate_field_coverage(resumes)
    result = ranker.process_batch(vacancy, resumes)
    report_path = save_report(case_name, vacancy, coverage, result)
    print_case_report(case_name, vacancy, resumes, result, coverage, report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the real HH-style Java ranking pipeline.')
    parser.add_argument(
        'case_name',
        nargs='?',
        default='senior',
        choices=['junior', 'middle', 'senior', 'all'],
        help='Which Java vacancy/csv case to run. Default: senior',
    )
    args = parser.parse_args()

    print('=== ML Resume Ranker v5 (Real HH-Compatible Pipeline) ===')
    ranker = ResumeRanker(seed=42)

    case_names = list(CASE_CONFIG.keys()) if args.case_name == 'all' else [args.case_name]
    for case_name in case_names:
        run_case(ranker, case_name)


if __name__ == '__main__':
    main()
