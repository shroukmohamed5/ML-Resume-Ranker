import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("OMP_NUM_THREADS",        "1")
os.environ.setdefault("MKL_NUM_THREADS",        "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK",  "TRUE")

import logging

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sklearn").setLevel(logging.ERROR)

import json
import time
import re
import csv
import io
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False


class PositionType(Enum):
    """High-level role categories used for vacancy and resume matching."""
    BACKEND    = "backend"
    FRONTEND   = "frontend"
    FULLSTACK  = "fullstack"
    DEVOPS     = "devops"
    MOBILE     = "mobile"
    QA         = "qa"
    ML         = "ml"
    DATA       = "data"
    SECURITY   = "security"
    MANAGEMENT = "management"
    UNKNOWN    = "unknown"


@dataclass
class RankResult:
    resume_id:   str
    score:       float
    confidence:  float
    explanation: Dict[str, Any]
    warnings:    List[str]


class ResumeRanker:
    """Hybrid resume ranker combining semantic similarity with hard business rules."""

    # ── Position keywords (Russian + English) ─────────────────────────────────
    BACKEND_KEYWORDS = {
        "backend", "back-end", "server-side", "python", "java", "node",
        "django", "flask", "spring", "golang", "бэкенд",
        "бэкенд-разработчик", "серверн", "микросервис",
    }
    FRONTEND_KEYWORDS = {
        "frontend", "front-end", "react", "vue", "angular", "html", "css",
        "javascript", "typescript", "uiux", "фронтенд", "веб-интерфейс",
    }
    FULLSTACK_KEYWORDS  = {"fullstack", "full-stack", "full stack", "полный стек", "фуллстек"}
    DEVOPS_KEYWORDS     = {
        "devops", "kubernetes", "docker", "terraform", "jenkins",
        "gitlab", "aws", "azure", "ansible", "helm", "ci/cd",
    }
    MOBILE_KEYWORDS     = {"mobile", "ios", "android", "swift", "kotlin", "мобильн", "мобайл"}
    QA_KEYWORDS         = {
        "qa", "qa specialist", "quality assurance", "test", "selenium",
        "pytest", "тест", "тестирован", "автотест",
    }
    ML_KEYWORDS         = {
        "machine learning", "ml", "deep learning", "tensorflow", "pytorch",
        "scikit", "ml engineer", "data scientist",
    }
    DATA_KEYWORDS       = {
        "data analyst", "data engineer", "analytics", "spark", "hadoop",
        "etl", "pandas", "numpy", "tableau", "power bi", "аналитик", "данных",
    }
    SECURITY_KEYWORDS   = {
        "информационная безопасность", "information security",
        "кибербезопасность", "cybersecurity", "penetration", "soc analyst",
        "корпоративная безопасность", "защита информации", "security engineer",
        "security analyst", "security specialist", "security officer",
        "специалист по безопасности", "инженер безопасности",
    }
    MANAGEMENT_KEYWORDS = {
        "product manager", "project manager", "менеджер", "руководитель",
        "управление проектами", "scrum master", "бизнес-аналитик",
        "team lead", "тимлид", "директор", "cto", "ceo", "coo",
        "product owner", "agile coach",
    }

    _LOCATION_ALIASES: Dict[str, str] = {
        "москва": "moscow", "moscow": "moscow", "msk": "moscow",
        "санкт-петербург": "spb", "saint petersburg": "spb",
        "st. petersburg": "spb", "спб": "spb", "питер": "spb",
        "новосибирск": "novosibirsk", "novosibirsk": "novosibirsk",
        "екатеринбург": "yekaterinburg", "yekaterinburg": "yekaterinburg",
        "казань": "kazan", "kazan": "kazan",
        "нижний новгород": "nizhny novgorod",
        "самара": "samara", "омск": "omsk",
        "челябинск": "chelyabinsk", "ростов-на-дону": "rostov",
        "уفا": "ufa", "красноярск": "krasnoyarsk",
        "удаленно": "remote", "remote": "remote",
        "дистанционно": "remote", "удалённо": "remote",
    }

    def __init__(self, model_name: str = 'intfloat/multilingual-e5-base', seed: int = 42):
        self.model_name = model_name
        self.seed       = seed
        self.model      = None
        self._device    = 'cpu'
        self._set_seed(seed)

        self.weights = {
            'semantic': 0.30,  # Vacancy-resume semantic similarity.
            'hard_skills': 0.25,  # Direct overlap of required skills.
            'experience': 0.20,  # Experience requirement satisfaction.
            'position_fit': 0.10,  # Role-type match penalty.
            'salary_fit': 0.05,  # Salary budget compatibility.
            'work_format': 0.03,  # Schedule / remote preference match.
            'education': 0.04,  # Education bonus.
            'location': 0.03  # Location bonus.
        }

        # CHANGE 3: cache_max 2048 → 10000
        self._embedding_cache:     Dict[str, Any] = {}
        self._embedding_cache_max: int            = 10000

        self._MAX_EXPERIENCE_YEARS: float = 50.0
        self._MAX_SALARY:           float = 10_000_000.0
        self._MIN_SALARY:           float = 100.0

        self.load_model()

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark     = False

    def load_model(self) -> None:
        """Auto GPU detection + print model name on startup."""
        if not MODEL_AVAILABLE:
            logger.warning("sentence-transformers not installed. Fallback mode active.")
            return

        if TORCH_AVAILABLE and torch.cuda.is_available():
            self._device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            logger.info("GPU detected: %s (%.1fGB VRAM)", gpu_name, vram_gb)
        else:
            self._device = 'cpu'
            logger.info("No GPU detected, using CPU")

        # طباعة اسم الموديل عند التشغيل
        logger.info("Loading model %s on %s...", self.model_name, self._device.upper())
        try:
            self.model = SentenceTransformer(self.model_name, device=self._device)
        except Exception as exc:
            logger.warning("Remote load failed, trying local cache: %s", exc)
            try:
                self.model = SentenceTransformer(
                    self.model_name, device=self._device, local_files_only=True
                )
            except Exception as exc2:
                logger.error("Could not load model: %s", exc2)
                self.model = None
                return
        logger.info("Model loaded on %s successfully.", self._device.upper())

    @staticmethod
    def _parse_russian_experience(exp_string: str) -> float:
        if not exp_string or not isinstance(exp_string, str):
            return 0.0
        exp_string = exp_string.lower().strip().replace(',', '.')
        if not exp_string:
            return 0.0
        years, months = 0.0, 0.0
        y = re.search(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?|год(?:а)?|лет|г\.)', exp_string)
        if y:
            years = float(y.group(1))
        m = re.search(r'(\d+(?:\.\d+)?)\s*(?:months?|mos?|мес(?:яц(?:ев|а)?)?|мес\.|м\.)', exp_string)
        if m:
            months = float(m.group(1))
        if years == 0.0 and months == 0.0:
            plain = re.fullmatch(r'\d+(?:\.\d+)?', exp_string.strip())
            if plain:
                years = float(plain.group())
        return round(years + months / 12.0, 2)

    @staticmethod
    def _detect_position_type(text: str) -> PositionType:
        if not text:
            return PositionType.UNKNOWN
        t = text.lower()
        if any(k in t for k in ResumeRanker.ML_KEYWORDS):         return PositionType.ML
        if any(k in t for k in ResumeRanker.FULLSTACK_KEYWORDS):  return PositionType.FULLSTACK
        if any(k in t for k in ResumeRanker.BACKEND_KEYWORDS):    return PositionType.BACKEND
        if any(k in t for k in ResumeRanker.FRONTEND_KEYWORDS):   return PositionType.FRONTEND
        if any(k in t for k in ResumeRanker.DEVOPS_KEYWORDS):     return PositionType.DEVOPS
        if any(k in t for k in ResumeRanker.MOBILE_KEYWORDS):     return PositionType.MOBILE
        if any(k in t for k in ResumeRanker.QA_KEYWORDS):         return PositionType.QA
        if any(k in t for k in ResumeRanker.DATA_KEYWORDS):       return PositionType.DATA
        if any(k in t for k in ResumeRanker.SECURITY_KEYWORDS):   return PositionType.SECURITY
        if any(k in t for k in ResumeRanker.MANAGEMENT_KEYWORDS): return PositionType.MANAGEMENT
        return PositionType.UNKNOWN

    @staticmethod
    def _is_position_type_match(vacancy_type: PositionType,
                                 resume_type: PositionType) -> Tuple[bool, float]:
        if vacancy_type == PositionType.UNKNOWN:                                  return True,  1.0
        if resume_type  == PositionType.UNKNOWN:                                  return False, 0.6
        if vacancy_type == resume_type:                                            return True,  1.0
        if vacancy_type == PositionType.FULLSTACK and resume_type in (
                PositionType.BACKEND, PositionType.FRONTEND):                     return True,  0.95
        if resume_type == PositionType.FULLSTACK and vacancy_type in (
                PositionType.BACKEND, PositionType.FRONTEND):                     return True,  0.85
        if {vacancy_type, resume_type} == {PositionType.BACKEND, PositionType.DEVOPS}:   return False, 0.6
        if {vacancy_type, resume_type} == {PositionType.BACKEND, PositionType.FRONTEND}: return False, 0.1
        if PositionType.MOBILE in (vacancy_type, resume_type) and (
            vacancy_type in (PositionType.BACKEND, PositionType.FRONTEND) or
            resume_type  in (PositionType.BACKEND, PositionType.FRONTEND)):       return False, 0.1
        non_tech = {PositionType.SECURITY, PositionType.MANAGEMENT}
        tech     = {PositionType.BACKEND, PositionType.FRONTEND, PositionType.FULLSTACK,
                    PositionType.DEVOPS, PositionType.MOBILE, PositionType.ML, PositionType.DATA}
        if vacancy_type in tech     and resume_type in non_tech: return False, 0.05
        if vacancy_type in non_tech and resume_type in tech:     return False, 0.3
        return False, 0.5

    @classmethod
    def _normalize_location(cls, loc: str) -> str:
        if not loc:
            return ""
        n = loc.lower().strip()
        if n in cls._LOCATION_ALIASES:
            return cls._LOCATION_ALIASES[n]
        for alias, canonical in cls._LOCATION_ALIASES.items():
            if alias in n:
                return canonical
        return n

    def _sanitize_experience(self, years: float) -> Tuple[float, Optional[str]]:
        if years < 0:
            return 0.0, f"Negative experience ({years}) replaced with 0"
        if years > self._MAX_EXPERIENCE_YEARS:
            return self._MAX_EXPERIENCE_YEARS, f"Implausible experience ({years:.1f}) capped"
        return years, None

    def _sanitize_salary(self, salary: Optional[float]) -> Tuple[Optional[float], Optional[str]]:
        if salary is None:               return None, None
        if salary < self._MIN_SALARY:    return None, f"Implausibly low salary ({salary}) ignored"
        if salary > self._MAX_SALARY:    return None, f"Implausibly high salary ({salary}) ignored"
        return salary, None

    def _normalize_hh_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if 'experience_years' in data and 'skills' in data:
            return data

        normalized: Dict[str, Any] = {
            'id':       data.get('id') or data.get('resume_id', 'unknown'),
            'warnings': [],
        }

        try:
            raw_exp = data.get('experience', '')
            if isinstance(raw_exp, dict):
                exp_str = raw_exp.get('total_experience', '') or ''
            elif isinstance(raw_exp, str):
                exp_str = raw_exp
            else:
                exp_str = str(raw_exp) if raw_exp else ''
            if not exp_str:
                exp_str = str(data.get('total_experience', '') or '')
            normalized['experience_years'] = self._parse_russian_experience(exp_str)
        except Exception as e:
            normalized['warnings'].append(f"Error parsing experience: {e}")
            normalized['experience_years'] = 0.0

        try:
            raw_skills = data.get('skills', [])
            if isinstance(raw_skills, dict):
                raw_skills = raw_skills.get('skills', [])
            if isinstance(raw_skills, str):
                normalized['skills'] = [s.strip() for s in re.split(r'[;,\n]', raw_skills) if s.strip()]
            elif raw_skills and isinstance(raw_skills[0], dict):
                normalized['skills'] = [s.get('name', '') for s in raw_skills if isinstance(s, dict)]
            else:
                normalized['skills'] = [str(s) for s in raw_skills if s]
            if not normalized['skills']:
                ts = data.get('technical_skills', '')
                if isinstance(ts, str):
                    normalized['skills'] = [s.strip() for s in re.split(r'[;,\n]', ts) if s.strip()]
        except Exception:
            normalized['skills'] = []

        edu = data.get('education', '')
        normalized['education'] = (
            edu.get('highest_level', '') if isinstance(edu, dict) else edu
        ) or data.get('education_level', '')

        try:
            if 'location' in data:
                normalized['location'] = data['location']
            elif 'location_city' in data:
                normalized['location'] = data['location_city']
            else:
                loc_data = data.get('personal_info', {})
                normalized['location'] = (
                    loc_data.get('location', {}).get('city', '')
                    if isinstance(loc_data, dict) else ''
                )
        except Exception:
            normalized['location'] = ''

        try:
            summary  = data.get('summary', '') or ''
            exp_text = data.get('experience_text', '') or ''
            if not summary:
                about = (
                    data.get('additional_info', {}).get('about_me', '')
                    if isinstance(data.get('additional_info'), dict)
                    else data.get('about_me', '')
                )
                summary = about or data.get('about_me', '') or ''
            if not exp_text:
                exp_text = data.get('work_experience', '') or ''
            if not exp_text:
                hh_exp = data.get('experience', {})
                if isinstance(hh_exp, dict):
                    chunks = []
                    for entry in hh_exp.get('experience', [])[:10]:
                        if not isinstance(entry, dict):
                            continue
                        period = (" ".join(entry.get('period', []))
                                  if isinstance(entry.get('period'), list)
                                  else str(entry.get('period', '')))
                        chunk = " | ".join(
                            p for p in [period, entry.get('company', ''),
                                        entry.get('position', ''),
                                        (" ".join(entry.get('description', []))
                                         if isinstance(entry.get('description'), list)
                                         else str(entry.get('description', '')))]
                            if p
                        )
                        if chunk:
                            chunks.append(chunk)
                    exp_text = " || ".join(chunks)
            normalized['experience_text'] = exp_text
            normalized['summary']         = summary[:500]
        except Exception:
            normalized['experience_text'] = ''
            normalized['summary']         = ''

        position = data.get('position', '')
        if not position:
            dp = data.get('desired_position', '')
            position = dp.get('title', '') if isinstance(dp, dict) else dp
        normalized['position'] = position

        salary_raw = (
            data.get('desired_salary')
            or data.get('salary_expectations')
            or data.get('salary', '')
        )
        try:
            if isinstance(salary_raw, (int, float)) and salary_raw > 0:
                normalized['desired_salary'] = float(salary_raw)
            elif isinstance(salary_raw, str) and salary_raw.strip():
                clean = salary_raw.replace('\xa0', '').replace(' ', '')
                nums  = re.findall(r'\d+', clean)
                if len(nums) >= 2:
                    normalized['desired_salary'] = (float(nums[0]) + float(nums[1])) / 2.0
                elif len(nums) == 1:
                    normalized['desired_salary'] = float(nums[0])
                else:
                    normalized['desired_salary'] = None
            else:
                normalized['desired_salary'] = None
        except (ValueError, IndexError):
            normalized['desired_salary'] = None

        wf  = data.get('work_schedule', '') or data.get('work_format', '')
        emp = data.get('employment_types', '') or data.get('employment_type', '')
        normalized['work_format'] = f"{wf} {emp}".strip()

        return normalized

    def _normalize_hh_vacancy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if 'title' in data and 'required_experience_years' in data:
            return data

        normalized: Dict[str, Any] = {}
        normalized['title']       = data.get('title', '') or data.get('name', '') or 'Unknown'
        normalized['description'] = re.sub(r'<[^>]+>', '', data.get('description', ''))

        k_skills = data.get('skills', []) or data.get('key_skills', [])
        if isinstance(k_skills, str):
            normalized['skills'] = [s.strip() for s in k_skills.split(',') if s.strip()]
        elif isinstance(k_skills, list) and k_skills and isinstance(k_skills[0], dict):
            normalized['skills'] = [s.get('name', '') for s in k_skills]
        else:
            normalized['skills'] = [str(s) for s in k_skills if s]

        salary = data.get('salary', {})
        if isinstance(salary, dict):
            normalized['salary_from']     = salary.get('from')
            normalized['salary_to']       = salary.get('to')
            normalized['salary_currency'] = salary.get('currency')
        else:
            normalized['salary_from'] = normalized['salary_to'] = normalized['salary_currency'] = None

        schedule   = data.get('schedule', {})
        employment = data.get('employment', {})
        normalized['work_format']    = schedule.get('name', '')   if isinstance(schedule,   dict) else str(schedule   or '')
        normalized['employment_type'] = employment.get('name', '') if isinstance(employment, dict) else str(employment or '')

        exp_raw = data.get('experience', {})
        exp_id  = (exp_raw.get('id', '') or exp_raw.get('name', '')
                   if isinstance(exp_raw, dict) else str(exp_raw))
        eid = exp_id.lower()
        if   'no' in eid or 'нет опыта' in eid or 'без опыта' in eid: normalized['required_experience_years'] = 0
        elif 'between1' in eid or ('1' in eid and '3' in eid):         normalized['required_experience_years'] = 1
        elif 'between3' in eid or ('3' in eid and '6' in eid):         normalized['required_experience_years'] = 3
        elif 'between6' in eid or ('6' in eid and 'лет' in eid):       normalized['required_experience_years'] = 6
        elif 'more' in eid or 'morethan6' in eid or 'более' in eid:    normalized['required_experience_years'] = 6
        else:
            nm = re.search(r'(\d+)', exp_id)
            normalized['required_experience_years'] = int(nm.group(1)) if nm else 1

        addr = data.get('address', {}); area = data.get('area', {}); loc = data.get('location', '')
        if   loc:                                   normalized['location'] = loc
        elif isinstance(addr, dict) and addr.get('city'): normalized['location'] = addr['city']
        elif isinstance(area, dict):                normalized['location'] = area.get('name', '')
        else:                                       normalized['location'] = ''

        return normalized

    # ── Semantic scoring ──────────────────────────────────────────────────────
    def _calculate_semantic_scores_batch(self, vacancy_text: str,
                                          resume_texts: List[str],
                                          batch_size: int = 0) -> List[float]:
        """Batch semantic scoring with embedding cache + LRU eviction.

        CHANGE 1: batch_size GPU 512 → 1024
        CHANGE 4: LRU eviction instead of full cache clear
        CHANGE 6: show_progress_bar=False on all encode calls
        """
        if not resume_texts:
            return []
        if not self.model or not vacancy_text:
            return [0.5] * len(resume_texts)

        # CHANGE 1: batch_size 1024 for GPU
        if batch_size == 0:
            device     = getattr(self, '_device', 'cpu')
            batch_size = 1024 if device == 'cuda' else 64

        try:
            # Vacancy embedding (cached)
            if vacancy_text not in self._embedding_cache:
                if len(self._embedding_cache) >= self._embedding_cache_max:
                    # CHANGE 4: LRU eviction — keep last 5000 instead of full clear
                    self._embedding_cache = dict(
                        list(self._embedding_cache.items())[-5000:]
                    )
                    logger.debug("Embedding cache: LRU eviction, kept last 5000")
                self._embedding_cache[vacancy_text] = self.model.encode(
                    [vacancy_text], convert_to_tensor=True,
                    show_progress_bar=False,  # CHANGE 6
                )[0]
            vacancy_emb = self._embedding_cache[vacancy_text]

            # Only encode texts not yet cached
            uncached = [t for t in resume_texts if t not in self._embedding_cache]
            if uncached:
                new_embs = self.model.encode(
                    uncached,
                    convert_to_tensor=True,
                    batch_size=batch_size,        # CHANGE 1+5
                    show_progress_bar=False,      # CHANGE 6
                )
                for txt, emb in zip(uncached, new_embs):
                    if len(self._embedding_cache) < self._embedding_cache_max:
                        self._embedding_cache[txt] = emb

            try:
                import torch as _torch
                resume_embs = _torch.stack([self._embedding_cache[t] for t in resume_texts])
                sims        = util.pytorch_cos_sim(vacancy_emb, resume_embs)[0]
                return [max(0.0, min(1.0, float(s))) for s in sims]
            except Exception:
                all_embs = self.model.encode(
                    resume_texts,
                    convert_to_tensor=False,
                    batch_size=batch_size,    # CHANGE 1+5
                    show_progress_bar=False,  # CHANGE 6
                )
                vac_np = vacancy_emb.numpy() if hasattr(vacancy_emb, 'numpy') else vacancy_emb
                import numpy as _np
                scores = []
                for emb in all_embs:
                    dot  = _np.dot(vac_np, emb)
                    norm = _np.linalg.norm(vac_np) * _np.linalg.norm(emb)
                    scores.append(float(dot / norm) if norm > 0 else 0.5)
                return [max(0.0, min(1.0, s)) for s in scores]
        except Exception:
            return [self._calculate_semantic_score(vacancy_text, t) for t in resume_texts]

    def _calculate_semantic_score(self, vacancy_text: str, resume_text: str) -> float:
        if not self.model or not vacancy_text or not resume_text:
            return 0.5
        try:
            embs  = self.model.encode(
                [vacancy_text, resume_text],
                convert_to_tensor=True,
                show_progress_bar=False,  # CHANGE 6
            )
            return max(0.0, min(1.0, util.pytorch_cos_sim(embs[0], embs[1]).item()))
        except Exception:
            return 0.5

    @staticmethod
    def _compact_text(parts: List[Any], limit: int = 500) -> str:
        text = " ".join(str(p).strip() for p in parts if p)
        return re.sub(r'\s+', ' ', text).strip()[:limit]

    # ── Scoring components ────────────────────────────────────────────────────
    def _calculate_experience_score(self, required_years: float,
                                     candidate_years: float) -> Tuple[float, str]:
        if required_years <= 0:
            return 1.0, "No experience requirement"
        if candidate_years <= 0:
            return 0.0, f"No experience listed (required: {required_years})"
        ratio = candidate_years / required_years
        if ratio >= 1.0:  return 1.0,          f"Meets requirement: {candidate_years:.1f}/{required_years:.1f} yrs"
        if ratio >= 0.5:  return round(ratio,2),f"Partial: {candidate_years:.1f}/{required_years:.1f} = {ratio:.2f}"
        if ratio >= 0.25: return 0.25,          f"Junior: {candidate_years:.1f}/{required_years:.1f} → 0.25"
        return 0.10,                             f"Very junior: {candidate_years:.1f}/{required_years:.1f} → 0.10"

    _SKILL_ALIASES: Dict[str, str] = {
        "spring boot": "spring",   "spring framework": "spring",
        "spring data": "spring",   "spring mvc": "spring",
        "spring cloud": "spring",  "spring security": "spring",
        "java core": "java",       "java se": "java",   "java ee": "java",
        "postgresql": "postgres",  "postgres": "postgres", "postgre": "postgres",
        "apache kafka": "kafka",
        "docker compose": "docker","docker-compose": "docker",
        "rest api": "rest",        "restful": "rest",   "restful api": "rest",
        "ci/cd": "cicd",           "ci cd": "cicd",     "continuous integration": "cicd",
        "microservices": "microservices",
        "микросервисы": "microservices", "микросервис": "microservices",
        "javascript": "js",        "js": "js",
        "typescript": "ts",        "ts": "ts",
        "react.js": "react",       "reactjs": "react",
        "vue.js": "vue",           "vuejs": "vue",
        "node.js": "nodejs",       "nodejs": "nodejs",
        "machine learning": "ml",  "deep learning": "dl",
        "scikit-learn": "sklearn", "scikit learn": "sklearn",
    }

    @staticmethod
    def _normalize_skill(skill: str) -> str:
        return re.sub(r'[.,:;!?]+$', '', skill.lower().strip())

    @classmethod
    def _canonicalize_skill(cls, skill: str) -> str:
        normed = cls._normalize_skill(skill)
        return cls._SKILL_ALIASES.get(normed, normed)

    def _calculate_skills_score(self, required_skills: List[str],
                                  candidate_skills: List[str]) -> Tuple[float, List[str], str]:
        if not required_skills: return 1.0, [], "No skills required"
        if not candidate_skills: return 0.0, [], "No skills listed"
        req_normed = [self._normalize_skill(s) for s in required_skills if s]
        cand_normed = [self._normalize_skill(s) for s in candidate_skills if s]
        if not req_normed: return 1.0, [], "Invalid required skills"
        req_canonical  = [self._canonicalize_skill(s) for s in required_skills if s]
        cand_canonical = {self._canonicalize_skill(s) for s in candidate_skills if s}
        matched: List[str] = []
        for idx, req_canon in enumerate(req_canonical):
            if req_canon in cand_canonical:
                matched.append(required_skills[idx]); continue
            req_norm = req_normed[idx]
            if any(req_norm in cn or cn in req_norm or
                   req_canon in cn or cn in req_canon
                   for cn in cand_normed):
                matched.append(required_skills[idx])
        score = len(matched) / len(req_normed)
        return score, matched[:5], f"Skills: {len(matched)}/{len(req_normed)} = {score:.2f}"

    def _calculate_education_score(self, education: str) -> Tuple[float, str]:
        if not education or not isinstance(education, str):
            return 0.5, "Education not specified"
        keywords = [
            'bachelor','master','phd','degree','university','college',
            'высшее','бакалавр','магистр','кандидат наук','доктор',
            'аспирант','колледж','техникум','училище',
            'институт','академия','mba','специалист',
        ]
        if any(k in education.lower() for k in keywords):
            return 1.0, f"Has higher education: {education[:50]}"
        return 0.5, f"Education type unclear: {education[:50]}"

    def _calculate_location_score(self, required_loc: str,
                                    candidate_loc: str) -> Tuple[float, str]:
        if not required_loc or not candidate_loc:
            return 1.0, "Location not specified"
        req_n  = self._normalize_location(required_loc)
        cand_n = self._normalize_location(candidate_loc)
        if cand_n == "remote":
            return 0.9, f"Candidate is remote (vacancy: {required_loc})"
        if req_n == cand_n or req_n in cand_n or cand_n in req_n:
            return 1.0, f"Location match: {candidate_loc}"
        return 0.3, f"Location mismatch: required {required_loc}, have {candidate_loc}"

    @staticmethod
    def _calculate_salary_fit(vacancy_from: Optional[float],
                               vacancy_to:   Optional[float],
                               candidate_salary: Optional[float]) -> Tuple[float, str]:
        if candidate_salary is not None:
            try:
                clean = str(candidate_salary).replace('\xa0', '').replace(' ', '')
                nums  = re.findall(r'\d+', clean)
                if len(nums) >= 2:
                    candidate_salary = (float(nums[0]) + float(nums[1])) / 2.0
                elif len(nums) == 1:
                    candidate_salary = float(nums[0])
                else:
                    candidate_salary = None
            except (ValueError, IndexError):
                candidate_salary = None

        if candidate_salary is None or candidate_salary <= 0:
            return 0.7, "Candidate salary expectations unknown"
        if vacancy_from is None and vacancy_to is None:
            return 0.7, "Vacancy salary not specified"

        low  = float(vacancy_from or 0)
        high = float(vacancy_to   or 0)
        if low > 0 and high <= 0:   high = low * 1.5
        elif high > 0 and low <= 0: low  = high * 0.6
        if low <= 0 and high <= 0:  return 0.7, "Vacancy salary not specified"

        if low <= candidate_salary <= high:
            return 1.0, f"In range: {candidate_salary} ∈ [{low}, {high}]"
        if candidate_salary < low:
            ratio = candidate_salary / low
            if ratio < 0.05:
                return 0.5, f"Suspiciously low ({candidate_salary}); possible data error"
            return 0.85, f"Candidate asks less than budget: {candidate_salary} < {low}"
        return max(0.1, high / candidate_salary), f"Over budget: {candidate_salary} > {high}"

    @staticmethod
    def _calculate_work_format_score(vacancy_format: str, vacancy_employment: str,
                                      candidate_format: str) -> Tuple[float, str]:
        if not vacancy_format and not vacancy_employment:
            return 1.0, "Work format not specified in vacancy"
        if not candidate_format:
            return 0.7, "Candidate format preferences unknown"
        v = (vacancy_format + " " + vacancy_employment).lower()
        c = candidate_format.lower()
        remote = {"удал", "remote", "дистанц"}
        office = {"офис", "office", "on-site", "полный день"}
        vr, vo = any(k in v for k in remote), any(k in v for k in office)
        cr, co = any(k in c for k in remote), any(k in c for k in office)
        if vr and cr: return 1.0, "Both remote"
        if vo and co: return 1.0, "Both office"
        if vr and co: return 0.5, "Vacancy remote, candidate prefers office"
        if vo and cr: return 0.5, "Vacancy office, candidate prefers remote"
        return 0.8, f"Format unclear: vacancy='{vacancy_format}', candidate='{candidate_format}'"

    def _filter_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        keys = ['gender','age','nationality','photo_url','address','phone',
                'email','name','пол','возраст','национальность','фото',
                'телефон','почта','имя']
        clean = data.copy()
        for k in keys: clean.pop(k, None)
        return clean

    # ── CSV parser ────────────────────────────────────────────────────────────
    def parse_csv_resumes(self, csv_content: str) -> List[Dict[str, Any]]:
        resumes = []
        try:
            reader = csv.DictReader(io.StringIO(csv_content))
            if not reader.fieldnames:
                raise ValueError("CSV has no header row")

            field_mapping = {
                'id':               ['id','resume_id','ID'],
                'position':         ['position','desired_position'],
                'experience':       ['experience','total_experience'],
                'skills':           ['skills','technical_skills'],
                'education':        ['education','education_level'],
                'location':         ['location','location_city'],
                'summary':          ['summary','about_me','description'],
                'experience_text':  ['experience_text','work_experience'],
                'desired_salary':   ['desired_salary','salary','salary_expectations'],
                'work_schedule':    ['work_schedule','work_format'],
                'employment_types': ['employment_types','employment_type'],
            }
            actual_fields: Dict[str, str] = {}
            for std, alts in field_mapping.items():
                for alt in alts:
                    if alt in reader.fieldnames:
                        actual_fields[std] = alt; break

            for row_idx, row in enumerate(reader, start=2):
                try:
                    def _col(key: str) -> str:
                        col = actual_fields.get(key)
                        return (row.get(col) or '') if col else ''

                    raw_skills    = _col('skills')
                    parsed_skills = (
                        [s.strip() for s in re.split(r'[;,\n]', raw_skills) if s.strip()]
                        if raw_skills else []
                    )
                    exp_str  = _col('experience')
                    exp_years = self._parse_russian_experience(exp_str)

                    resumes.append({
                        'id':               _col('id') or f'csv_row_{row_idx}',
                        'position':         _col('position'),
                        'experience':       exp_str,
                        'total_experience': exp_str,
                        'experience_years': exp_years,
                        'skills':           parsed_skills,
                        'technical_skills': raw_skills,
                        'education':        _col('education'),
                        'education_level':  _col('education'),
                        'location':         _col('location'),
                        'location_city':    _col('location'),
                        'summary':          _col('summary'),
                        'about_me':         _col('summary'),
                        'experience_text':  _col('experience_text'),
                        'work_experience':  _col('experience_text'),
                        'desired_salary':   _col('desired_salary'),
                        'work_schedule':    _col('work_schedule'),
                        'employment_types': _col('employment_types'),
                    })
                except Exception as e:
                    logger.warning("Failed to parse CSV row %d: %s", row_idx, e)
        except Exception as e:
            logger.error("Failed to parse CSV: %s", e)
            raise
        return resumes

    # ── Main ranking entry point ──────────────────────────────────────────────
    def process_batch(self, vacancy: Dict[str, Any],
                      resumes: List[Dict[str, Any]],
                      chunk_size: int = 1024) -> Dict[str, Any]:  # CHANGE 2: 256 → 1024
        """Rank resumes against a vacancy.

        CHANGE 2: chunk_size default 256 → 1024
        CHANGE 7: استدعاء process_batch(vacancy, resumes, chunk_size=1024)
        """
        if len(resumes) > chunk_size:
            logger.info("Large batch (%d resumes): chunking at %d", len(resumes), chunk_size)
            merged:     List[Dict] = []
            total_time: float      = 0.0
            last_meta:  Dict       = {}
            for i in range(0, len(resumes), chunk_size):
                partial    = self._process_batch_internal(vacancy, resumes[i: i + chunk_size])
                merged.extend(partial["ranked_list"])
                total_time += partial["meta"]["execution_time_sec"]
                last_meta   = partial["meta"]
            merged.sort(key=lambda x: x["score"], reverse=True)
            return {
                "meta": {
                    **last_meta,
                    "resumes_processed":  len(resumes),
                    "execution_time_sec": round(total_time, 4),
                    "chunks": (len(resumes) + chunk_size - 1) // chunk_size,
                },
                "ranked_list": merged,
            }
        return self._process_batch_internal(vacancy, resumes)

    def _process_batch_internal(self, vacancy: Dict[str, Any],
                                  resumes: List[Dict[str, Any]]) -> Dict[str, Any]:
        start_time    = time.time()
        clean_vacancy = self._normalize_hh_vacancy(vacancy)

        vacancy_text = self._compact_text([
            clean_vacancy.get('title', ''),
            clean_vacancy.get('description', '')[:300],
            ' '.join(clean_vacancy.get('skills', [])[:20]),
        ], limit=500)

        req_exp             = clean_vacancy.get('required_experience_years', 0)
        req_skills          = clean_vacancy.get('skills', [])
        req_loc             = clean_vacancy.get('location', '')
        vacancy_salary_from = clean_vacancy.get('salary_from')
        vacancy_salary_to   = clean_vacancy.get('salary_to')
        vacancy_work_format = clean_vacancy.get('work_format', '')
        vacancy_employment  = clean_vacancy.get('employment_type', '')

        vacancy_type = self._detect_position_type(clean_vacancy.get('title', ''))
        if vacancy_type == PositionType.UNKNOWN:
            vacancy_type = self._detect_position_type(
                clean_vacancy.get('title', '') + ' ' + clean_vacancy.get('description', '')
            )

        results:       List[RankResult] = []
        batch_warnings: List[str]       = []
        prepared:       List[Dict]      = []
        resume_texts:   List[str]       = []

        for resume in resumes:
            resume_warnings: List[str] = []
            try:
                norm = self._normalize_hh_format(resume)
                if norm.get('warnings'):
                    resume_warnings.extend(norm['warnings'])
                clean = self._filter_sensitive_data(norm)

                skills_list  = clean.get('skills', [])
                res_position = clean.get('position', '')
                summary      = clean.get('summary', '')
                exp_text     = clean.get('experience_text', '')

                raw_exp = float(clean.get('experience_years', 0.0) or 0.0)
                res_exp, exp_warn = self._sanitize_experience(raw_exp)
                if exp_warn: resume_warnings.append(exp_warn)

                resume_text = self._compact_text([
                    res_position, summary[:180],
                    ' '.join(skills_list[:20]), exp_text[:220],
                ])
                if not resume_text:
                    resume_warnings.append("Resume text is empty")
                    resume_text = " ".join(skills_list) if skills_list else "unknown"

                resume_type = self._detect_position_type(res_position)
                if resume_type == PositionType.UNKNOWN:
                    resume_type = self._detect_position_type(resume_text)

                raw_salary = clean.get('desired_salary')
                try:
                    raw_salary_f = float(raw_salary) if raw_salary not in (None, '') else None
                except (ValueError, TypeError):
                    raw_salary_f = None
                res_salary, sal_warn = self._sanitize_salary(raw_salary_f)
                if sal_warn: resume_warnings.append(sal_warn)

                prepared.append({
                    'clean_resume':    clean,
                    'resume_warnings': resume_warnings,
                    'res_exp':         res_exp,
                    'res_skills':      clean.get('skills', []),
                    'res_edu':         clean.get('education', ''),
                    'res_loc':         clean.get('location', ''),
                    'res_salary':      res_salary,
                    'res_work_format': clean.get('work_format', ''),
                    'resume_type':     resume_type,
                    'resume_text':     resume_text,
                })
                resume_texts.append(resume_text)

            except Exception as e:
                rid = resume.get('id', 'unknown') if isinstance(resume, dict) else 'unknown'
                logger.error("Resume %s: %s", rid, e)
                results.append(RankResult(
                    resume_id=rid, score=0.0, confidence=0.0,
                    explanation={"error": str(e)},
                    warnings=[f"Critical error: {e}"],
                ))

        semantic_scores = self._calculate_semantic_scores_batch(vacancy_text, resume_texts)

        for prep, sem_score in zip(prepared, semantic_scores):
            clean           = prep['clean_resume']
            resume_warnings = prep['resume_warnings']
            res_exp         = prep['res_exp']
            res_skills      = prep['res_skills']
            res_salary      = prep['res_salary']

            exp_score, exp_expl = self._calculate_experience_score(req_exp, res_exp)
            is_match, pos_mult  = self._is_position_type_match(vacancy_type, prep['resume_type'])
            pos_score           = (1.0 if is_match else 0.2) * pos_mult
            pos_expl            = (f"Vacancy: {vacancy_type.value}, "
                                   f"Resume: {prep['resume_type'].value}, "
                                   f"Multiplier: {pos_mult:.2f}")
            if not is_match and pos_mult < 0.5:
                resume_warnings.append(
                    f"Position type mismatch: {vacancy_type.value} vs {prep['resume_type'].value}"
                )

            sk_score, matched_sk, sk_expl = self._calculate_skills_score(req_skills, res_skills)
            sal_score, sal_expl = self._calculate_salary_fit(
                vacancy_salary_from, vacancy_salary_to, res_salary)
            wf_score,  wf_expl  = self._calculate_work_format_score(
                vacancy_work_format, vacancy_employment, prep['res_work_format'])
            edu_score, edu_expl = self._calculate_education_score(prep['res_edu'])
            loc_score, loc_expl = self._calculate_location_score(req_loc, prep['res_loc'])

            total = (
                sem_score  * self.weights['semantic']     +
                sk_score   * self.weights['hard_skills']  +
                exp_score  * self.weights['experience']   +
                pos_score  * self.weights['position_fit'] +
                sal_score  * self.weights['salary_fit']   +
                wf_score   * self.weights['work_format']  +
                edu_score  * self.weights['education']    +
                loc_score  * self.weights['location']
            )
            if pos_mult < 1.0:
                total *= pos_mult

            explanation = {
                "total_score": round(total, 4),
                "components": {
                    "semantic":     {"value": round(sem_score,  2), "weight": self.weights['semantic'],     "contribution": round(sem_score  * self.weights['semantic'],    4)},
                    "skills":       {"value": round(sk_score,   2), "weight": self.weights['hard_skills'],  "contribution": round(sk_score   * self.weights['hard_skills'], 4), "matched": matched_sk[:3], "explanation": sk_expl},
                    "experience":   {"value": round(exp_score,  2), "weight": self.weights['experience'],   "contribution": round(exp_score  * self.weights['experience'],  4), "explanation": exp_expl},
                    "position_fit": {"value": round(pos_score,  2), "weight": self.weights['position_fit'], "contribution": round(pos_score  * self.weights['position_fit'],4), "explanation": pos_expl},
                    "salary_fit":   {"value": round(sal_score,  2), "weight": self.weights['salary_fit'],   "contribution": round(sal_score  * self.weights['salary_fit'],  4), "explanation": sal_expl},
                    "work_format":  {"value": round(wf_score,   2), "weight": self.weights['work_format'],  "contribution": round(wf_score   * self.weights['work_format'], 4), "explanation": wf_expl},
                    "education":    {"value": round(edu_score,  2), "weight": self.weights['education'],    "contribution": round(edu_score  * self.weights['education'],   4), "explanation": edu_expl},
                    "location":     {"value": round(loc_score,  2), "weight": self.weights['location'],     "contribution": round(loc_score  * self.weights['location'],    4), "explanation": loc_expl},
                },
                "summary": ("Excellent match" if total > 0.8 else
                            "Good match"      if total > 0.6 else
                            "Moderate match"  if total > 0.4 else
                            "Poor match"      if total > 0.2 else
                            "Not suitable"),
            }

            results.append(RankResult(
                resume_id=clean.get('id', 'unknown'),
                score=total, confidence=0.0,
                explanation=explanation, warnings=resume_warnings,
            ))
            if resume_warnings:
                batch_warnings.append(
                    f"Resume {clean.get('id')}: {'; '.join(resume_warnings)}"
                )

        results.sort(key=lambda x: x.score, reverse=True)

        scores = [r.score for r in results]
        if len(scores) >= 3:
            mean_s = float(np.mean(scores))
            std_s  = float(np.std(scores))
            if std_s < 1e-6:
                for r in results: r.confidence = 0.5
            else:
                for r in results:
                    z = max(-3.0, min(3.0, (r.score - mean_s) / std_s))
                    r.confidence = round(1.0 / (1.0 + np.exp(-z)), 3)
        else:
            for r in results: r.confidence = 0.5

        return {
            "meta": {
                "model_version":      "v3.0-stable",
                "model_name":         self.model_name,
                "seed":               self.seed,
                "execution_time_sec": round(time.time() - start_time, 4),
                "resumes_processed":  len(resumes),
                "vacancy_title":      clean_vacancy.get('title', ''),
                "batch_warnings":     batch_warnings,
            },
            "ranked_list": [
                {
                    "resume_id":   r.resume_id,
                    "score":       round(r.score, 4),
                    "confidence":  round(r.confidence, 2),
                    "explanation": r.explanation,
                    "warnings":    r.warnings,
                }
                for r in results
            ],
        }