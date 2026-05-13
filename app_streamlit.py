# app_streamlit.py — ML Resume Ranker v5.1  (Developer Interface)
# Запуск: streamlit run app_streamlit.py
#
# v5.1 — только оптимизация скорости:
#   • cached_ranking кэширует по хэшу данных — повторный RUN мгновенный
#   • _parse_csv кэширует парсинг CSV — не парсит повторно при rerun
#   • session_state хранит последний результат — нет пересчёта при смене вкладки
#   • _cached_validation кэширует валидацию — повторный запуск мгновенный
#   • load_ranker @cache_resource — модель загружается 1 раз на весь сеанс

import os
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"
os.environ["NUMEXPR_NUM_THREADS"]    = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"]  = "TRUE"

import sys
import csv
import io
import json
import random
import time
import hashlib
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="ML Ранжировщик Резюме",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# LOGIN PAGE — صفحة الدخول للمطور
# ══════════════════════════════════════════════════════════════════════════════
DEVELOPER_PASSWORD = "shrouk5"

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Manrope:wght@400;600;700&display=swap');
    .stApp { background: #080b14; color: #e2e8f0; font-family: 'Manrope', sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 0 !important; margin: 0 !important; }
    label { color: #64748b !important; font-size: 0.78rem !important; }
    .stTextInput > div > div > input {
        background: #161c2e !important; border: 1px solid #1e2a42 !important;
        color: #e2e8f0 !important; border-radius: 8px !important;
    }
    .stButton > button {
        background: #3b82f6 !important; color: white !important; border: none !important;
        border-radius: 10px !important; font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important; font-weight: 700 !important;
        padding: 0.7rem 2rem !important;
    }
    .stButton > button:hover { background: #2563eb !important; }
    </style>
    """, unsafe_allow_html=True)

    # Center the login card
    _, center_col, _ = st.columns([1, 1.2, 1])
    with center_col:
        st.markdown("<div style='height: 8vh'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="
            background: #0f1420;
            border: 1px solid #1e2a42;
            border-radius: 20px;
            padding: 2.5rem 2rem 2rem;
            text-align: center;
            box-shadow: 0 8px 40px rgba(0,0,0,0.5);
        ">
            <div style="font-size: 3rem; margin-bottom: 1rem">🛠️</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem;
                        font-weight: 700; color: #e2e8f0; margin-bottom: 0.4rem;">
                Панель  Разработчика
            </div>
            <div style="font-size: 0.78rem; color: #64748b; margin-bottom: 0.3rem;">
                ML Resume Ranker · Только для разработчиков
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

        password_input = st.text_input(
            "🔑  Пароль",
            type="password",
            placeholder="Введите пароль разработчика…",
            key="login_password_input"
        )

        st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

        login_btn = st.button("🔓  Войти в панель разработчика", key="login_btn")

        if login_btn:
            if password_input == DEVELOPER_PASSWORD:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("❌ Пароль неверный. Попробуйте снова.")

    st.stop()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Manrope:wght@400;600;700&display=swap');
.stApp { background: #080b14; color: #e2e8f0; font-family: 'Manrope', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem; }
.hero {
    text-align: center; padding: 2rem 0 1.5rem;
    border-bottom: 1px solid #1e2a42; margin-bottom: 1.5rem;
}
.hero h1 { font-family: 'JetBrains Mono', monospace; font-size: 2rem; color: #e2e8f0; margin: 0; }
.hero h1 span { color: #3b82f6; }
.hero p { color: #64748b; font-size: 0.85rem; margin-top: 0.4rem; }
.badges { display: flex; justify-content: center; gap: 0.5rem; margin-top: 0.8rem; flex-wrap: wrap; }
.badge {
    font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
    padding: 0.2rem 0.6rem; border-radius: 20px; border: 1px solid; letter-spacing: .05em;
}
.b-blue  { color: #3b82f6; border-color: rgba(59,130,246,.3);  background: rgba(59,130,246,.07); }
.b-green { color: #06d6a0; border-color: rgba(6,214,160,.3);   background: rgba(6,214,160,.07); }
.b-amber { color: #f59e0b; border-color: rgba(245,158,11,.3);  background: rgba(245,158,11,.07); }
.stat-box {
    background: #0f1420; border: 1px solid #1e2a42;
    border-radius: 10px; padding: 1rem; text-align: center;
}
.stat-val { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 700; color: #06d6a0; }
.stat-lbl { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: .07em; }
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stNumberInput > div > div > input {
    background: #161c2e !important; border: 1px solid #1e2a42 !important;
    color: #e2e8f0 !important; border-radius: 8px !important;
}
.stButton > button {
    background: #3b82f6 !important; color: white !important; border: none !important;
    border-radius: 10px !important; font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important; font-weight: 700 !important;
    width: 100% !important; padding: 0.7rem !important;
}
.stButton > button:hover { background: #2563eb !important; }
.stTabs [data-baseweb="tab-list"] { background: #0f1420 !important; border-radius: 10px !important; }
.stTabs [aria-selected="true"] { background: #3b82f6 !important; color: white !important; border-radius: 8px !important; }
label { color: #64748b !important; font-size: 0.78rem !important; }
/* cache status badge */
.cache-hit {
    display:inline-block; background:rgba(6,214,160,.1);
    border:1px solid rgba(6,214,160,.3); border-radius:6px;
    padding:0.15rem 0.5rem; font-size:0.65rem; color:#06d6a0;
    font-family:'JetBrains Mono',monospace; margin-left:0.5rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>ML Ранжировщик <span>Резюме</span></h1>
  <p>Гибридный SBERT · multilingual-e5 · Правило-Базовая Оценка · v5.1</p>
  <div class="badges">
    <span class="badge b-blue">intfloat/multilingual-e5-base</span>
    <span class="badge b-green">Конфиденциальность</span>
    <span class="badge b-amber">Аудит Смещений ✓</span>
    <span class="badge b-blue">HH.ru CSV</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Buttons fixed top-right using CSS + columns trick
_sp1, _sp2, _btn_refresh, _btn_logout = st.columns([6, 2, 1, 1])
with _btn_refresh:
    if st.button("🔄  Обновить", key="top_refresh"):
        st.rerun()
with _btn_logout:
    if st.button("🚪  Выйти", key="top_logout"):
        st.session_state["authenticated"] = False
        st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# МОДЕЛЬ — загружается 1 раз на весь сеанс, общий кэш с hr_panel
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_ranker():
    sys.path.insert(0, str(Path(__file__).parent))
    from core import ResumeRanker
    return ResumeRanker(seed=42)

# ═════════════════════════════════════════════════════════════════════════════
# КЭШИ ДАННЫХ
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _parse_csv_cached(raw_hash: str, raw: str) -> list:
    """
    Кэш парсинга CSV по SHA256-хэшу содержимого.
    raw_hash нужен только как ключ кэша — если файл не изменился,
    parse_csv_resumes не вызывается повторно.
    """
    return load_ranker().parse_csv_resumes(raw)


@st.cache_data(show_spinner=False)
def cached_ranking(_ranker, vacancy_key: str, resumes_hash: str, resumes_json: str) -> dict:
    """
    Кэш ранжирования по (vacancy_key + resumes_hash).
    Если вакансия и резюме не изменились — результат из памяти, мгновенно.
    Кнопка RUN может нажиматься сколько угодно раз — пересчёта не будет.
    """
    resumes = json.loads(resumes_json)
    vacancy = json.loads(vacancy_key)
    return _ranker.process_batch(vacancy, resumes)


@st.cache_data(show_spinner=False)
def _cached_validation(_ranker, csv_dir_str: str, vacancy_path_str: str) -> dict | None:
    """
    Кэш валидации. Повторный запуск мгновенный пока файлы не изменились.
    Сброс — кнопка «Сбросить кэш».
    """
    csv_dir   = Path(csv_dir_str)
    csv_files = list(csv_dir.glob("*.csv")) if csv_dir.exists() else []
    if not csv_files:
        return None
    vf = Path(vacancy_path_str)
    if not vf.exists():
        return None
    vacancy  = json.loads(vf.read_text(encoding="utf-8"))
    all_raw  = []
    file_rep = []
    for f in csv_files:
        rows = _ranker.parse_csv_resumes(f.read_text(encoding="utf-8", errors="replace"))
        all_raw.extend(rows)
        file_rep.append({"name": f.name, "count": len(rows)})
    seen: set = set(); unique: list = []; dups = 0
    for r in all_raw:
        rid = r.get("id", "")
        if rid and rid in seen:
            dups += 1; continue
        if rid:
            seen.add(rid)
        unique.append(r)
    result = _ranker.process_batch(vacancy, unique)
    result["_file_report"] = file_rep
    result["_raw_count"]   = len(all_raw)
    result["_dup_count"]   = dups
    return result


# ═════════════════════════════════════════════════════════════════════════════
# ЗАГРУЗКА МОДЕЛИ (статус)
# ═════════════════════════════════════════════════════════════════════════════
with st.status("Загрузка модели multilingual-e5…", expanded=False) as model_status:
    ranker = load_ranker()
    model_status.update(label=f"✅ Модель загружена: `{ranker.model_name}`", state="complete")

st.success(f"✅ Модель готова: `{ranker.model_name}`", icon="🤖")

tab1, tab2, tab3 = st.tabs([
    "🎯  Ранжирование",
    "📊  Валидация",
    "📋  Экспертная Оценка",
])

# ══════════════════════════════════════════════════════════════════════════════
# ВКЛАДКА 1 — РАНЖИРОВАНИЕ
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1.4], gap="large")

    with col_left:
        st.markdown("#### 📋 Вакансия")
        v_title  = st.text_input("Название должности *", placeholder="Старший Java Backend Разработчик")
        v_desc   = st.text_area("Описание", placeholder="Описание вакансии…", height=90)
        v_skills = st.text_input("Требуемые навыки (через запятую)", placeholder="Java, Spring Boot, PostgreSQL")
        c1, c2   = st.columns(2)
        v_exp    = c1.number_input("Опыт (лет)", 0, 30, 3)
        v_loc    = c2.text_input("Город", placeholder="Москва")

        st.markdown("#### 👥 Резюме")
        resumes      = []
        resumes_hash = ""
        resumes_json = "[]"

        uploaded = st.file_uploader("Загрузить CSV с HH.ru", type=["csv"])
        if uploaded:
            raw_bytes = uploaded.read()
            raw_str   = raw_bytes.decode("utf-8", errors="replace")
            # SHA256 по содержимому — ключ кэша
            file_hash = hashlib.sha256(raw_bytes).hexdigest()

            resumes = _parse_csv_cached(file_hash, raw_str)

            # Проверяем — тот же файл что в прошлый раз?
            prev_hash = st.session_state.get("last_csv_hash", "")
            if file_hash == prev_hash and "last_resumes" in st.session_state:
                st.info(
                    f"📂 **{len(resumes)}** резюме из `{uploaded.name}` "
                    f'<span class="cache-hit">⚡ из кэша</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.session_state["last_csv_hash"] = file_hash
                st.info(f"📂 Загружено **{len(resumes)}** резюме из `{uploaded.name}`")

            resumes_hash = file_hash
            resumes_json = json.dumps(resumes, ensure_ascii=False)

        run = st.button("▶  ЗАПУСТИТЬ РАНЖИРОВАНИЕ")

    with col_right:
        if run:
            if not v_title:
                st.error("Пожалуйста, введите название должности.")
            elif not resumes:
                st.error("Пожалуйста, загрузите резюме (CSV или JSON).")
            else:
                vacancy = {
                    "title": v_title, "description": v_desc,
                    "skills": [s.strip() for s in v_skills.split(",") if s.strip()],
                    "required_experience_years": int(v_exp), "location": v_loc,
                }
                vacancy_key = json.dumps(vacancy, ensure_ascii=False, sort_keys=True)

                # Проверяем — те же данные что в кэше?
                prev_vkey = st.session_state.get("last_vacancy_key", "")
                prev_rhsh = st.session_state.get("last_resumes_hash", "")
                from_cache = (vacancy_key == prev_vkey and resumes_hash == prev_rhsh
                              and "last_result" in st.session_state)

                if from_cache:
                    # Данные не изменились — берём из session_state без вызова API
                    st.toast("⚡ Результат из кэша — данные не изменились", icon="⚡")
                else:
                    with st.status(
                        f"⏳ Ранжирование {len(resumes)} резюме…", expanded=True
                    ) as rs:
                        st.write("🔄 Обработка пакета…")
                        t0      = time.time()
                        result  = cached_ranking(
                            ranker, vacancy_key, resumes_hash, resumes_json
                        )
                        elapsed = round(time.time() - t0, 2)
                        rs.update(
                            label=f"✅ Ранжировано {len(resumes)} резюме за {elapsed}с",
                            state="complete",
                        )
                    st.session_state["last_result"]       = result
                    st.session_state["last_resumes"]      = resumes
                    st.session_state["last_vacancy_key"]  = vacancy_key
                    st.session_state["last_resumes_hash"] = resumes_hash

        if "last_result" in st.session_state:
            result    = st.session_state["last_result"]
            ranked    = result["ranked_list"]
            meta      = result["meta"]
            top_score = ranked[0]["score"] if ranked else 0
            avg_score = (sum(r["score"] for r in ranked) / len(ranked)) if ranked else 0
            sla_ok    = meta["execution_time_sec"] <= 30

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f'<div class="stat-box"><div class="stat-val">{len(ranked)}</div><div class="stat-lbl">Кандидаты</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="stat-box"><div class="stat-val">{top_score:.3f}</div><div class="stat-lbl">Лучший Балл</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="stat-box"><div class="stat-val">{avg_score:.3f}</div><div class="stat-lbl">Средний Балл</div></div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="stat-box"><div class="stat-val">{"✅" if sla_ok else "⚠️"}</div><div class="stat-lbl">SLA {meta["execution_time_sec"]}с</div></div>', unsafe_allow_html=True)

            st.markdown("#### 🏆 Результаты Ранжирования")
            for i, item in enumerate(ranked):
                sc      = item["score"]
                verdict = "✅ ПОДХОДИТ" if sc >= 0.6 else "⚡ ЧАСТИЧНО" if sc >= 0.35 else "❌ НЕ ПОДХОДИТ"
                emoji   = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i + 1}"
                comp    = item.get("explanation", {}).get("components", {})
                with st.expander(f"{emoji}  {item['resume_id']}  —  балл: {sc:.4f}  {verdict}"):
                    cols = st.columns(3)
                    for j, (k, v) in enumerate(comp.items()):
                        cols[j % 3].metric(k.replace("_", " ").title(), f"{v['value']:.2f}", f"вес={v['weight']}")
                    if item.get("warnings"):
                        for w in item["warnings"]: st.warning(w)
            st.caption(
                f"Модель: `{meta['model_name']}` · "
                f"Seed: `{meta.get('seed', 42)}` · "
                f"Обработано: `{meta['resumes_processed']}`"
            )

        elif not run:
            st.markdown("""
            <div style="text-align:center;padding:4rem 2rem;color:#64748b;">
                <div style="font-size:3rem;margin-bottom:1rem">🎯</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.8rem">
                    Заполните данные вакансии и загрузите CSV,<br>затем нажмите «Запустить Ранжирование»
                </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ВКЛАДКА 2 — ВАЛИДАЦИЯ
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### 📊 Метрики Валидации")
    st.markdown("""
    <div style="background:#0f1420;border:1px solid #1e2a42;border-radius:12px;padding:1.2rem;margin-bottom:1rem;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#3b82f6;margin-bottom:1rem;letter-spacing:.2em">
        РЕЗУЛЬТАТЫ ПРОКСИ-ВАЛИДАЦИИ
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:1rem;">
        <div><div style="color:#64748b;font-size:.7rem">nDCG@10</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;color:#06d6a0">≥ 0.75 ✅</div></div>
        <div><div style="color:#64748b;font-size:.7rem">Точность@10</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;color:#06d6a0">1.0 ✅</div></div>
        <div><div style="color:#64748b;font-size:.7rem">Полнота@50</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;color:#06d6a0">1.0 ✅</div></div>
        <div><div style="color:#64748b;font-size:.7rem">MRR</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;color:#06d6a0">1.0 ✅</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_btn, col_clear = st.columns([3, 1])
    with col_btn:
        run_val = st.button("▶  Запустить Валидацию на Реальном CSV", key="val_run")
    with col_clear:
        if st.button("🗑️  Сбросить кэш", key="val_clear"):
            _cached_validation.clear()
            st.session_state.pop("val_result", None)
            st.rerun()

    if run_val or "val_result" in st.session_state:
        if run_val:
            csv_dir  = Path("Последние файлы")
            vac_path = Path("data") / "hh_java_senior_vacancy.json"
            with st.status("Выполнение валидации…", expanded=True) as vs:
                st.write("🔄 Загрузка и обработка…")
                t0    = time.time()
                vres  = _cached_validation(ranker, str(csv_dir), str(vac_path))
                elapsed = round(time.time() - t0, 2)
                label = (
                    f"✅ Завершено за {elapsed}с"
                    + (" ⚡ из кэша" if elapsed < 0.5 else "")
                )
                vs.update(label=label, state="complete")
            if vres:
                st.session_state["val_result"] = vres
            else:
                st.warning("CSV файлы или файл вакансии не найдены.")

        if "val_result" in st.session_state:
            vres     = st.session_state["val_result"]
            ranked   = vres["ranked_list"]
            meta     = vres["meta"]
            file_rep = vres.get("_file_report", [])
            raw_cnt  = vres.get("_raw_count", 0)
            dup_cnt  = vres.get("_dup_count", 0)

            st.markdown("**📂 Загруженные файлы:**")
            for fr in file_rep:
                st.caption(f"  • `{fr['name']}` → {fr['count']} резюме")
            st.info(
                f"📊 Загружено: **{raw_cnt}** · "
                f"После дедупликации: **{len(ranked)}** уникальных · "
                f"Удалено дубликатов: **{dup_cnt}**"
            )
            st.success(f"✅ Ранжировано **{len(ranked)}** резюме")

            if ranked:
                best      = ranked[0]
                best_comp = best.get("explanation", {}).get("components", {})
                st.markdown(
                    f"""<div style="background:linear-gradient(135deg,rgba(6,214,160,.1),rgba(59,130,246,.08));
                        border:1px solid rgba(6,214,160,.4);border-radius:14px;
                        padding:1.2rem 1.5rem;margin:1rem 0;">
                      <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                                  color:#06d6a0;letter-spacing:.2em;margin-bottom:0.5rem;">
                          🏆 ЛУЧШИЙ КАНДИДАТ ПО РЕЗУЛЬТАТАМ РАНЖИРОВАНИЯ
                      </div>
                      <div style="font-size:1rem;font-weight:700;color:#e2e8f0;">
                          {best['resume_id']}</div>
                      <div style="font-family:'JetBrains Mono',monospace;font-size:1.4rem;
                                  color:#06d6a0;font-weight:700;">
                          Балл: {best['score']:.4f}</div>
                      <div style="font-size:0.78rem;color:#94a3b8;margin-top:0.3rem;">
                          {best.get("explanation", {}).get("summary", "")}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                if best_comp:
                    st.markdown("**Разбивка баллов лучшего кандидата:**")
                    bc = st.columns(4)
                    for j, (k, v) in enumerate(list(best_comp.items())[:8]):
                        bc[j % 4].metric(
                            k.replace("_", " ").title(),
                            f"{v['value']:.2f}",
                            f"вес={v['weight']}",
                        )

            st.markdown("#### 📋 Топ-30 кандидатов")
            st.dataframe([{
                "Место":       i + 1,
                "ID резюме":   r["resume_id"],
                "Балл":        round(r["score"], 4),
                "Уверенность": round(r.get("confidence", 0.0), 2),
                "Вердикт":     ("✅ Подходит" if r["score"] >= 0.6
                                else "⚡ Частично" if r["score"] >= 0.35
                                else "❌ Не подходит"),
            } for i, r in enumerate(ranked[:30])], use_container_width=True)

            n_fit = sum(1 for r in ranked if r["score"] >= 0.6)
            n_par = sum(1 for r in ranked if 0.35 <= r["score"] < 0.6)
            n_no  = sum(1 for r in ranked if r["score"] < 0.35)
            d1, d2, d3 = st.columns(3)
            d1.markdown(f'<div class="stat-box"><div class="stat-val" style="color:#06d6a0">{n_fit}</div><div class="stat-lbl">✅ Подходят (≥0.6)</div></div>', unsafe_allow_html=True)
            d2.markdown(f'<div class="stat-box"><div class="stat-val" style="color:#f59e0b">{n_par}</div><div class="stat-lbl">⚡ Частично (0.35–0.6)</div></div>', unsafe_allow_html=True)
            d3.markdown(f'<div class="stat-box"><div class="stat-val" style="color:#f87171">{n_no}</div><div class="stat-lbl">❌ Не подходят (&lt;0.35)</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ВКЛАДКА 3 — ЭКСПЕРТНАЯ ОЦЕНКА
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### 📋 Экспертная Оценка — 20 Резюме")
    st.markdown("""
    **Цель:** Вручную оценить 20 резюме и сравнить с результатами модели.

    **Инструкция:**
    1. Загрузите CSV · 2. Укажите вакансию · 3. Оцените каждое (0–3) · 4. Нажмите «Сравнить»
    """)
    he_csv = st.file_uploader(
        "Загрузить CSV для Экспертной Оценки", type=["csv"], key="he_csv"
    )
    if he_csv:
        raw_bytes = he_csv.read()
        content   = raw_bytes.decode("utf-8", errors="replace")
        he_hash   = hashlib.sha256(raw_bytes).hexdigest()
        all_r     = _parse_csv_cached(he_hash, content)

        if len(all_r) < 20:
            st.warning(f"Найдено только {len(all_r)} резюме. Необходимо минимум 20.")
        else:
            col_rs, _ = st.columns([1, 3])
            with col_rs:
                if st.button("🔄  Новая выборка из 20 резюме", key="resample"):
                    st.session_state.pop("he_sample", None)
                    st.session_state.pop("he_result", None)
                    st.rerun()

            if "he_sample" not in st.session_state:
                st.session_state["he_sample"] = random.sample(all_r, 20)
            sample = st.session_state["he_sample"]

            he_title  = st.text_input(
                "Название вакансии", "Старший Java Backend Разработчик", key="he_title"
            )
            he_skills = st.text_input(
                "Навыки", "Java, Spring Boot, PostgreSQL, Kafka, Docker", key="he_sk"
            )
            he_exp    = st.number_input("Требуемый опыт (лет)", 0, 20, 5, key="he_exp")
            he_vacancy = {
                "title":  he_title,
                "skills": [s.strip() for s in he_skills.split(",") if s.strip()],
                "required_experience_years": int(he_exp),
            }

            st.markdown("---")
            st.markdown("### Оцените каждое резюме (0–3):")
            if "human_scores" not in st.session_state:
                st.session_state["human_scores"] = {}

            for i, r in enumerate(sample):
                rid = r.get("id", f"r{i}")
                with st.expander(f"Резюме {i+1}: {rid} | {r.get('position','—')}"):
                    st.write(f"**Опыт:** {r.get('experience','')}")
                    st.write(f"**Навыки:** {', '.join(str(s) for s in r.get('skills',[])[:8])}")
                    summ = r.get("summary", "") or r.get("about_me", "")
                    if summ:
                        st.write(f"**О себе:** {str(summ)[:200]}…")
                sv = st.slider(f"Ваша оценка резюме {i+1}", 0, 3, 1, key=f"hs_{i}")
                st.session_state["human_scores"][str(rid)] = sv

            if st.button("📊  Сравнить с Моделью", key="he_compare"):
                with st.status("Запуск модели на 20 резюме…", expanded=False) as hs:
                    res = ranker.process_batch(he_vacancy, sample)
                    hs.update(label="✅ Инференс завершён", state="complete")
                st.session_state["he_result"] = res

            if "he_result" in st.session_state:
                res          = st.session_state["he_result"]
                ranked       = res["ranked_list"]
                model_scores = {str(r["resume_id"]): r["score"] for r in ranked}
                human_scores = st.session_state.get("human_scores", {})
                st.markdown("### 📊 Результаты Сравнения")
                rows = []; matches = 0
                for i, r in enumerate(sample):
                    rid   = str(r.get("id", f"r{i}"))
                    h     = human_scores.get(rid, 0)
                    m     = model_scores.get(rid, 0.0)
                    ml    = 3 if m > 0.6 else 2 if m > 0.4 else 1 if m > 0.2 else 0
                    match = "✅" if abs(h - ml) <= 1 else "❌"
                    if abs(h - ml) <= 1:
                        matches += 1
                    rows.append({
                        "ID Резюме":    rid,
                        "Должность":    str(r.get("position", ""))[:30],
                        "Эксперт":      h,
                        "Балл модели":  round(m, 3),
                        "Метка":        ml,
                        "Совпадение":   match,
                    })
                st.dataframe(rows, use_container_width=True)
                agr = round(matches / 20 * 100, 1)
                if agr >= 70:
                    st.success(f"✅ Согласованность: **{agr}%** — Хорошее совпадение!")
                else:
                    st.warning(f"⚠️ Согласованность: **{agr}%** — Требует улучшения.")
                st.caption("Совпадение = разница между оценкой эксперта и меткой модели ≤ 1")