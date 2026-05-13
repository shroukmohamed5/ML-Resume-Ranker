# pages/hr_panel.py — Панель HR v4.1
# ─────────────────────────────────────────────────
# ЗАПУСК: streamlit run app_streamlit.py
#
# v4.1 — добавлена кнопка копирования ID кандидата

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
import time
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Панель HR — Resume Ranker",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═════════════════════════════════════════════════════════════════════════════
# CSS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Manrope:wght@400;600;700&display=swap');

.stApp { background: #080b14; color: #e2e8f0; font-family: 'Manrope', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem; }

.hr-hero {
    background: linear-gradient(135deg,#0f1420,#0d1528);
    border: 1px solid #1e2a42; border-radius: 16px;
    padding: 1.6rem 2rem; margin-bottom: 1.5rem;
    display: flex; align-items: center; gap: 1.4rem;
}
.hr-hero-title {
    font-family:'JetBrains Mono',monospace; font-size:1.5rem;
    font-weight:700; color:#e2e8f0; margin:0 0 0.2rem;
}
.hr-hero-title span { color:#3b82f6; }
.hr-hero-sub  { color:#64748b; font-size:0.78rem; }
.badges { display:flex; gap:0.5rem; margin-top:0.5rem; flex-wrap:wrap; }
.badge  {
    font-family:'JetBrains Mono',monospace; font-size:0.6rem;
    padding:0.15rem 0.5rem; border-radius:20px; border:1px solid; letter-spacing:.05em;
}
.b-blue  { color:#3b82f6; border-color:rgba(59,130,246,.3);  background:rgba(59,130,246,.07); }
.b-green { color:#06d6a0; border-color:rgba(6,214,160,.3);   background:rgba(6,214,160,.07); }
.b-amber { color:#f59e0b; border-color:rgba(245,158,11,.3);  background:rgba(245,158,11,.07); }
.b-red   { color:#f87171; border-color:rgba(248,113,113,.3); background:rgba(248,113,113,.07); }

.stat-box {
    background:#0f1420; border:1px solid #1e2a42;
    border-radius:10px; padding:0.9rem 0.5rem; text-align:center;
}
.stat-val { font-family:'JetBrains Mono',monospace; font-size:1.4rem; font-weight:700; color:#06d6a0; }
.stat-lbl { font-size:0.62rem; color:#64748b; text-transform:uppercase; letter-spacing:.07em; margin-top:0.2rem; }

.dash-card {
    background:#0f1420; border:1px solid #1e2a42; border-radius:12px;
    padding:1.1rem 1.3rem; margin-bottom:0.8rem;
}
.dash-card-title {
    font-family:'JetBrains Mono',monospace; font-size:0.62rem;
    color:#3b82f6; letter-spacing:.18em; text-transform:uppercase; margin-bottom:0.7rem;
}

.comp-card {
    background:#0d1220; border:1px solid #1e2a42; border-radius:8px;
    padding:0.7rem 1rem; margin-bottom:0.5rem;
}
.comp-card-title {
    font-family:'JetBrains Mono',monospace; font-size:0.7rem;
    font-weight:700; color:#e2e8f0; margin-bottom:0.25rem;
}
.comp-card-body    { font-size:0.72rem; color:#64748b; line-height:1.5; }
.comp-card-matched { font-size:0.68rem; color:#06d6a0; margin-top:0.2rem; }

/* ── ID-блок с кнопкой копирования ── */
.id-block {
    display:flex; align-items:center; gap:0.5rem;
    background:#0a0f1e; border:1px solid #1e2a42; border-radius:7px;
    padding:0.35rem 0.7rem; margin-bottom:0.6rem; width:fit-content;
}
.id-value {
    font-family:'JetBrains Mono',monospace; font-size:0.72rem;
    color:#64748b; user-select:all; cursor:text;
}
.copy-btn {
    background:none; border:none; cursor:pointer;
    font-size:0.8rem; padding:0; line-height:1;
    opacity:0.6; transition:opacity .15s;
}
.copy-btn:hover { opacity:1; }

/* ── Карточка описания кандидата ── */
.cand-info-card {
    background:#0d1424; border:1px solid #1e2a42; border-radius:10px;
    padding:0.85rem 1.1rem; margin-bottom:0.7rem;
}
.cand-info-row {
    display:flex; gap:0.5rem; align-items:baseline;
    font-size:0.78rem; margin-bottom:0.35rem; flex-wrap:wrap;
}
.cand-info-label {
    font-family:'JetBrains Mono',monospace; font-size:0.65rem;
    color:#3b82f6; font-weight:700; min-width:4.5rem; flex-shrink:0;
}
.cand-info-value { color:#cbd5e1; line-height:1.5; }

.verdict-bar {
    padding:0.35rem 0.8rem; border-radius:5px; font-size:0.78rem;
    margin:0.4rem 0; border-left:3px solid;
}

.login-box {
    max-width:420px; margin:5rem auto;
    background:#0f1420; border:1px solid #1e2a42;
    border-radius:18px; padding:2.8rem 2.2rem; text-align:center;
}

.prog-wrap { background:#1e2a42; border-radius:20px; height:8px; margin:0.3rem 0 0.8rem; overflow:hidden; }
.prog-fill  { height:8px; border-radius:20px; }

.persist-badge {
    display:inline-block; background:rgba(6,214,160,.1);
    border:1px solid rgba(6,214,160,.3); border-radius:6px;
    padding:0.2rem 0.6rem; font-size:0.65rem; color:#06d6a0;
    font-family:'JetBrains Mono',monospace; margin-left:0.5rem; vertical-align:middle;
}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stNumberInput > div > div > input {
    background:#161c2e !important; border:1px solid #1e2a42 !important;
    color:#e2e8f0 !important; border-radius:8px !important;
}
.stButton > button,
.stDownloadButton > button {
    background:#3b82f6 !important; color:white !important; border:none !important;
    border-radius:10px !important; font-family:'JetBrains Mono',monospace !important;
    font-size:0.85rem !important; font-weight:700 !important;
    width:100% !important; padding:0.7rem !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover { background:#2563eb !important; }
label { color:#64748b !important; font-size:0.78rem !important; }
</style>

<!-- JS: копирование ID в буфер обмена -->
<script>
function copyId(val) {
    navigator.clipboard.writeText(val).then(function() {
        var btns = document.querySelectorAll('.copy-btn[data-id="' + val + '"]');
        btns.forEach(function(b){ b.textContent = '✅'; setTimeout(function(){ b.textContent = '📋'; }, 1200); });
    });
}
</script>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# КОНСТАНТЫ
# ═════════════════════════════════════════════════════════════════════════════
HR_PASSWORD       = "hr2024"
CSV_DIR           = Path("Последние файлы")
VACANCY_FILE      = Path("data") / "hh_java_senior_vacancy.json"
HR_DECISIONS_FILE = Path("data") / "hr_decisions.json"
STATUS_OPTIONS    = ["Новый", "На рассмотрении", "Принят", "Отклонён"]
STATUS_ICONS      = {"Принят":"🟢","Отклонён":"🔴","На рассмотрении":"🟡","Новый":"🔵"}
STATUS_COLORS     = {
    "Принят":"#06d6a0","Отклонён":"#f87171",
    "На рассмотрении":"#f59e0b","Новый":"#64748b",
}

# ═════════════════════════════════════════════════════════════════════════════
# PERSISTENT STORAGE
# ═════════════════════════════════════════════════════════════════════════════

def _load_decisions() -> dict:
    try:
        if HR_DECISIONS_FILE.exists():
            data = json.loads(HR_DECISIONS_FILE.read_text(encoding="utf-8"))
            return {"statuses": data.get("statuses", {}), "comments": data.get("comments", {})}
    except Exception:
        pass
    return {"statuses": {}, "comments": {}}


def _save_decisions(statuses: dict, comments: dict) -> None:
    try:
        HR_DECISIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        HR_DECISIONS_FILE.write_text(
            json.dumps({"statuses": statuses, "comments": comments}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        st.toast(f"⚠️ Ошибка сохранения: {e}", icon="⚠️")

# ═════════════════════════════════════════════════════════════════════════════
# РАНЖИРОВЩИК
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_ranker():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core import ResumeRanker
    return ResumeRanker(seed=42)

# ═════════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═════════════════════════════════════════════════════════════════════════════

def _dedup(resumes: list) -> tuple:
    seen: set = set(); unique: list = []; dups = 0
    for r in resumes:
        rid = r.get("id", "")
        if rid and rid in seen: dups += 1; continue
        if rid: seen.add(rid)
        unique.append(r)
    return unique, dups


def _export_csv(ranked: list, statuses: dict, comments: dict) -> bytes:
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["Место","ID резюме","Балл","Уверенность","Вердикт","Статус HR","Комментарий HR"])
    for i, item in enumerate(ranked, 1):
        rid = item["resume_id"]; sc = item["score"]
        verdict = "Подходит" if sc >= 0.6 else "Частично" if sc >= 0.35 else "Не подходит"
        w.writerow([i, rid, round(sc,4), round(item.get("confidence",0.0),2),
                    verdict, statuses.get(rid,"Новый"), comments.get(rid,"")])
    return buf.getvalue().encode("utf-8-sig")


def _prog_bar(pct: float, color: str) -> str:
    return (f'<div class="prog-wrap"><div class="prog-fill" '
            f'style="width:{pct:.1f}%;background:{color};"></div></div>')


def _score_color(sc: float) -> str:
    return "#06d6a0" if sc >= 0.6 else "#f59e0b" if sc >= 0.35 else "#f87171"


def _score_label(sc: float) -> str:
    return "✅ ПОДХОДИТ" if sc >= 0.6 else "⚡ ЧАСТИЧНО" if sc >= 0.35 else "❌ НЕ ПОДХОДИТ"


def _id_block_html(rid: str) -> str:
    """
    Блок ID с кнопкой копирования.
    Клик на 📋 копирует ID в буфер обмена через JS.
    """
    safe = rid.replace("'", "\\'").replace('"', '&quot;')
    return (
        f'<div class="id-block">'
        f'<span style="font-size:0.6rem;color:#334155;">🪪 ID:</span>'
        f'<span class="id-value" id="id-{safe}">{rid}</span>'
        f'<button class="copy-btn" data-id="{safe}" '
        f'onclick="copyId(\'{safe}\')" title="Копировать ID">📋</button>'
        f'</div>'
    )


def _build_expander_title(item: dict, resume_data: dict,
                           rank_emoji: str, status: str, s_icon: str) -> str:
    sc       = item["score"]
    vlbl     = _score_label(sc)
    position = str(resume_data.get("position", "") or "").strip()
    if not position:
        position = str(item.get("resume_id", ""))
    exp_raw  = resume_data.get("experience", "") or resume_data.get("experience_years", "")
    exp_str  = f" | Опыт: {str(exp_raw).strip()}" if exp_raw else ""
    return f"{rank_emoji}  {position}{exp_str}  —  {sc:.4f}  {vlbl}  {s_icon} {status}"


def _render_candidate_info(resume_data: dict, rid: str) -> None:
    """Рендерит описание кандидата из CSV + блок ID с копированием."""
    position = str(resume_data.get("position", "") or "").strip()
    exp_raw  = resume_data.get("experience", "") or resume_data.get("experience_years", "")
    skills   = resume_data.get("skills", []) or []
    about    = (resume_data.get("summary", "") or resume_data.get("about_me", "") or
                resume_data.get("about", "") or "").strip()
    salary   = resume_data.get("salary", "") or resume_data.get("desired_salary", "")
    location = resume_data.get("location", "") or resume_data.get("city", "")
    age      = resume_data.get("age", "")

    skills_str = (", ".join(str(s) for s in skills if s)
                  if isinstance(skills, list) else str(skills))

    rows_html = ""
    if position:
        rows_html += f'<div class="cand-info-row"><span class="cand-info-label">💼 Должность</span><span class="cand-info-value">{position}</span></div>'
    if exp_raw:
        rows_html += f'<div class="cand-info-row"><span class="cand-info-label">🕐 Опыт</span><span class="cand-info-value">{str(exp_raw).strip()}</span></div>'
    if age:
        rows_html += f'<div class="cand-info-row"><span class="cand-info-label">🎂 Возраст</span><span class="cand-info-value">{age}</span></div>'
    if location:
        rows_html += f'<div class="cand-info-row"><span class="cand-info-label">📍 Город</span><span class="cand-info-value">{location}</span></div>'
    if salary:
        rows_html += f'<div class="cand-info-row"><span class="cand-info-label">💰 Зарплата</span><span class="cand-info-value">{salary}</span></div>'
    if skills_str:
        rows_html += f'<div class="cand-info-row"><span class="cand-info-label">🛠 Навыки</span><span class="cand-info-value">{skills_str[:300]}</span></div>'
    if about:
        short = about[:300] + ("…" if len(about) > 300 else "")
        rows_html += f'<div class="cand-info-row"><span class="cand-info-label">📝 О себе</span><span class="cand-info-value" style="font-style:italic;color:#94a3b8;">{short}</span></div>'

    # ID с кнопкой копирования
    id_html = _id_block_html(rid)

    if rows_html:
        st.markdown(
            f'<div class="cand-info-card">'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.6rem;'
            f'color:#3b82f6;letter-spacing:.15em;text-transform:uppercase;margin-bottom:0.55rem;">'
            f'👤 ИНФОРМАЦИЯ О КАНДИДАТЕ</div>'
            f'{rows_html}'
            f'{id_html}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="cand-info-card">{id_html}</div>',
            unsafe_allow_html=True,
        )

# ═════════════════════════════════════════════════════════════════════════════
# КЭШИРОВАННОЕ РАНЖИРОВАНИЕ
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _run_ranking(_ranker, vacancy_json_str: str, csv_dir_str: str) -> dict | None:
    csv_dir   = Path(csv_dir_str)
    csv_files = sorted(csv_dir.glob("*.csv")) if csv_dir.exists() else []
    if not csv_files:
        return None

    vacancy      = json.loads(vacancy_json_str)
    all_raw: list = []
    file_report: list = []

    for f in csv_files:
        rows = _ranker.parse_csv_resumes(f.read_text(encoding="utf-8", errors="replace"))
        all_raw.extend(rows)
        file_report.append({"name": f.name, "count": len(rows)})

    unique, dups = _dedup(all_raw)
    resume_map   = {str(r.get("id", "")): r for r in unique if r.get("id")}

    t0 = time.time()
    result = _ranker.process_batch(vacancy, unique)
    result["_elapsed"]      = round(time.time() - t0, 2)
    result["_file_report"]  = file_report
    result["_raw_count"]    = len(all_raw)
    result["_dup_count"]    = dups
    result["_unique_count"] = len(unique)
    result["_resume_map"]   = resume_map
    return result

# ═════════════════════════════════════════════════════════════════════════════
# ЭКРАН ВХОДА
# ═════════════════════════════════════════════════════════════════════════════
if not st.session_state.get("hr_logged_in", False):
    st.markdown("""
    <div class="login-box">
      <div style="font-size:3rem;margin-bottom:1rem;">👔</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;
                  font-weight:700;color:#e2e8f0;margin-bottom:0.4rem;">Панель HR</div>
      <div style="font-size:0.78rem;color:#64748b;margin-bottom:0.3rem;">
          ML Resume Ranker · Только для специалистов по подбору персонала
      </div>
    </div>
    """, unsafe_allow_html=True)
    _, cc, _ = st.columns([1, 1, 1])
    with cc:
        pwd = st.text_input("🔑 Пароль", type="password", placeholder="Введите пароль…", key="hr_pwd")
        st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
        if st.button("🔐  Войти в панель HR", key="hr_login"):
            if pwd == HR_PASSWORD:
                st.session_state["hr_logged_in"] = True
                saved = _load_decisions()
                st.session_state["hr_statuses"] = saved["statuses"]
                st.session_state["hr_comments"] = saved["comments"]
                st.rerun()
            else:
                st.error("❌ Неверный пароль.")
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# ОСНОВНАЯ ПАНЕЛЬ
# ═════════════════════════════════════════════════════════════════════════════
if "hr_statuses" not in st.session_state:
    saved = _load_decisions()
    st.session_state["hr_statuses"] = saved["statuses"]
    st.session_state["hr_comments"] = saved["comments"]

hr_statuses: dict = st.session_state["hr_statuses"]
hr_comments: dict = st.session_state["hr_comments"]
decisions_count   = len([v for v in hr_statuses.values() if v != "Новый"])

with st.status("⚙️ Загрузка модели…", expanded=False) as ms:
    ranker = load_ranker()
    ms.update(label=f"✅ Модель готова: `{ranker.model_name}`", state="complete")

# ── Шапка ────────────────────────────────────────────────────────────────────
col_h, col_act = st.columns([5, 1])
with col_h:
    files_str = "  ·  ".join(
        f.name for f in sorted(CSV_DIR.glob("*.csv"))
    ) if CSV_DIR.exists() else "нет файлов"
    persist_badge = (
        f'<span class="persist-badge">💾 Сохранено: {decisions_count} решений</span>'
        if decisions_count > 0 else ""
    )
    st.markdown(f"""
    <div class="hr-hero">
      <div style="font-size:2.8rem;line-height:1;">👔</div>
      <div>
        <div class="hr-hero-title">Панель <span>HR</span> {persist_badge}</div>
        <div class="hr-hero-sub">
            Дашборд · Статусы · Комментарии · Экспорт<br>
            <code style="font-size:0.7rem;">{files_str}</code>
        </div>
        <div class="badges">
          <span class="badge b-blue">intfloat/multilingual-e5-base</span>
          <span class="badge b-green">💾 Данные сохраняются</span>
          <span class="badge b-amber">Автозагрузка CSV</span>
          <span class="badge b-red">Только чтение</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
with col_act:
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    if st.button("🔄  Обновить", key="hr_refresh"):
        _run_ranking.clear(); st.rerun()
    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
    if st.button("🚪  Выйти", key="hr_logout"):
        st.session_state["hr_logged_in"] = False; st.rerun()

# ── Проверка файлов ───────────────────────────────────────────────────────────
if not (CSV_DIR.exists() and list(CSV_DIR.glob("*.csv")) and VACANCY_FILE.exists()):
    missing = []
    if not CSV_DIR.exists() or not list(CSV_DIR.glob("*.csv")):
        missing.append(f"CSV в папке `{CSV_DIR}/`")
    if not VACANCY_FILE.exists():
        missing.append(f"Файл вакансии `{VACANCY_FILE}`")
    st.error("❌ Не найдены: " + " · ".join(missing)); st.stop()

vacancy_json_str = VACANCY_FILE.read_text(encoding="utf-8")
with st.status("📊 Загрузка данных…", expanded=False) as rank_st:
    result = _run_ranking(ranker, vacancy_json_str, str(CSV_DIR))
    rank_st.update(label="✅ Данные готовы", state="complete")

if result is None:
    st.error("❌ CSV файлы не найдены."); st.stop()

ranked     = result["ranked_list"]
meta       = result["meta"]
file_rep   = result.get("_file_report", [])
raw_cnt    = result.get("_raw_count", 0)
dup_cnt    = result.get("_dup_count", 0)
uniq_cnt   = result.get("_unique_count", len(ranked))
elapsed    = result.get("_elapsed", 0)
resume_map = result.get("_resume_map", {})

st.caption(
    f"📂 {len(file_rep)} файл(ов) · Загружено: {raw_cnt} · "
    f"Уникальных: {uniq_cnt} · Дублей: {dup_cnt} · Время: {elapsed}с"
)
st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# ДАШБОРД
# ═════════════════════════════════════════════════════════════════════════════
total      = len(ranked)
n_accepted = sum(1 for r in ranked if hr_statuses.get(r["resume_id"]) == "Принят")
n_rejected = sum(1 for r in ranked if hr_statuses.get(r["resume_id"]) == "Отклонён")
n_review   = sum(1 for r in ranked if hr_statuses.get(r["resume_id"]) == "На рассмотрении")
n_new      = total - n_accepted - n_rejected - n_review
n_fit      = sum(1 for r in ranked if r["score"] >= 0.6)
n_partial  = sum(1 for r in ranked if 0.35 <= r["score"] < 0.6)
n_nofit    = sum(1 for r in ranked if r["score"] < 0.35)
avg_score  = sum(r["score"] for r in ranked) / total if ranked else 0
top_score  = ranked[0]["score"] if ranked else 0

st.markdown("### 📊 Дашборд")

sc1, sc2, sc3, sc4, sc5, sc6, sc7 = st.columns(7)
for col, val, lbl in [
    (sc1, total,      "Всего"),
    (sc2, n_fit,      "✅ Подходят"),
    (sc3, n_partial,  "⚡ Частично"),
    (sc4, n_nofit,    "❌ Не подходят"),
    (sc5, n_new,      "🔵 Новые"),
    (sc6, n_accepted, "🟢 Приняты"),
    (sc7, n_rejected, "🔴 Отклонены"),
]:
    col.markdown(
        f'<div class="stat-box"><div class="stat-val">{val}</div>'
        f'<div class="stat-lbl">{lbl}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

dash1, dash2, dash3 = st.columns(3)

with dash1:
    reviewed_pct = round((n_accepted + n_rejected + n_review) / total * 100, 1) if total else 0
    acc_pct = round(n_accepted / total * 100, 1) if total else 0
    rej_pct = round(n_rejected / total * 100, 1) if total else 0
    rev_pct = round(n_review   / total * 100, 1) if total else 0
    new_pct = round(n_new      / total * 100, 1) if total else 0
    st.markdown(f"""
    <div class="dash-card">
      <div class="dash-card-title">📋 Прогресс отбора HR</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:1.8rem;font-weight:700;color:#3b82f6;">{reviewed_pct}%</div>
      <div style="font-size:0.7rem;color:#64748b;margin-bottom:0.8rem;">кандидатов обработано</div>
      <div style="font-size:0.72rem;color:#06d6a0;margin-bottom:0.1rem;">🟢 Принято: {n_accepted} ({acc_pct}%)</div>
      {_prog_bar(acc_pct, "#06d6a0")}
      <div style="font-size:0.72rem;color:#f87171;margin-bottom:0.1rem;">🔴 Отклонено: {n_rejected} ({rej_pct}%)</div>
      {_prog_bar(rej_pct, "#f87171")}
      <div style="font-size:0.72rem;color:#f59e0b;margin-bottom:0.1rem;">🟡 На рассмотрении: {n_review} ({rev_pct}%)</div>
      {_prog_bar(rev_pct, "#f59e0b")}
      <div style="font-size:0.72rem;color:#64748b;margin-bottom:0.1rem;">🔵 Новые: {n_new} ({new_pct}%)</div>
      {_prog_bar(new_pct, "#64748b")}
    </div>
    """, unsafe_allow_html=True)

with dash2:
    fit_pct = round(n_fit     / total * 100, 1) if total else 0
    par_pct = round(n_partial / total * 100, 1) if total else 0
    nof_pct = round(n_nofit   / total * 100, 1) if total else 0
    st.markdown(f"""
    <div class="dash-card">
      <div class="dash-card-title">🤖 Вердикт модели</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:1.8rem;font-weight:700;color:#06d6a0;">{top_score:.4f}</div>
      <div style="font-size:0.7rem;color:#64748b;margin-bottom:0.8rem;">лучший балл · средний: {avg_score:.4f}</div>
      <div style="font-size:0.72rem;color:#06d6a0;margin-bottom:0.1rem;">✅ Подходят ≥0.6: {n_fit} ({fit_pct}%)</div>
      {_prog_bar(fit_pct, "#06d6a0")}
      <div style="font-size:0.72rem;color:#f59e0b;margin-bottom:0.1rem;">⚡ Частично 0.35–0.6: {n_partial} ({par_pct}%)</div>
      {_prog_bar(par_pct, "#f59e0b")}
      <div style="font-size:0.72rem;color:#f87171;margin-bottom:0.1rem;">❌ Не подходят &lt;0.35: {n_nofit} ({nof_pct}%)</div>
      {_prog_bar(nof_pct, "#f87171")}
    </div>
    """, unsafe_allow_html=True)

with dash3:
    top5_html = ""
    for i, r in enumerate(ranked[:5]):
        sc     = r["score"]
        c      = _score_color(sc)
        emojis = ["🥇","🥈","🥉","4️⃣","5️⃣"][i]
        rid_s  = str(r["resume_id"])
        r_data = resume_map.get(rid_s, {})
        dname  = str(r_data.get("position", "") or rid_s)
        dname  = dname[:28] + "…" if len(dname) > 28 else dname
        status = hr_statuses.get(r["resume_id"], "Новый")
        s_icon = STATUS_ICONS.get(status, "🔵")
        top5_html += (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:0.35rem 0;border-bottom:1px solid #1e2a42;">'
            f'<span style="font-size:0.8rem;">{emojis} '
            f'<span style="font-size:0.75rem;color:#cbd5e1;">{dname}</span></span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.78rem;'
            f'color:{c};font-weight:700;">{sc:.4f} {s_icon}</span>'
            f'</div>'
        )
    st.markdown(f"""
    <div class="dash-card">
      <div class="dash-card-title">🏆 Топ-5 кандидатов</div>
      {top5_html}
      <div style="font-size:0.65rem;color:#64748b;margin-top:0.6rem;">Полный список — в разделе ниже</div>
    </div>
    """, unsafe_allow_html=True)

if ranked and top_score >= 0.35:
    best      = ranked[0]
    best_sum  = best.get("explanation", {}).get("summary", "")
    best_rid  = str(best["resume_id"])
    best_data = resume_map.get(best_rid, {})
    best_pos  = str(best_data.get("position", "") or best_rid)
    best_exp  = str(best_data.get("experience", "") or best_data.get("experience_years", "") or "")
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(6,214,160,.08),rgba(59,130,246,.06));
        border:1px solid rgba(6,214,160,.35);border-radius:12px;padding:1rem 1.5rem;margin:0.8rem 0;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                  color:#06d6a0;letter-spacing:.2em;margin-bottom:0.4rem;">🏆 ЛУЧШИЙ КАНДИДАТ</div>
      <div style="font-size:0.95rem;font-weight:700;color:#e2e8f0;">{best_pos}</div>
      {f'<div style="font-size:0.75rem;color:#64748b;">🕐 Опыт: {best_exp}</div>' if best_exp else ''}
      <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                  color:#06d6a0;font-weight:700;">Балл: {best['score']:.4f}</div>
      <div style="font-size:0.75rem;color:#94a3b8;margin-top:0.2rem;font-style:italic;">{best_sum}</div>
      {_id_block_html(best_rid)}
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# ФИЛЬТРЫ
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("#### 🔍 Фильтры")
f1, f2, f3 = st.columns(3)
with f1:
    filter_status = st.selectbox("Статус", ["Все"] + STATUS_OPTIONS, key="hr_f_status")
with f2:
    filter_min_score = st.slider("Минимальный балл", 0.0, 1.0, 0.0, 0.05, key="hr_f_score")
with f3:
    filter_search = st.text_input("Поиск по должности / ID", placeholder="Введите текст…", key="hr_f_search")

filtered = []
for item in ranked:
    rid    = str(item["resume_id"])
    r_data = resume_map.get(rid, {})
    pos    = str(r_data.get("position", "") or "").lower()
    if (
        (filter_status == "Все" or hr_statuses.get(item["resume_id"], "Новый") == filter_status)
        and item["score"] >= filter_min_score
        and (not filter_search
             or filter_search.lower() in rid.lower()
             or filter_search.lower() in pos)
    ):
        filtered.append(item)

display_items = filtered[:30]
st.caption(
    f"Показано **{len(display_items)}** из **{len(filtered)}** (всего: {total})  ·  "
    f"Вакансия: **{meta.get('vacancy_title','—')}**"
)
st.markdown("---")
st.markdown("#### 📋 Топ-30 кандидатов")

# ═════════════════════════════════════════════════════════════════════════════
# КАРТОЧКИ КАНДИДАТОВ
# ═════════════════════════════════════════════════════════════════════════════
for i, item in enumerate(display_items):
    rid        = item["resume_id"]
    rid_str    = str(rid)
    sc         = item["score"]
    status     = hr_statuses.get(rid, "Новый")
    conf       = item.get("confidence", 0.0)
    vc         = _score_color(sc)
    s_icon     = STATUS_ICONS.get(status, "🔵")
    rank_emoji = ["🥇","🥈","🥉"][i] if i < 3 else f"#{i+1}"
    r_data     = resume_map.get(rid_str, {})

    expander_title = _build_expander_title(item, r_data, rank_emoji, status, s_icon)

    with st.expander(expander_title, expanded=False):

        # 1. Описание кандидата из CSV + ID с кнопкой копирования
        _render_candidate_info(r_data, rid_str)

        # 2. Метрики компонентов
        comp = item.get("explanation", {}).get("components", {})
        if comp:
            mc = st.columns(4)
            for j, (k, v) in enumerate(list(comp.items())[:8]):
                val_num = v.get("value", 0) if isinstance(v, dict) else float(v)
                weight  = v.get("weight", 0) if isinstance(v, dict) else 0
                mc[j % 4].metric(k.replace("_", " ").title(), f"{val_num:.2f}", f"вес={weight}")

        # 3. Вердикт
        summary = item.get("explanation", {}).get("summary", "")
        if summary:
            st.markdown(
                f'<div class="verdict-bar" style="background:#161c2e;'
                f'border-left-color:{vc};color:{vc};">'
                f'🤖 Вердикт: <b>{summary}</b>'
                f'<span style="color:#64748b;font-size:0.72rem;"> · Уверенность: {conf:.2f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # 4. Детальное объяснение
        if comp:
            with st.expander("📐 Детальное объяснение компонентов", expanded=False):
                col_a, col_b = st.columns(2)
                for j, (k, v) in enumerate(list(comp.items())):
                    if isinstance(v, dict):
                        val_num     = v.get("value", 0)
                        weight      = v.get("weight", 0)
                        explanation = str(v.get("explanation", "") or "").strip()
                        matched     = v.get("matched", [])
                        raw_score   = v.get("raw_score", None)
                    else:
                        val_num = float(v); weight = 0; explanation = ""; matched = []; raw_score = None

                    matched_list = matched if isinstance(matched, list) else []
                    matched_text = ""
                    if matched_list:
                        shown = ", ".join(str(m) for m in matched_list[:6])
                        tail  = f" +{len(matched_list)-6}" if len(matched_list) > 6 else ""
                        matched_text = f"✅ Совпавшие: {shown}{tail}"

                    raw_text     = (f'<div style="font-size:0.65rem;color:#64748b;margin-top:0.15rem;">Сырой балл: {raw_score:.3f}</div>') if raw_score is not None else ""
                    expl_html    = f'<div class="comp-card-body">{explanation}</div>' if explanation else ""
                    matched_html = f'<div class="comp-card-matched">{matched_text}</div>' if matched_text else ""

                    cc = col_a if j % 2 == 0 else col_b
                    cc.markdown(f"""
                    <div class="comp-card">
                      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.3rem;">
                        <div class="comp-card-title">{k.replace('_',' ').title()}</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;
                                    font-weight:700;color:{_score_color(val_num)};">{val_num:.3f}</div>
                      </div>
                      <div style="font-size:0.65rem;color:#3b82f6;margin-bottom:0.3rem;">Вес: {weight}</div>
                      {expl_html}{matched_html}{raw_text}
                    </div>
                    """, unsafe_allow_html=True)

        if item.get("warnings"):
            for w in item["warnings"]: st.warning(w)

        st.markdown("---")

        # 5. Управление HR
        hc1, hc2 = st.columns([1, 2])
        with hc1:
            new_status = st.selectbox(
                "📌 Статус кандидата", STATUS_OPTIONS,
                index=STATUS_OPTIONS.index(status),
                key=f"hr_st_{rid}_{i}",
            )
            if new_status != status:
                st.session_state["hr_statuses"][rid] = new_status
                hr_statuses[rid] = new_status
                _save_decisions(hr_statuses, hr_comments)
                st.markdown(
                    f'<div style="font-size:0.7rem;margin-top:0.3rem;'
                    f'color:{STATUS_COLORS.get(new_status,"#64748b")};">'
                    f'💾 Сохранено: {STATUS_ICONS.get(new_status,"")} {new_status}</div>',
                    unsafe_allow_html=True,
                )
        with hc2:
            comment_val = hr_comments.get(rid, "")
            new_comment = st.text_area(
                "💬 Комментарий HR", value=comment_val, height=90,
                placeholder="Напишите заметку: причина отказа, интересный навык, вопрос на собеседование…",
                key=f"hr_cm_{rid}_{i}",
            )
            if new_comment != comment_val:
                st.session_state["hr_comments"][rid] = new_comment
                hr_comments[rid] = new_comment
                _save_decisions(hr_statuses, hr_comments)

# ═════════════════════════════════════════════════════════════════════════════
# ЭКСПОРТ
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("#### 📥 Экспорт результатов")

e1, e2, e3 = st.columns(3)
with e1:
    st.download_button(
        "⬇️  Топ-30 кандидатов (CSV)",
        data=_export_csv(ranked[:30], hr_statuses, hr_comments),
        file_name=f"hr_топ30_{meta.get('vacancy_title','').replace(' ','_')}.csv",
        mime="text/csv", key="hr_exp_top30",
    )
with e2:
    accepted = [r for r in ranked if hr_statuses.get(r["resume_id"]) == "Принят"]
    if accepted:
        st.download_button(
            f"⬇️  Только принятые ({len(accepted)})",
            data=_export_csv(accepted, hr_statuses, hr_comments),
            file_name=f"hr_принятые_{meta.get('vacancy_title','').replace(' ','_')}.csv",
            mime="text/csv", key="hr_exp_acc",
        )
    else:
        st.download_button(
            "⬇️  Принятые (нет)",
            data=b"", file_name="empty.csv", mime="text/csv",
            key="hr_exp_acc_empty", disabled=True,
        )
with e3:
    st.download_button(
        "⬇️  Все кандидаты (CSV)",
        data=_export_csv(ranked, hr_statuses, hr_comments),
        file_name=f"hr_все_{meta.get('vacancy_title','').replace(' ','_')}.csv",
        mime="text/csv", key="hr_exp_all",
    )

st.markdown("")
_, cr, _ = st.columns([2, 1, 2])
with cr:
    if st.button("🗑️  Сбросить все решения HR", key="hr_reset"):
        st.session_state["hr_statuses"] = {}
        st.session_state["hr_comments"] = {}
        hr_statuses.clear(); hr_comments.clear()
        try:
            if HR_DECISIONS_FILE.exists():
                HR_DECISIONS_FILE.unlink()
        except Exception:
            pass
        st.rerun()