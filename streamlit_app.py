from pathlib import Path
import hashlib
import sqlite3
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title=" AI House Price Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "ames_users.db"

# ============================================================
# MODEL CATALOG
# ============================================================
MODELS = {
    "Random Forest": {"file": "models/09_random_forest.joblib", "tagline": "300 averaged trees — a strong default for tabular data.", "mae": 14217, "rmse": 22342, "r2": 0.930, "note": "Bagged ensemble"},
    "Ridge (L2)": {"file": "models/04_ridge.joblib", "tagline": "Regularised linear regression that controls large coefficients.", "mae": 13829, "rmse": 22230, "r2": 0.930, "note": "Polynomial + shrinkage"},
    "Lasso (L1)": {"file": "models/05_lasso.joblib", "tagline": "Regularisation that can remove unnecessary features.", "mae": 13905, "rmse": 21661, "r2": 0.934, "note": "83/245 features kept"},
    "Elastic Net": {"file": "models/06_elastic_net.joblib", "tagline": "A blend of Ridge-style and Lasso-style regularisation.", "mae": 13905, "rmse": 21661, "r2": 0.934, "note": "l1_ratio = 1.0"},
    "Polynomial": {"file": "models/03_polynomial.joblib", "tagline": "Adds squared and interaction terms to capture curvature.", "mae": 14422, "rmse": 24002, "r2": 0.919, "note": "Squares + interactions"},
    "Multiple Linear": {"file": "models/02_multiple_linear.joblib", "tagline": "One coefficient per feature — highly explainable.", "mae": 15392, "rmse": 23701, "r2": 0.921, "note": "27 features"},
    "Support Vector Regression": {"file": "models/07_svr.joblib", "tagline": "Flexible kernel model with a tolerance tube.", "mae": 15396, "rmse": 22674, "r2": 0.928, "note": "RBF kernel, C=10"},
    "Decision Tree": {"file": "models/08_decision_tree.joblib", "tagline": "Human-readable yes/no decision rules.", "mae": 19285, "rmse": 28036, "r2": 0.889, "note": "Max depth 10"},
    "Simple Linear": {"file": "models/01_simple_linear.joblib", "tagline": "A simple living-area-only baseline.", "mae": 39604, "rmse": 58520, "r2": 0.518, "note": "Living area only"},
}

DATASET_MEDIAN = 160000
DATASET_MEAN = 180796
DATASET_MIN = 12789
DATASET_MAX = 755000

NEIGHBORHOODS = [
    "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr", "Crawfor",
    "Edwards", "Gilbert", "Greens", "GrnHill", "IDOTRR", "Landmrk", "MeadowV",
    "Mitchel", "NAmes", "NPkVill", "NWAmes", "NoRidge", "NridgHt", "OldTown",
    "SWISU", "Sawyer", "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker",
]

# ============================================================
# SESSION STATE
# ============================================================
def init_state():
    defaults = {
        "theme": "light",
        "entry": None,
        "username": "",
        "full_name": "",
        "page": "🏡 Predict",
        "house_values": None,
        "last_model": "Random Forest",
        "history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()

# ============================================================
# DATABASE / AUTH
# ============================================================
def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, full_name TEXT NOT NULL, password_hash TEXT NOT NULL, created_at TEXT NOT NULL)"
    )
    conn.commit()
    return conn


def password_hash(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def create_user(full_name: str, username: str, password: str):
    username = username.strip().lower()
    conn = db_connect()
    try:
        conn.execute(
            "INSERT INTO users(username, full_name, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (username, full_name.strip(), password_hash(password), datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "This username already exists."
    finally:
        conn.close()


def authenticate(username: str, password: str):
    conn = db_connect()
    row = conn.execute(
        "SELECT username, full_name FROM users WHERE username=? AND password_hash=?",
        (username.strip().lower(), password_hash(password)),
    ).fetchone()
    conn.close()
    return row


# ============================================================
# DATA / MODELS
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(str(BASE_DIR / path))


@st.cache_data(show_spinner=False)
def load_dataset():
    path = BASE_DIR / "data" / "ames.csv"
    return pd.read_csv(path) if path.exists() else None


def build_house_row(values: dict) -> pd.DataFrame:
    return pd.DataFrame([values])


def predict_with(model_label: str, house: pd.DataFrame) -> float:
    model = load_model(MODELS[model_label]["file"])
    cols = list(model.feature_names_in_)
    return float(model.predict(house[cols])[0])


def get_feature_signal(model_label: str):
    model = load_model(MODELS[model_label]["file"])
    if hasattr(model, "named_steps"):
        prep = model.named_steps.get("prep")
        est = model.named_steps.get("model")
        if prep is None or est is None:
            return None, None
        try:
            names = [n.replace("num__", "").replace("cat__", "") for n in prep.get_feature_names_out()]
        except Exception:
            return None, None
    else:
        est = model
        names = list(getattr(model, "feature_names_in_", []))
        if not names:
            return None, None

    if hasattr(est, "feature_importances_"):
        vals, kind = est.feature_importances_, "importance"
    elif hasattr(est, "coef_"):
        vals, kind = np.ravel(est.coef_), "coefficient"
    else:
        return None, None

    return pd.DataFrame({"feature": names, "value": vals}), kind


def recommend_model():
    return min(MODELS, key=lambda name: MODELS[name]["mae"])


def add_history(model_label, prediction, values):
    if st.session_state.username == "Guest":
        return
    st.session_state.history.insert(0, {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Model": model_label,
        "Prediction": round(prediction, 2),
        "Neighborhood": values.get("Neighborhood", ""),
        "Living Area": values.get("Gr Liv Area", 0),
    })
    st.session_state.history = st.session_state.history[:20]


# ============================================================
# THEME CSS
# ============================================================
dark = st.session_state.theme == "dark"
BG = "#07111F" if dark else "#F5F7FB"
SURFACE = "#0E1B2D" if dark else "#FFFFFF"
SURFACE2 = "#13253B" if dark else "#F8FAFC"
TEXT = "#F1F5FF" if dark else "#0F172A"
MUTED = "#A7B6CB" if dark else "#64748B"
BORDER = "#29405E" if dark else "#E2E8F0"
PRIMARY = "#5B8CFF" if dark else "#315EFB"
SECONDARY = "#8B5CF6"
CYAN = "#22D3EE"
GREEN = "#34D399"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');
:root {{ --bg:{BG}; --surface:{SURFACE}; --surface2:{SURFACE2}; --text:{TEXT}; --muted:{MUTED}; --border:{BORDER}; --primary:{PRIMARY}; --secondary:{SECONDARY}; --cyan:{CYAN}; --green:{GREEN}; }}
html, body, [class*="css"] {{ font-family:'Inter',sans-serif; }}
#MainMenu, footer {{ visibility:hidden; }}
.stApp {{ background:radial-gradient(circle at 8% 0%, rgba(49,94,251,.13), transparent 30%), radial-gradient(circle at 96% 5%, rgba(139,92,246,.12), transparent 27%), var(--bg); color:var(--text); }}
.block-container {{ max-width:1480px; padding-top:1.3rem; padding-bottom:3rem; }}
[data-testid="stSidebar"] {{ background:var(--surface); border-right:1px solid var(--border); }}
[data-testid="stSidebar"] * {{ color:var(--text); }}
h1,h2,h3,h4,h5,h6,p,label,.stMarkdown {{ color:var(--text); }}
.hero {{ position:relative; overflow:hidden; padding:2.2rem 2.5rem; border-radius:25px; color:#fff; margin-bottom:1.25rem; background:linear-gradient(115deg,#315EFB,#7048F5,#06B6D4,#315EFB); background-size:300% 300%; animation:gradient 10s ease infinite, rise .55s ease both; box-shadow:0 20px 55px rgba(49,94,251,.23); }}
.hero h1 {{ color:#fff!important; font-family:'Manrope',sans-serif; font-size:clamp(2rem,3.5vw,3.25rem); margin:0 0 .4rem; letter-spacing:-.04em; }}
.hero p {{ color:#fff!important; opacity:.93; max-width:900px; margin:0; }}
.badges {{ display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1.1rem; }}
.badge {{ padding:.35rem .75rem; border:1px solid rgba(255,255,255,.3); background:rgba(255,255,255,.13); border-radius:999px; font-size:.78rem; font-weight:800; backdrop-filter:blur(8px); }}
.section-label {{ color:var(--primary); font-size:.76rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; margin:.3rem 0 .65rem; }}
.card {{ background:linear-gradient(180deg,var(--surface),var(--surface2)); border:1px solid var(--border); border-radius:18px; padding:1.25rem 1.35rem; box-shadow:0 10px 30px rgba(15,23,42,.07); animation:rise .45s ease both; transition:.2s ease; }}
.card:hover {{ transform:translateY(-3px); box-shadow:0 17px 40px rgba(49,94,251,.13); }}
.price {{ color:var(--primary); font-family:'Manrope',sans-serif; font-weight:800; font-size:clamp(2.4rem,5vw,3.8rem); letter-spacing:-.05em; }}
.muted {{ color:var(--muted)!important; }}
div.stButton > button, div.stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {{ border:0; border-radius:12px; font-weight:800; color:white!important; background:linear-gradient(100deg,var(--primary),var(--secondary)); box-shadow:0 8px 22px rgba(49,94,251,.18); transition:.18s ease; }}
div.stButton > button:hover, div.stDownloadButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {{ transform:translateY(-2px); filter:brightness(1.07); box-shadow:0 13px 30px rgba(49,94,251,.25); }}
[data-baseweb="input"] > div, [data-baseweb="select"] > div, [data-baseweb="base-input"], .stNumberInput input {{ background:var(--surface)!important; color:var(--text)!important; border-color:var(--border)!important; border-radius:11px!important; }}
[data-testid="stMetric"] {{ background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:.75rem; }}
[data-testid="stMetricValue"] {{ color:var(--primary); font-weight:800; }}
[data-testid="stMetricLabel"], .stCaption {{ color:var(--muted)!important; }}
[data-testid="stDataFrame"] {{ border:1px solid var(--border); border-radius:14px; overflow:hidden; }}
hr {{ border-color:var(--border)!important; }}
.login-shell {{ max-width:620px; margin:7vh auto 0; text-align:center; animation:rise .55s ease both; }}
.logo {{ width:82px;height:82px;margin:0 auto 1rem;border-radius:24px;display:grid;place-items:center;font-size:2.5rem;background:linear-gradient(135deg,var(--primary),var(--secondary));box-shadow:0 18px 40px rgba(49,94,251,.25); }}
.auth-card {{ background:linear-gradient(180deg,var(--surface),var(--surface2)); border:1px solid var(--border); border-radius:22px; padding:1.5rem; box-shadow:0 18px 50px rgba(15,23,42,.10); }}
.option-card {{ text-align:center; background:var(--surface); border:1px solid var(--border); border-radius:18px; padding:1rem; margin:.4rem 0 1rem; }}
.small-pill {{ display:inline-block; padding:.25rem .55rem; border-radius:999px; background:rgba(34,211,238,.12); color:var(--cyan); font-size:.72rem; font-weight:800; }}
@keyframes rise {{ from{{opacity:0;transform:translateY(13px)}} to{{opacity:1;transform:translateY(0)}} }}
@keyframes gradient {{ 0%{{background-position:0% 50%}} 50%{{background-position:100% 50%}} 100%{{background-position:0% 50%}} }}
@media(max-width:800px) {{ .hero{{padding:1.5rem;border-radius:18px}} .block-container{{padding-top:.8rem}} }}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# LANDING / AUTH
# ============================================================
def go_entry(value):
    st.session_state.entry = value
    st.rerun()


if st.session_state.entry is None:
    st.markdown(
        '<div class="login-shell"><div class="logo">🏠</div><h1>Ames AI</h1><p class="muted">House Price Intelligence Platform</p><div class="option-card"><h3>How would you like to continue?</h3><p class="muted">Choose the experience that fits you.</p></div></div>',
        unsafe_allow_html=True,
    )
    a, b, c = st.columns([1, 1.35, 1])
    with b:
        if st.button("✨ New User — Create Account", use_container_width=True):
            go_entry("register")
        if st.button("👤 Continue as Guest", use_container_width=True):
            st.session_state.username = "Guest"
            st.session_state.full_name = "Guest User"
            go_entry("guest")
        st.markdown('<p style="text-align:center;color:var(--muted)">────────  OR  ────────</p>', unsafe_allow_html=True)
        if st.button("🔐 Existing User — Sign In", use_container_width=True):
            go_entry("login")
    st.stop()


if st.session_state.entry == "register":
    st.markdown('<div class="login-shell"><div class="logo">✨</div><h1>Create your account</h1><p class="muted">Save your session and keep your recent predictions organised.</p></div>', unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.15, 1])
    with col:
        with st.form("register_form"):
            full_name = st.text_input("Full name", placeholder="Amgad Farg")
            username = st.text_input("Username", placeholder="amgad")
            password = st.text_input("Password", type="password", placeholder="At least 6 characters")
            confirm = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create Account →", use_container_width=True)
        if submitted:
            if len(full_name.strip()) < 2 or len(username.strip()) < 3:
                st.error("Please enter a valid name and username.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters.")
            elif password != confirm:
                st.error("Passwords do not match.")
            else:
                ok, message = create_user(full_name, username, password)
                if ok:
                    st.session_state.username = username.strip().lower()
                    st.session_state.full_name = full_name.strip()
                    st.session_state.entry = "app"
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        if st.button("← Back", use_container_width=True):
            go_entry(None)
    st.stop()


if st.session_state.entry == "login":
    st.markdown('<div class="login-shell"><div class="logo">🔐</div><h1>Welcome back</h1><p class="muted">Sign in to your Ames AI workspace.</p></div>', unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.15, 1])
    with col:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In →", use_container_width=True)
        if submitted:
            row = authenticate(username, password)
            if row:
                st.session_state.username, st.session_state.full_name = row
                st.session_state.entry = "app"
                st.rerun()
            else:
                st.error("Incorrect username or password.")
        st.caption("No account? Use New User on the previous screen.")
        if st.button("← Back", use_container_width=True):
            go_entry(None)
    st.stop()


if st.session_state.entry == "guest":
    st.session_state.entry = "app"

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("# 🏠 Ames AI")
    st.caption("House Price Intelligence")
    st.divider()
    st.markdown(f"**👤 {st.session_state.full_name or 'Guest User'}**")
    st.caption(f"@{st.session_state.username}")
    if st.session_state.username == "Guest":
        st.info("Guest mode: explore all prediction tools. Your account data is not saved.")
    else:
        st.success("Signed in")

    st.divider()
    st.markdown("### 🧭 Navigation")
    pages = ["🏡 Predict", "🔬 Compare", "🏆 Leaderboard", "📈 Analytics", "🔍 Explain", "🕘 History", "ℹ️ About"]
    st.session_state.page = st.radio("Go to", pages, index=pages.index(st.session_state.page), label_visibility="collapsed")

    st.divider()
    st.markdown("### 🎨 Appearance")
    theme_choice = st.radio("Theme", ["☀️ Light", "🌙 Dark"], index=1 if dark else 0, horizontal=True, label_visibility="collapsed")
    new_theme = "dark" if theme_choice.startswith("🌙") else "light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.divider()
    if st.button("🚪 Log out", use_container_width=True):
        for key in ["entry", "username", "full_name", "house_values", "history"]:
            st.session_state.pop(key, None)
        st.rerun()

# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
<div class="hero">
  <h1>🏠 Ames House Price Predictor</h1>
  <p>Predict a home's sale price, compare nine regression models, understand model behaviour, and export your analysis.</p>
  <div class="badges"><span class="badge">🤖 9 ML Models</span><span class="badge">📊 2,930 Sales</span><span class="badge">🎯 Best R² · 0.934</span><span class="badge">⚡ Fast Local Prediction</span></div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# PREDICT PAGE
# ============================================================
if st.session_state.page == "🏡 Predict":
    left, right = st.columns([1.02, 1.35], gap="large")

    with left:
        st.markdown('<div class="section-label">01 · Model</div>', unsafe_allow_html=True)
        model_label = st.selectbox("Choose prediction model", list(MODELS.keys()), index=list(MODELS.keys()).index(st.session_state.last_model))
        st.session_state.last_model = model_label
        info = MODELS[model_label]
        st.markdown(f'<div class="card"><span class="small-pill">{info["note"]}</span><h3>{model_label}</h3><p class="muted">{info["tagline"]}</p></div>', unsafe_allow_html=True)

        rec = recommend_model()
        if model_label != rec:
            st.info(f"💡 Recommended default: **{rec}** — it has the lowest MAE in the held-out evaluation.")
        else:
            st.success("⭐ You selected the current lowest-MAE model.")

        st.markdown('<div class="section-label" style="margin-top:1.2rem">02 · House details</div>', unsafe_allow_html=True)
        with st.form("house_form"):
            t1, t2, t3, t4 = st.tabs(["📐 Size", "⭐ Quality", "🛠️ Amenities", "📍 Location"])
            with t1:
                gr_liv_area = st.number_input("Above-ground living area (sq ft)", 300, 6000, 1500, step=50)
                total_bsmt_sf = st.number_input("Total basement area (sq ft)", 0, 6500, 990, step=50)
                first_flr_sf = st.number_input("1st floor area (sq ft)", 300, 5500, 1080, step=50)
                lot_area = st.number_input("Lot area (sq ft)", 1000, 220000, 9500, step=500)
                tot_rms = st.slider("Total rooms above grade", 2, 15, 6)
            with t2:
                overall_qual = st.slider("Overall quality (1–10)", 1, 10, 6)
                overall_cond = st.slider("Overall condition (1–10)", 1, 10, 5)
                exter_qual = st.selectbox("Exterior quality", ["Ex", "Gd", "TA", "Fa"], index=1)
                kitchen_qual = st.selectbox("Kitchen quality", ["Ex", "Gd", "TA", "Fa", "Po"], index=1)
                bsmt_qual = st.selectbox("Basement quality", ["Ex", "Gd", "TA", "Fa", "Po"], index=1)
                heating_qc = st.selectbox("Heating quality", ["Ex", "Gd", "TA", "Fa", "Po"], index=1)
            with t3:
                garage_cars = st.slider("Garage capacity (cars)", 0, 5, 2)
                garage_area = st.number_input("Garage area (sq ft)", 0, 1500, 480, step=20)
                garage_type = st.selectbox("Garage type", ["Attchd", "Detchd", "BuiltIn", "Basment", "CarPort", "2Types"])
                full_bath = st.slider("Full bathrooms", 0, 4, 2)
                half_bath = st.slider("Half bathrooms", 0, 2, 0)
                fireplaces = st.slider("Fireplaces", 0, 4, 1)
                central_air = st.selectbox("Central air", ["Y", "N"])
                wood_deck_sf = st.number_input("Wood deck area (sq ft)", 0, 1400, 0, step=25)
                open_porch_sf = st.number_input("Open porch area (sq ft)", 0, 750, 30, step=25)
                mas_vnr_area = st.number_input("Masonry veneer area (sq ft)", 0, 1600, 0, step=25)
            with t4:
                neighborhood = st.selectbox("Neighborhood", NEIGHBORHOODS, index=5)
                house_style = st.selectbox("House style", ["1Story", "1.5Fin", "1.5Unf", "2Story", "2.5Fin", "2.5Unf", "SFoyer", "SLvl"])
                bldg_type = st.selectbox("Building type", ["1Fam", "2fmCon", "Duplex", "Twnhs", "TwnhsE"])
                foundation = st.selectbox("Foundation", ["PConc", "CBlock", "BrkTil", "Slab", "Stone", "Wood"])
                year_built = st.number_input("Year built", 1870, 2010, 1973)
                year_remod = st.number_input("Year remodeled/added", 1950, 2010, 1993)
            submitted = st.form_submit_button("🚀 Predict House Price", use_container_width=True)

        if submitted:
            values = {
                "Gr Liv Area": gr_liv_area, "Total Bsmt SF": total_bsmt_sf, "1st Flr SF": first_flr_sf,
                "Lot Area": lot_area, "Overall Qual": overall_qual, "Overall Cond": overall_cond,
                "Year Built": year_built, "Year Remod/Add": year_remod, "Garage Cars": garage_cars,
                "Garage Area": garage_area, "Full Bath": full_bath, "Half Bath": half_bath,
                "TotRms AbvGrd": tot_rms, "Fireplaces": fireplaces, "Mas Vnr Area": mas_vnr_area,
                "Wood Deck SF": wood_deck_sf, "Open Porch SF": open_porch_sf, "Neighborhood": neighborhood,
                "House Style": house_style, "Bldg Type": bldg_type, "Exter Qual": exter_qual,
                "Kitchen Qual": kitchen_qual, "Bsmt Qual": bsmt_qual, "Heating QC": heating_qc,
                "Central Air": central_air, "Foundation": foundation, "Garage Type": garage_type,
            }
            st.session_state.house_values = values
            st.session_state.last_model = model_label
            st.rerun()

    with right:
        st.markdown('<div class="section-label">03 · AI estimate</div>', unsafe_allow_html=True)
        if st.session_state.house_values is None:
            st.markdown('<div class="card"><h2>Ready when you are 👋</h2><p class="muted">Fill in the house details on the left and click <b>Predict House Price</b>. Your estimate, range, model metrics, and explanation will appear here.</p></div>', unsafe_allow_html=True)
            a, b, c = st.columns(3)
            a.metric("Dataset median", f"${DATASET_MEDIAN:,.0f}")
            b.metric("Dataset mean", f"${DATASET_MEAN:,.0f}")
            c.metric("Observed range", f"${DATASET_MIN/1000:.0f}k–${DATASET_MAX/1000:.0f}k")
        else:
            house = build_house_row(st.session_state.house_values)
            info = MODELS[st.session_state.last_model]
            with st.spinner("🤖 Running the model..."):
                prediction = predict_with(st.session_state.last_model, house)
            low = max(0, prediction - info["mae"])
            high = prediction + info["mae"]
            vs_median = (prediction / DATASET_MEDIAN - 1) * 100
            st.markdown(f'<div class="card"><span class="muted">{st.session_state.last_model} estimate</span><div class="price">${prediction:,.0f}</div><p class="muted">Typical-error range: <b>${low:,.0f} – ${high:,.0f}</b></p></div>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("vs median", f"${DATASET_MEDIAN:,.0f}", f"{vs_median:+.1f}%")
            m2.metric("R²", f"{info['r2']:.3f}")
            m3.metric("MAE", f"${info['mae']:,.0f}")
            m4.metric("RMSE", f"${info['rmse']:,.0f}")

            add_history(st.session_state.last_model, prediction, st.session_state.house_values)
            st.markdown("#### 🧠 Quick interpretation")
            quality = st.session_state.house_values["Overall Qual"]
            size = st.session_state.house_values["Gr Liv Area"]
            price_position = "above" if prediction > DATASET_MEDIAN else "below"
            st.markdown(f'<div class="card"><p>This home is estimated <b>{price_position}</b> the dataset median. Its living area is <b>{size:,} sq ft</b> and overall quality is <b>{quality}/10</b>.</p><p class="muted">The range is an uncertainty guide based on the model MAE; it is not a formal confidence interval.</p></div>', unsafe_allow_html=True)

            cols_needed = list(load_model(info["file"]).feature_names_in_)
            with st.expander("🔎 See exact model input"):
                st.dataframe(house[cols_needed].T.rename(columns={0: "value"}), use_container_width=True)

            report = pd.DataFrame([{"Model": st.session_state.last_model, "Predicted Price": prediction, "Lower Range": low, "Upper Range": high, **st.session_state.house_values}])
            st.download_button("⬇️ Download prediction CSV", report.to_csv(index=False), "ames_prediction.csv", "text/csv", use_container_width=True)

# ============================================================
# COMPARE
# ============================================================
elif st.session_state.page == "🔬 Compare":
    st.markdown('<div class="section-label">Model laboratory</div>', unsafe_allow_html=True)
    if st.session_state.house_values is None:
        st.info("Go to **🏡 Predict**, enter a house, and run one prediction first.")
    else:
        house = build_house_row(st.session_state.house_values)
        rows = []
        with st.spinner("Running all nine models..."):
            for label in MODELS:
                pred = predict_with(label, house)
                rows.append({"Model": label, "Prediction": pred, "MAE": MODELS[label]["mae"], "RMSE": MODELS[label]["rmse"], "R²": MODELS[label]["r2"]})
        comp = pd.DataFrame(rows).sort_values("Prediction", ascending=False).reset_index(drop=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Highest", f"${comp.Prediction.max():,.0f}")
        c2.metric("Lowest", f"${comp.Prediction.min():,.0f}")
        c3.metric("Model spread", f"${comp.Prediction.max()-comp.Prediction.min():,.0f}")
        c4.metric("Recommended", recommend_model())
        chart = comp.set_index("Model")[["Prediction"]].sort_values("Prediction")
        st.bar_chart(chart, horizontal=True)
        display = comp.copy()
        display["Prediction"] = display["Prediction"].map("${:,.0f}".format)
        display["MAE"] = display["MAE"].map("${:,.0f}".format)
        display["RMSE"] = display["RMSE"].map("${:,.0f}".format)
        display["R²"] = display["R²"].map("{:.3f}".format)
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export comparison CSV", comp.to_csv(index=False), "model_comparison.csv", "text/csv")

# ============================================================
# LEADERBOARD
# ============================================================
elif st.session_state.page == "🏆 Leaderboard":
    board = pd.DataFrame([
        {"Model": name, "MAE": v["mae"], "RMSE": v["rmse"], "R²": v["r2"], "Note": v["note"]}
        for name, v in MODELS.items()
    ]).sort_values("MAE").reset_index(drop=True)
    winner = board.iloc[0]["Model"]
    st.markdown(f'<div class="card"><h2>🏆 Current champion: {winner}</h2><p class="muted">Lowest MAE on the held-out evaluation set.</p></div>', unsafe_allow_html=True)
    a,b,c = st.columns(3)
    a.metric("Best MAE", f"${board.MAE.min():,.0f}")
    b.metric("Best RMSE", f"${board.RMSE.min():,.0f}")
    c.metric("Best R²", f"{board['R²'].max():.3f}")
    st.bar_chart(board.set_index("Model")[["MAE"]].sort_values("MAE"), horizontal=True)
    display = board.copy()
    display["MAE"] = display["MAE"].map("${:,.0f}".format)
    display["RMSE"] = display["RMSE"].map("${:,.0f}".format)
    display["R²"] = display["R²"].map("{:.3f}".format)
    st.dataframe(display, use_container_width=True, hide_index=True)

# ============================================================
# ANALYTICS
# ============================================================
elif st.session_state.page == "📈 Analytics":
    dataset = load_dataset()
    if dataset is None:
        st.warning("`data/ames.csv` was not found, so dataset analytics are unavailable.")
    else:
        st.markdown('<div class="section-label">Dataset intelligence</div>', unsafe_allow_html=True)
        a,b,c,d = st.columns(4)
        a.metric("Homes", f"{len(dataset):,}")
        b.metric("Median price", f"${dataset.SalePrice.median():,.0f}")
        c.metric("Mean price", f"${dataset.SalePrice.mean():,.0f}")
        d.metric("Features", f"{dataset.shape[1]}")
        st.markdown("#### Sale price distribution")
        hist = np.histogram(dataset["SalePrice"].dropna(), bins=30)
        hist_df = pd.DataFrame({"Sale price": hist[1][:-1], "Homes": hist[0]}).set_index("Sale price")
        st.bar_chart(hist_df)
        st.markdown("#### Missing values snapshot")
        missing = dataset.isna().sum().sort_values(ascending=False).head(15)
        st.dataframe(missing.rename("Missing values").to_frame(), use_container_width=True)

# ============================================================
# EXPLAIN
# ============================================================
elif st.session_state.page == "🔍 Explain":
    st.markdown('<div class="section-label">Model transparency</div>', unsafe_allow_html=True)
    label = st.selectbox("Choose model to inspect", list(MODELS.keys()))
    df, kind = get_feature_signal(label)
    if df is None:
        st.info("This model does not expose a simple feature-importance or coefficient view. Try Random Forest, Decision Tree, or a linear model.")
    else:
        df["abs"] = df.value.abs()
        top = df.sort_values("abs", ascending=False).head(15).sort_values("abs")
        if kind == "importance":
            st.caption("Higher importance means the feature contributed more to the tree model's split decisions.")
            st.bar_chart(top.set_index("feature")[["value"]].rename(columns={"value": "Importance"}), horizontal=True)
        else:
            st.caption("Coefficient sign indicates direction; magnitude indicates relative effect after preprocessing/scaling.")
            st.bar_chart(top.set_index("feature")[["value"]].rename(columns={"value": "Coefficient"}), horizontal=True)
        with st.expander("Show raw feature values"):
            st.dataframe(df.drop(columns="abs").sort_values("value", key=lambda s: s.abs(), ascending=False).head(30), use_container_width=True, hide_index=True)

# ============================================================
# HISTORY
# ============================================================
elif st.session_state.page == "🕘 History":
    st.markdown('<div class="section-label">Your recent activity</div>', unsafe_allow_html=True)
    if st.session_state.username == "Guest":
        st.info("Guest mode does not keep a persistent prediction history. Create an account to keep your session organised.")
    elif not st.session_state.history:
        st.info("No predictions yet. Run your first prediction from the Predict page.")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download history CSV", history_df.to_csv(index=False), "prediction_history.csv", "text/csv")
        if st.button("🗑️ Clear session history"):
            st.session_state.history = []
            st.rerun()

# ============================================================
# ABOUT
# ============================================================
elif st.session_state.page == "ℹ️ About":
    c1, c2 = st.columns([1.25, 1], gap="large")
    with c1:
        st.markdown('<div class="card"><h2>About Ames AI</h2><p>This dashboard is built around nine regression models trained for Ames house-price prediction. It lets you predict, compare, rank, inspect and export results from one clean workspace.</p><h3>What is included?</h3><p>🔹 9 regression models<br>🔹 Guest / account experience<br>🔹 Light / dark theme<br>🔹 Interactive prediction form<br>🔹 Model comparison<br>🔹 Leaderboard<br>🔹 Dataset analytics<br>🔹 Feature explanations<br>🔹 Prediction history<br>🔹 CSV exports</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h3>Dataset</h3><p class="muted">Ames, Iowa residential sales, 2006–2010.</p><h3>Model stack</h3><p><span class="small-pill">Python</span> <span class="small-pill">pandas</span> <span class="small-pill">scikit-learn</span> <span class="small-pill">Streamlit</span></p><h3>Project files</h3><p class="muted">Models are loaded from <code>models/*.joblib</code>. Dataset analytics use <code>data/ames.csv</code> when available.</p></div>', unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.caption(f"Ames AI · {st.session_state.full_name or 'Guest User'} · Local ML Dashboard · {datetime.now().strftime('%Y')}")
