# Inspiration - Inspiracija
## https://github.com/filipseva96/loto-serbia-ai-main



"""
cd /
streamlit run lotto_ai/gui/app.py
"""



# promena sifre za ulaz u aplikaciju Enter password to access 
# u .streamlit/secrets.toml
# app_password = "1234"
# trenutna sifra je "1234"
# pa ako hoces promeni i ponovo pokreni aplikaciju
# streamlit run lotto_ai/gui/app.py



"""
Streamlit UI for Loto Serbia Portfolio Optimizer - v4.0
Coverage optimization, wheeling, Monte Carlo evaluation, honest mathematics.
"""
import streamlit as st
import sys
from pathlib import Path
import json
import pandas as pd
from datetime import datetime, timedelta

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from lotto_ai.core.db import init_db, get_session, Draw, Prediction, PredictionResult
from lotto_ai.core.tracker import PredictionTracker, PlayedTicketsTracker
from lotto_ai.core.models import generate_adaptive_portfolio, portfolio_statistics
from lotto_ai.core.coverage_optimizer import generate_random_portfolio
from lotto_ai.core.math_engine import (
    expected_value_per_ticket,
    portfolio_expected_value,
    portfolio_monte_carlo_statistics,
    compare_portfolio_to_random_baseline,
    match_probability,
    kelly_criterion_lottery,
    test_lottery_fairness,
)
from lotto_ai.core.wheeling import generate_abbreviated_wheel, wheel_cost_estimate
from lotto_ai.features.features import build_feature_matrix, load_draws, get_number_summary
from lotto_ai.config import (
    APP_TITLE, APP_SUBTITLE, APP_VERSION,
    IS_CLOUD, logger, DRAW_DAYS, DRAW_HOUR,
    NUMBERS_PER_DRAW, MAX_NUMBER, MIN_NUMBER,
    PRIZE_TABLE, TICKET_COST, TOTAL_COMBINATIONS,
    DEFAULT_OPTIMIZER_WEIGHTS,
    DEFAULT_OPTIMIZER_CONSTRAINTS,
    PORTFOLIO_SIMULATIONS_FAST,
    RANDOM_BASELINE_PORTFOLIOS,
)

init_db()


# ============================================================================
# HELPERS
# ============================================================================

def get_next_draw_info():
    now = datetime.now()
    current_weekday = now.weekday()
    current_hour = now.hour

    if current_weekday in DRAW_DAYS and current_hour < DRAW_HOUR:
        hours_until = DRAW_HOUR - current_hour
        return now.strftime("%Y-%m-%d"), True, hours_until

    days_ahead = 1
    while days_ahead <= 7:
        next_date = now + timedelta(days=days_ahead)
        if next_date.weekday() in DRAW_DAYS:
            draw_dt = next_date.replace(hour=DRAW_HOUR, minute=0, second=0, microsecond=0)
            hours_until = (draw_dt - now).total_seconds() / 3600
            return next_date.strftime("%Y-%m-%d"), False, hours_until
        days_ahead += 1

    fallback = now + timedelta(days=1)
    return fallback.strftime("%Y-%m-%d"), False, 24


def get_next_draw_date():
    return get_next_draw_info()[0]


def format_draw_info_message(draw_date, is_today, hours_until):
    day_names = {
        0: "Ponedeljak", 1: "Utorak", 2: "Sreda",
        3: "Četvrtak", 4: "Petak", 5: "Subota", 6: "Nedelja"
    }
    if is_today:
        if hours_until > 2:
            return f"🎯 **DANAŠNJE IZVLAČENJE** - {draw_date} u {DRAW_HOUR:02d}:00 (za ~{int(hours_until)}h)"
        return f"⚡ **DANAŠNJE IZVLAČENJE** - {draw_date} - USKORO!"
    draw_dt = datetime.strptime(draw_date, "%Y-%m-%d")
    day_name = day_names.get(draw_dt.weekday(), draw_dt.strftime("%A"))
    days_until = max(1, int(hours_until / 24))
    return f"📅 Sledeće izvlačenje: **{day_name}, {draw_date}** (za {days_until} dan{'a' if days_until != 1 else ''})"


def get_secret_password():
    """
    Robust password loader.
    Returns (password, error_message)
    """
    try:
        # Temporary debug info - remove later if you want
        if "debug_secrets_checked" not in st.session_state:
            st.session_state["debug_secrets_checked"] = True
            try:
                secret_keys = list(st.secrets.keys())
                logger.info(f"Streamlit secrets keys available: {secret_keys}")
            except Exception as e:
                logger.error(f"Could not inspect st.secrets keys: {e}")

        if "app_password" not in st.secrets:
            return None, "Lozinka nije pronađena u Streamlit secrets pod ključem 'app_password'."

        value = st.secrets["app_password"]

        if value is None:
            return None, "app_password postoji ali je None."

        value = str(value).strip()
        if not value:
            return None, "app_password postoji ali je prazan string."

        return value, None

    except Exception as e:
        return None, f"Greška pri čitanju secrets: {e}"


def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    correct_password, secret_error = get_secret_password()

    if not st.session_state["password_correct"]:
        st.markdown("### 🔐 Loto Serbia Portfolio Optimizer - Prijava")
        entered = st.text_input("Unesite lozinku", type="password", key="password_input")

        if secret_error:
            st.error("❌ Lozinka nije podešena u Streamlit secrets.")
            st.caption(secret_error)

            if not IS_CLOUD:
                st.info('Lokalno napravite `.streamlit/secrets.toml` sa: app_password = "your_password"')
            else:
                st.info('Na Streamlit Cloud-u otvorite App Settings → Secrets i dodajte: app_password = "your_password"')
            return False

        if st.button("Prijava", type="primary"):
            if entered == correct_password:
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("❌ Pogrešna lozinka")
                return False

        return False

    return True


def format_ticket_html(ticket):
    return "".join([f'<span class="number-ball">{n:02d}</span>' for n in ticket])


def deserialize_result_payload(result):
    """
    Supports both old format (list of ticket matches)
    and new format (dict with ticket_matches + empirical_random_baseline).
    """
    try:
        payload = json.loads(result.ticket_matches) if result.ticket_matches else None
    except Exception:
        payload = None

    if isinstance(payload, list):
        return {
            "ticket_matches": payload,
            "empirical_random_baseline": None
        }
    if isinstance(payload, dict):
        return payload

    return {
        "ticket_matches": [],
        "empirical_random_baseline": None
    }


# ============================================================================
# ACCESS CONTROL
# ============================================================================

if not check_password():
    st.stop()

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #c9082a 0%, #17408b 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .ticket-box {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #c9082a;
        margin: 0.7rem 0;
    }
    .ticket-box.selected {
        background: #e3f2fd;
        border-left: 5px solid #17408b;
    }
    .number-ball {
        display: inline-block;
        background: linear-gradient(135deg, #c9082a 0%, #17408b 100%);
        color: white;
        font-weight: bold;
        font-size: 1.05rem;
        padding: 0.45rem 0.75rem;
        border-radius: 50%;
        margin: 0.18rem;
        min-width: 42px;
        text-align: center;
    }
    .math-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    .honest-box {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 0.5rem 0;
    }
    .good-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .neutral-box {
        background: #e9ecef;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #6c757d;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

st.markdown(f"""
<div class="main-header">
    <h1>🎰 {APP_TITLE}</h1>
    <p style="font-size: 1.2rem; margin: 0;">{APP_SUBTITLE}</p>
    <p style="font-size: 0.9rem; margin: 0; opacity: 0.9;">LOTO 7/39 • v{APP_VERSION}</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### ℹ️ Kako Radi")
    st.info("""
🧮 **Matematički pristup:**

1️⃣ **Optimizacija Pokrivanja**  
Maksimizuje pokrivanje parova/trojki i smanjuje preklapanje tiketa

2️⃣ **Monte Carlo Evaluacija**  
Procena realnih šansi portfolija na osnovu simulacija

3️⃣ **Wheeling Sistem**  
Kombinatorne garancije pod uslovom da su ključni brojevi pogođeni

4️⃣ **Fer Igra Testovi**  
Dijagnostika da li istorijski podaci izgledaju nasumično

5️⃣ **Odgovorno Igranje**  
Bankroll disciplina i negativan očekivani povraćaj

⚠️ Ova aplikacija ne predviđa buduće brojeve.
Optimizuje strukturu portfolija, ne budućnost.
    """)

    st.markdown("---")
    page = st.radio("📄 Stranica", [
        "🎲 Portfolio Generator",
        "🎯 Wheeling Sistem",
        "📊 Matematika",
        "🔬 Fer Igra Test",
        "📈 Istorija",
    ])

    st.markdown("---")
    st.markdown("### ⏰ Sledeće Izvlačenje")
    draw_date, is_today, hours_until = get_next_draw_info()
    if is_today:
        st.success(f"**DANAS** u {DRAW_HOUR:02d}:00")
        progress_val = max(0.0, min((DRAW_HOUR - hours_until) / DRAW_HOUR, 1.0))
        st.progress(progress_val)
        st.caption(f"~{int(hours_until)} sati preostalo")
    else:
        draw_dt = datetime.strptime(draw_date, "%Y-%m-%d")
        day_names = {0: "Ponedeljak", 1: "Utorak", 2: "Sreda", 3: "Četvrtak", 4: "Petak", 5: "Subota", 6: "Nedelja"}
        day_name = day_names.get(draw_dt.weekday(), "")
        days_until = max(1, int(hours_until / 24))
        st.info(f"**{day_name}**\n{draw_date}")
        st.caption(f"Za {days_until} dan{'a' if days_until != 1 else ''}")

    st.markdown("---")
    st.markdown("### 📡 Podaci")

    session_info = get_session()
    try:
        total_draws = session_info.query(Draw).count()
        latest = session_info.query(Draw).order_by(Draw.draw_date.desc()).first()
        latest_date = latest.draw_date if latest else "N/A"
        latest_nums = latest.get_numbers() if latest else []
    finally:
        session_info.close()

    st.caption(f"📊 Izvlačenja: **{total_draws}**")
    st.caption(f"📅 Poslednje: **{latest_date}**")
    if latest_nums:
        st.caption(f"🔢 {latest_nums}")
    if IS_CLOUD:
        st.caption("☁️ Cloud režim")

    if st.button("🔄 Ažuriraj Podatke"):
        with st.spinner("Preuzimanje..."):
            try:
                from lotto_ai.scraper.serbia_scraper import scrape_recent_draws
                n_new = scrape_recent_draws(max_pdfs=20)
                if n_new > 0:
                    st.success(f"✅ +{n_new} novih!")
                    st.rerun()
                else:
                    st.warning("Nema novih izvlačenja ili server nije dostupan.")
            except Exception as e:
                st.error(f"Greška: {str(e)}")

    with st.expander("✏️ Ručni unos izvlačenja"):
        if IS_CLOUD:
            st.warning(
                "⚠️ Na cloud-u ručni unos traje samo do restarta aplikacije. "
                "Za trajno čuvanje koristite lokalni rad i GitHub."
            )

        manual_date = st.text_input("Datum (YYYY-MM-DD)", placeholder="2026-02-25", key="manual_date")
        manual_nums = st.text_input("Brojevi (zarezom razdvojeni)", placeholder="1, 13, 16, 20, 25, 26, 28", key="manual_nums")
        manual_kolo = st.text_input("Kolo (opciono)", placeholder="17", key="manual_kolo")

        if st.button("💾 Sačuvaj"):
            if not manual_date or not manual_nums:
                st.error("Unesite datum i brojeve")
            else:
                try:
                    from lotto_ai.scraper.serbia_scraper import add_draw_manually
                    nums = [int(x.strip()) for x in manual_nums.split(",")]
                    kolo = int(manual_kolo) if manual_kolo.strip() else None

                    if len(nums) != 7:
                        st.error(f"Potrebno je 7 brojeva, uneto {len(nums)}")
                    elif add_draw_manually(manual_date.strip(), nums, kolo):
                        st.success(f"✅ Sačuvano: {manual_date} {sorted(nums)}")
                        st.rerun()
                    else:
                        st.warning("Već postoji ili su podaci neispravni")
                except ValueError as e:
                    st.error(f"Format greška: {e}")
                except Exception as e:
                    st.error(f"Greška: {e}")

# ============================================================================
# SESSION STATE
# ============================================================================

if "generated_tickets" not in st.session_state:
    st.session_state.generated_tickets = None
if "selected_tickets" not in st.session_state:
    st.session_state.selected_tickets = []
if "prediction_id" not in st.session_state:
    st.session_state.prediction_id = None
if "next_draw" not in st.session_state:
    st.session_state.next_draw = None
if "weights_meta" not in st.session_state:
    st.session_state.weights_meta = None
if "performance" not in st.session_state:
    st.session_state.performance = None
if "current_strategy" not in st.session_state:
    st.session_state.current_strategy = "coverage_optimized"
if "mc_stats" not in st.session_state:
    st.session_state.mc_stats = None
if "baseline_stats" not in st.session_state:
    st.session_state.baseline_stats = None

# ============================================================================
# PAGE: PORTFOLIO GENERATOR
# ============================================================================

if page == "🎲 Portfolio Generator":
    st.markdown("### 🎲 Portfolio Generator")

    st.markdown("""
    <div class="good-box">
    <strong>✅ Šta ovo radi:</strong> Generiše portfolio tiketa sa malim međusobnim preklapanjem
    i visokim pokrivanjem parova/trojki. To može poboljšati šansu da bar jedan tiket uhvati
    3+ pogodaka u odnosu na nasumično preklapajuće tikete.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="honest-box">
    <strong>⚠️ Iskreno upozorenje:</strong> Ova aplikacija ne predviđa buduće brojeve.
    Ako je lutrija fer, svako izvlačenje je nezavisno. Ovde optimizujete portfolio,
    a ne budućnost.
    </div>
    """, unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        n_tickets = st.slider("Broj tiketa", 3, 20, 7)
    with col_s2:
        strategy = st.selectbox(
            "Strategija",
            ["coverage_optimized", "hybrid", "pure_random"],
            format_func=lambda x: {
                "coverage_optimized": "🎯 Optimizovano Pokrivanje",
                "hybrid": "🔄 Hibrid",
                "pure_random": "🎲 Čisto Nasumično",
            }[x],
        )

    st.markdown("#### ⚙️ Podešavanje Optimizatora")
    c1, c2, c3 = st.columns(3)

    with c1:
        w_pairs = st.slider("Težina parova", 0.1, 3.0, float(DEFAULT_OPTIMIZER_WEIGHTS["w_pairs"]), 0.1)
        w_triples = st.slider("Težina trojki", 0.0, 2.0, float(DEFAULT_OPTIMIZER_WEIGHTS["w_triples"]), 0.1)

    with c2:
        w_overlap = st.slider("Kazna za preklapanje", 0.0, 10.0, float(DEFAULT_OPTIMIZER_WEIGHTS["w_overlap"]), 0.5)
        w_odd_even = st.slider("Kazna za odd/even ekstrem", 0.0, 10.0, float(DEFAULT_OPTIMIZER_WEIGHTS["w_odd_even_penalty"]), 0.5)

    with c3:
        w_sum = st.slider("Kazna za ekstreman zbir", 0.0, 10.0, float(DEFAULT_OPTIMIZER_WEIGHTS["w_sum_penalty"]), 0.5)
        mc_candidates = st.slider("Broj kandidata po koraku", 200, 5000, 1500, 100)

    optimizer_weights = {
        "w_pairs": w_pairs,
        "w_triples": w_triples,
        "w_overlap": w_overlap,
        "w_odd_even_penalty": w_odd_even,
        "w_sum_penalty": w_sum,
    }

    optimizer_constraints = DEFAULT_OPTIMIZER_CONSTRAINTS.copy()

    approx = portfolio_expected_value(n_tickets)
    col_ev1, col_ev2, col_ev3, col_ev4 = st.columns(4)
    col_ev1.metric("Ukupna Cena", f"{approx['total_cost']:,.0f} RSD")
    col_ev2.metric("Aproks. Očekivani Povraćaj", f"{approx['total_ev']:,.1f} RSD")
    col_ev3.metric("Aproks. ROI", f"{approx['roi_percent']:.1f}%")
    col_ev4.metric("Aproks. Šansa za 3+", f"{approx['prob_any_3plus']:.1%}")

    st.caption("Napomena: gornje portfolio verovatnoće koriste aproksimaciju nezavisnosti. Nakon generisanja portfolija prikazaće se Monte Carlo procena za konkretan portfolio.")

    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        generate_clicked = st.button("🎲 GENERIŠI PORTFOLIO", type="primary")

    if generate_clicked:
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            tracker = PredictionTracker()

            status_text.text("📊 Evaluacija prethodnih portfolija...")
            progress_bar.progress(15)
            tracker.auto_evaluate_pending()

            status_text.text("🧮 Učitavanje istorijskih feature-a...")
            progress_bar.progress(30)
            features = build_feature_matrix()

            status_text.text("🎟️ Generisanje portfolija...")
            progress_bar.progress(50)
            portfolio, metadata = generate_adaptive_portfolio(
                features=features,
                n_tickets=n_tickets,
                use_adaptive=True,
                strategy=strategy,
                optimizer_weights=optimizer_weights,
                optimizer_constraints=optimizer_constraints,
                monte_carlo_samples=mc_candidates,
            )

            status_text.text("🎲 Monte Carlo evaluacija portfolija...")
            progress_bar.progress(70)
            mc_stats = portfolio_monte_carlo_statistics(
                portfolio,
                n_simulations=PORTFOLIO_SIMULATIONS_FAST,
            )

            def _random_generator():
                rp, _ = generate_random_portfolio(len(portfolio))
                return rp

            baseline_stats = compare_portfolio_to_random_baseline(
                portfolio=portfolio,
                random_portfolio_generator=_random_generator,
                n_random_portfolios=min(100, RANDOM_BASELINE_PORTFOLIOS),
                n_simulations_per_portfolio=2000,
            )

            status_text.text("💾 Čuvanje portfolija...")
            progress_bar.progress(85)
            next_draw = get_next_draw_date()
            prediction_id = tracker.save_prediction(
                target_draw_date=next_draw,
                strategy_name=strategy,
                tickets=portfolio,
                model_version=f"{APP_VERSION}_optimizer",
                metadata={
                    **(metadata or {}),
                    "monte_carlo_stats": mc_stats,
                    "random_baseline_summary": baseline_stats,
                },
            )

            perf = tracker.get_strategy_performance(strategy, window=50)

            st.session_state.generated_tickets = portfolio
            st.session_state.prediction_id = prediction_id
            st.session_state.next_draw = next_draw
            st.session_state.selected_tickets = []
            st.session_state.weights_meta = metadata
            st.session_state.performance = perf
            st.session_state.current_strategy = strategy
            st.session_state.mc_stats = mc_stats
            st.session_state.baseline_stats = baseline_stats

            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()

            stats = portfolio_statistics(portfolio)
            st.success(f"✅ Generisano {len(portfolio)} tiketa!")
            st.info(
                f"📊 Pair coverage: {stats['pair_coverage_pct']:.1f}% | "
                f"Triple coverage: {stats['triple_coverage_pct']:.2f}% | "
                f"Jedinstveni brojevi: {stats['unique_numbers']}/{MAX_NUMBER} | "
                f"Prosečno preklapanje: {stats['avg_overlap']:.1f}"
            )
            st.rerun()

        except Exception as e:
            logger.error(f"Error generating portfolio: {e}")
            st.error(f"❌ Greška: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    if st.session_state.generated_tickets:
        st.markdown("---")

        portfolio = st.session_state.generated_tickets
        current_strategy = st.session_state.get("current_strategy", "coverage_optimized")
        perf = st.session_state.get("performance")
        mc_stats = st.session_state.get("mc_stats")
        baseline = st.session_state.get("baseline_stats")

        # Strategy history
        if perf:
            st.markdown("### 📊 Istorija Strategije")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Portfolija", perf["n_predictions"])
            c2.metric("Prosečno najbolji tiket", f"{perf['avg_best_match']:.2f}/7")
            c3.metric("Stopa 3+", f"{perf['hit_rate_3plus']:.1%}")
            c4.metric("Najbolje ikad", f"{perf['best_ever']}/7")
            c5.metric("Ukupna isplata", f"{perf['total_prize_won']:,.0f} RSD")

            c6, c7 = st.columns(2)
            with c6:
                rbm = perf.get("random_best_match_mean")
                if rbm is not None:
                    st.metric("Empirijski random best-match mean", f"{rbm:.2f}")
            with c7:
                st.metric("Outperform random best rate", f"{perf.get('outperform_random_best_rate', 0):.1%}")

        # Monte Carlo for generated portfolio
        if mc_stats:
            st.markdown("### 🎲 Monte Carlo Evaluacija Ovog Portfolija")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Šansa za 3+", f"{mc_stats['prob_any_3plus']:.2%}")
            m2.metric("Šansa za 4+", f"{mc_stats['prob_any_4plus']:.4%}")
            m3.metric("Očekivana ukupna isplata", f"{mc_stats['expected_total_prize']:.2f} RSD")
            m4.metric("Očekivani neto rezultat", f"{mc_stats['expected_net']:.2f} RSD")

            st.markdown(f"""
            <div class="math-box">
            <strong>Monte Carlo:</strong> {mc_stats['n_simulations']:,} simulacija za konkretan portfolio.
            Ovo je realnija procena od aproksimacije nezavisnih tiketa, jer koristi upravo vaše generisane tikete.
            </div>
            """, unsafe_allow_html=True)

        if baseline:
            st.markdown("### 📈 Poređenje sa Nasumičnim Portfolijima")
            cmp = baseline["comparison"]

            b1, b2 = st.columns(2)
            with b1:
                c = cmp["prob_any_3plus"]
                st.metric("Percentil za 3+ vs random", f"{c['percentile_vs_random']:.1f}")
                st.caption(f"Portfolio 3+ = {c['target']:.4f}, random mean = {c['random_mean']:.4f}")

                c = cmp["avg_best_match"]
                st.metric("Percentil za avg best match vs random", f"{c['percentile_vs_random']:.1f}")
                st.caption(f"Portfolio avg best = {c['target']:.4f}, random mean = {c['random_mean']:.4f}")

            with b2:
                c = cmp["prob_any_4plus"]
                st.metric("Percentil za 4+ vs random", f"{c['percentile_vs_random']:.1f}")
                st.caption(f"Portfolio 4+ = {c['target']:.6f}, random mean = {c['random_mean']:.6f}")

                c = cmp["expected_total_prize"]
                st.metric("Percentil za očekivanu isplatu vs random", f"{c['percentile_vs_random']:.1f}")
                st.caption(f"Portfolio prize EV = {c['target']:.2f}, random mean = {c['random_mean']:.2f}")

            st.markdown("""
            <div class="neutral-box">
            <strong>Kako čitati percentile:</strong> 50 ≈ prosečan random portfolio,
            80+ znači da je ovaj portfolio bolji od većine random portfolija po tom kriterijumu.
            To i dalje nije predikcija budućih brojeva — samo kvalitetnija konstrukcija portfolija.
            </div>
            """, unsafe_allow_html=True)

        # Ticket display
        st.markdown("---")
        st.markdown("### 🎟️ Izaberite Tikete za Igranje")
        draw_date_info, is_today_info, hours_until_info = get_next_draw_info()
        st.info(format_draw_info_message(draw_date_info, is_today_info, hours_until_info))

        for i, ticket in enumerate(portfolio, 1):
            col_check, col_ticket = st.columns([0.7, 9.3])
            with col_check:
                is_selected = st.checkbox(
                    f"#{i}",
                    key=f"ticket_{i}",
                    value=i in st.session_state.selected_tickets,
                )
                if is_selected and i not in st.session_state.selected_tickets:
                    st.session_state.selected_tickets.append(i)
                elif not is_selected and i in st.session_state.selected_tickets:
                    st.session_state.selected_tickets.remove(i)

            with col_ticket:
                label = {
                    "coverage_optimized": "Coverage-Optimized",
                    "hybrid": "Hibrid",
                    "pure_random": "Nasumičan",
                }.get(current_strategy, "Tiket")
                box_class = "ticket-box selected" if is_selected else "ticket-box"
                st.markdown(
                    f'<div class="{box_class}"><strong>Tiket {i}</strong> ({label})<br>{format_ticket_html(ticket)}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        col_play, col_download = st.columns(2)

        n_selected = len(st.session_state.selected_tickets)

        with col_play:
            if st.button(
                f"✅ OBELEŽITE {n_selected} TIKETA KAO ODIGRANO",
                type="primary",
                disabled=n_selected == 0,
            ):
                played_tracker = PlayedTicketsTracker()
                selected_nums = [portfolio[i - 1] for i in st.session_state.selected_tickets]
                played_tracker.save_played_tickets(
                    st.session_state.prediction_id,
                    selected_nums,
                    st.session_state.next_draw,
                )
                st.success(f"✅ Obeleženo {n_selected} tiketa kao odigrano!")
                st.balloons()

        with col_download:
            if st.session_state.selected_tickets:
                ev_info = expected_value_per_ticket()
                ticket_text = (
                    f"{APP_TITLE} v{APP_VERSION}\n"
                    f"Datum izvlačenja: {st.session_state.next_draw}\n"
                    f"Generisano: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                    f"Strategija: {current_strategy}\n"
                    f"{'=' * 50}\n"
                    f"IZABRANI TIKETI\n"
                    f"{'=' * 50}\n"
                )
                for idx, i in enumerate(st.session_state.selected_tickets, 1):
                    ticket = portfolio[i - 1]
                    ticket_text += f"Tiket {idx}: {' - '.join(f'{n:02d}' for n in ticket)}\n"

                if mc_stats:
                    ticket_text += (
                        f"\nMonte Carlo procena:\n"
                        f"Šansa za 3+: {mc_stats['prob_any_3plus']:.2%}\n"
                        f"Šansa za 4+: {mc_stats['prob_any_4plus']:.4%}\n"
                        f"Očekivani neto rezultat: {mc_stats['expected_net']:.2f} RSD\n"
                    )

                ticket_text += (
                    f"\n⚠️ Igrajte odgovorno!\n"
                    f"EV po tiketu: {ev_info['net_ev']:.2f} RSD\n"
                    f"Ovo nije predikcija budućih brojeva.\n"
                )

                st.download_button(
                    "💾 Preuzmite Tikete",
                    data=ticket_text,
                    file_name=f"portfolio_{st.session_state.next_draw}.txt",
                    mime="text/plain",
                )

# ============================================================================
# PAGE: WHEELING
# ============================================================================

elif page == "🎯 Wheeling Sistem":
    st.markdown("### 🎯 Wheeling Sistem")

    st.markdown("""
    <div class="good-box">
    <strong>✅ Šta je wheeling?</strong> Izaberete skup ključnih brojeva, a sistem pravi
    manji broj tiketa sa kombinatornim garancijama:
    ako dovoljno vaših ključnih brojeva zaista bude izvučeno, bar jedan tiket će garantovano
    imati određen broj pogodaka.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="honest-box">
    <strong>⚠️ Važno:</strong> Wheeling ne bira "bolje" brojeve sam po sebi.
    On efikasno organizuje vaše izabrane brojeve.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Izaberite ključne brojeve")

    if "wheel_key_numbers" not in st.session_state:
        st.session_state.wheel_key_numbers = []

    selected_key_numbers = list(st.session_state.wheel_key_numbers)
    cols_per_row = 13

    for row_start in range(MIN_NUMBER, MAX_NUMBER + 1, cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            num = row_start + j
            if num > MAX_NUMBER:
                break
            with col:
                was_selected = num in selected_key_numbers
                is_now = st.checkbox(f"{num:02d}", key=f"wheel_num_{num}", value=was_selected)
                if is_now and num not in selected_key_numbers:
                    selected_key_numbers.append(num)
                elif not is_now and num in selected_key_numbers:
                    selected_key_numbers.remove(num)

    st.session_state.wheel_key_numbers = sorted(selected_key_numbers)
    n_keys = len(selected_key_numbers)

    if n_keys > 0:
        st.info(f"Izabrano: **{n_keys}** ključnih brojeva: {', '.join(str(n) for n in sorted(selected_key_numbers))}")
    else:
        st.warning("Izaberite bar 4 ključna broja.")

    if n_keys >= 4:
        max_guarantee_hit = min(7, n_keys)
        col_w1, col_w2 = st.columns(2)

        with col_w1:
            guarantee_if_hit = st.slider(
                "Koliko vaših brojeva mora biti izvučeno",
                min_value=3,
                max_value=max_guarantee_hit,
                value=3,
            )

        with col_w2:
            guarantee_match = st.slider(
                "Garantovanih pogodaka na tiketu",
                min_value=3,
                max_value=guarantee_if_hit,
                value=3,
            )

        estimate = wheel_cost_estimate(n_keys, guarantee_if_hit, guarantee_match)
        st.markdown(f"""
        <div class="math-box">
        📊 Procena: Potrebno otprilike {estimate['estimated_min_tickets']} - {estimate['estimated_max_tickets']} tiketa
        za pokrivanje {estimate['subsets_to_cover']:,} podskupova.
        </div>
        """, unsafe_allow_html=True)

        max_wheel_tickets = st.slider("Maksimalan broj tiketa", 5, 100, 30)

        if st.button("🎯 GENERIŠI WHEELING TIKETE", type="primary"):
            try:
                with st.spinner("Generisanje wheeling sistema..."):
                    tickets, guarantee = generate_abbreviated_wheel(
                        sorted(selected_key_numbers),
                        guarantee_if_hit=guarantee_if_hit,
                        guarantee_match=guarantee_match,
                        max_tickets=max_wheel_tickets,
                    )

                if guarantee["verified"]:
                    st.success("✅ Garancija verifikovana!")
                else:
                    st.warning(f"⚠️ Nepotpuna garancija - {guarantee.get('warning', '')}")

                st.markdown(f"""
                <div class="good-box">
                <strong>📜 Garancija:</strong> {guarantee['guarantee']}
                <br>Tiketa: {guarantee['n_tickets']} |
                Pokrivanje: {guarantee['coverage_pct']:.1f}%
                </div>
                """, unsafe_allow_html=True)

                for i, ticket in enumerate(tickets, 1):
                    key_in = sum(1 for n in ticket if n in selected_key_numbers)
                    st.markdown(
                        f'<div class="ticket-box"><strong>Tiket {i}</strong> ({key_in} ključnih)<br>{format_ticket_html(ticket)}</div>',
                        unsafe_allow_html=True,
                    )

                tracker_w = PredictionTracker()
                next_draw_w = get_next_draw_date()
                pred_id = tracker_w.save_prediction(
                    target_draw_date=next_draw_w,
                    strategy_name=f"wheel_{guarantee_if_hit}of{n_keys}",
                    tickets=tickets,
                    model_version=f"{APP_VERSION}_wheel",
                    metadata=guarantee,
                )
                st.caption(f"Sačuvano kao portfolio #{pred_id}")

                ticket_text = (
                    f"WHEELING SISTEM - {APP_TITLE}\n"
                    f"Ključni brojevi: {sorted(selected_key_numbers)}\n"
                    f"Garancija: {guarantee['guarantee']}\n"
                    f"Verifikovano: {'DA' if guarantee['verified'] else 'NE'}\n"
                    f"{'=' * 40}\n"
                )
                for i, ticket in enumerate(tickets, 1):
                    ticket_text += f"Tiket {i}: {' - '.join(f'{n:02d}' for n in ticket)}\n"

                st.download_button(
                    "💾 Preuzmite Wheeling Tikete",
                    data=ticket_text,
                    file_name=f"wheeling_{next_draw_w}.txt",
                    mime="text/plain",
                )

            except Exception as e:
                st.error(f"Greška: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    elif n_keys == 3:
        st.info("Sa samo 3 ključna broja postoji samo 1 trojka. Izaberite bar 4 broja.")
    elif n_keys > 0:
        st.warning(f"Potrebno je bar 4 ključna broja. Trenutno: {n_keys}")

# ============================================================================
# PAGE: MATHEMATICS
# ============================================================================

elif page == "📊 Matematika":
    st.markdown("### 📊 Matematika Lutrije 7/39")

    st.markdown("""
    <div class="honest-box">
    <strong>⚠️ Matematička realnost:</strong> Svako izvlačenje je nezavisno.
    Prošli rezultati ne menjaju verovatnoće sledećeg izvlačenja.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🎲 Tačne verovatnoće po tiketu")
    prob_data = []
    for matches in range(0, NUMBERS_PER_DRAW + 1):
        p = match_probability(matches)
        prize = PRIZE_TABLE.get(matches, 0)
        prob_data.append({
            "Pogodaka": f"{matches}/7",
            "Verovatnoća": f"{p:.10f}",
            "Šanse": f"1 od {int(1/p):,}" if p > 0 else "-",
            "Nagrada (RSD)": f"{prize:,}" if prize > 0 else "-",
            "Doprinos EV": f"{p * prize:.2f} RSD" if prize > 0 else "-",
        })
    st.table(prob_data)

    ev_data = expected_value_per_ticket()
    col_ev1, col_ev2, col_ev3 = st.columns(3)
    col_ev1.metric("Cena tiketa", f"{TICKET_COST} RSD")
    col_ev2.metric("Očekivani povraćaj", f"{ev_data['expected_value']:.2f} RSD")
    col_ev3.metric("ROI", f"{ev_data['roi_percent']:.1f}%")

    st.markdown(f"""
    <div class="math-box">
    💰 Za svaki tiket od {TICKET_COST} RSD matematički očekujete povraćaj od {ev_data['expected_value']:.2f} RSD.
    To znači prosečan gubitak od {abs(ev_data['net_ev']):.2f} RSD po tiketu.
    <br><br>
    Ukupan broj kombinacija: <strong>{TOTAL_COMBINATIONS:,}</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📦 Aproksimativni kalkulator portfolija")
    n_calc = st.slider("Broj tiketa za kalkulaciju", 1, 50, 10, key="calc_tickets")
    pev = portfolio_expected_value(n_calc)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ukupna cena", f"{pev['total_cost']:,} RSD")
    c2.metric("Aproks. šansa za 3+", f"{pev['prob_any_3plus']:.2%}")
    c3.metric("Aproks. šansa za 4+", f"{pev['prob_any_4plus']:.4%}")
    c4.metric("Aproks. šansa za 5+", f"{pev['prob_any_5plus']:.6%}")

    st.caption("Za konkretan portfolio koristite Monte Carlo procenu na stranici Portfolio Generator.")

    st.markdown("#### 🏦 Bankroll upravljanje")
    bankroll = st.number_input("Vaš budžet (RSD)", min_value=1000, max_value=1000000, value=10000, step=1000)
    kelly = kelly_criterion_lottery(bankroll)
    st.markdown(f"""
    <div class="math-box">
    <strong>Kelly Criterion kaže:</strong> {kelly['kelly_says']}
    <br><br>
    <strong>Preporuka:</strong> {kelly['recommendation']}
    <br><br>
    Maksimalan odgovoran ulog: <strong>{kelly['entertainment_budget']:.0f} RSD</strong>
    ({kelly['max_responsible_tickets']} tiketa)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📈 Statistika brojeva (opisna)")
    st.caption("⚠️ Ovo je opisna statistika, ne predikcija.")

    try:
        summary = get_number_summary(n_recent=20)
        if summary:
            rows = []
            for num in sorted(summary.keys()):
                data = summary[num]
                rows.append({
                    "Broj": num,
                    "Ukupno": data["total_appearances"],
                    "Frekvencija": f"{data['overall_frequency']:.3f}",
                    "Očekivano": f"{data['expected_frequency']:.3f}",
                    "Odstupanje": f"{data['deviation']:+.3f}",
                    "Razmak": data["current_gap"],
                    "Status": data["status"],
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        else:
            st.info("Nema podataka.")
    except Exception as e:
        st.warning(f"Greška pri učitavanju statistike: {e}")

# ============================================================================
# PAGE: FAIRNESS
# ============================================================================

elif page == "🔬 Fer Igra Test":
    st.markdown("### 🔬 Dijagnostički Test Fer Igre")

    st.markdown("""
    <div class="good-box">
    <strong>✅ Svrha:</strong> Ovi testovi proveravaju da li istorijski podaci izgledaju
    približno kao fer nasumičan proces. Ovo je dijagnostika, ne dokaz eksploatabilnosti.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="honest-box">
    <strong>⚠️ Važno:</strong> Ako testovi ne nalaze anomalije, to ne znači da je "dokazano"
    da je sve savršeno nasumično. Ako nađu anomaliju, to opet ne znači automatski da postoji
    profitabilna strategija. Tumačite oprezno.
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔬 POKRENI TESTOVE", type="primary"):
        with st.spinner("Pokrećem testove..."):
            try:
                df = load_draws()
                if len(df) < 30:
                    st.warning(f"Potrebno minimum 30 izvlačenja. Trenutno: {len(df)}")
                else:
                    results = test_lottery_fairness(df)
                    overall = results["overall"]

                    if overall.get("is_fair", True):
                        st.success(overall["conclusion"])
                    else:
                        st.warning(overall["conclusion"])

                    st.info(f"Analizirano: {results['n_draws']} izvlačenja")

                    st.markdown("#### 1️⃣ Hi-kvadrat test uniformnosti")
                    chi = results["chi_square"]
                    st.write(f"- **Statistika:** {chi['statistic']:.2f}")
                    st.write(f"- **p-vrednost:** {chi['p_value']:.4f}")
                    st.write(f"- **Zaključak:** {chi['conclusion']}")

                    st.markdown("#### 2️⃣ Runs test")
                    runs = results["runs_test"]
                    st.write(f"- **Zaključak:** {runs['conclusion']}")

                    st.markdown("#### 3️⃣ Serijska korelacija")
                    serial = results["serial_correlation"]
                    st.write(f"- **Korelacija:** {serial['correlation']:.4f}")
                    st.write(f"- **Zaključak:** {serial['conclusion']}")

                    st.markdown("#### 4️⃣ Test parova")
                    pair = results.get("pair_test", {})
                    st.write(f"- **Zaključak:** {pair.get('conclusion', 'N/A')}")

                    box_class = "good-box" if overall.get("is_fair", True) else "honest-box"
                    st.markdown(f"""
                    <div class="{box_class}">
                    <strong>{overall['conclusion']}</strong>
                    <br><strong>Preporuka:</strong> {overall['recommendation']}
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Greška: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# ============================================================================
# PAGE: HISTORY
# ============================================================================

elif page == "📈 Istorija":
    st.markdown("### 📈 Istorija Portfolija")

    tracker = PredictionTracker()
    evaluated = tracker.auto_evaluate_pending()
    if evaluated > 0:
        st.info(f"Automatski evaluirano {evaluated} portfolija")

    for strat in ["coverage_optimized", "hybrid", "pure_random"]:
        perf = tracker.get_strategy_performance(strat, window=50)
        if perf and perf["n_predictions"] > 0:
            st.markdown(f"#### Strategija: `{strat}`")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Portfolija", perf["n_predictions"])
            c2.metric("Prosek najboljeg tiketa", f"{perf['avg_best_match']:.2f}/7")
            c3.metric("Stopa 3+", f"{perf['hit_rate_3plus']:.1%}")
            c4.metric("Outperform random", f"{perf.get('outperform_random_best_rate', 0):.1%}")
            c5.metric("Ukupna isplata", f"{perf['total_prize_won']:,.0f} RSD")

    st.markdown("#### Poslednji generisani portfoliji")
    session = get_session()
    try:
        recent = session.query(Prediction).order_by(Prediction.created_at.desc()).limit(20).all()

        for pred in recent:
            result_str = "⏳ Čeka evaluaciju"
            result = None
            if pred.evaluated:
                result = session.query(PredictionResult).filter_by(prediction_id=pred.prediction_id).first()
                if result:
                    icon = "🎉" if result.best_match >= 3 else "📊"
                    result_str = f"{icon} Najbolje: {result.best_match}/7 | Isplata: {result.prize_value:,.0f} RSD"

            with st.expander(
                f"#{pred.prediction_id} | {pred.target_draw_date} | {pred.strategy_name} | {result_str}"
            ):
                st.write(f"**Kreirano:** {pred.created_at}")
                st.write(f"**Model version:** {pred.model_version}")
                tickets = json.loads(pred.tickets)
                for i, ticket in enumerate(tickets, 1):
                    st.write(f"Tiket {i}: {ticket}")

                if pred.model_metadata:
                    try:
                        meta = json.loads(pred.model_metadata)
                        with st.expander("Metadata"):
                            st.json(meta)
                    except Exception:
                        st.caption("Metadata nije moguće parsirati.")

                if result:
                    actual = json.loads(result.actual_numbers)
                    payload = deserialize_result_payload(result)

                    st.write(f"**Izvučeni brojevi:** {actual}")
                    st.write(f"**Pogoci po tiketu:** {payload.get('ticket_matches', [])}")

                    empirical = payload.get("empirical_random_baseline")
                    if empirical:
                        st.write("**Empirijski random baseline:**")
                        st.write({
                            "n_random_baselines": empirical.get("n_random_baselines"),
                            "random_best_match_mean": empirical.get("random_best_match_mean"),
                            "random_prize_mean": empirical.get("random_prize_mean"),
                        })
    finally:
        session.close()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(f"""
<div class="math-box">
💡 <strong>Savet:</strong> Koristite optimizaciju pokrivanja za kvalitetniji portfolio
ili wheeling za kombinatorne garancije. Igrajte odgovorno — lutrija je zabava, ne investicija.
<br><br>
📊 Ukupan broj kombinacija u Loto 7/39: <strong>{TOTAL_COMBINATIONS:,}</strong>
</div>
""", unsafe_allow_html=True)





"""
🎟️ Izaberite Tikete za Igranje
📅 Sledeće izvlačenje: Utorak, 2026-04-14 (za 1 dan)


Tiket 1 (Coverage-Optimized) 
02 03 x y z 35 38 

Tiket 2 (Coverage-Optimized) 
03 04 x y z 33 36 

Tiket 3 (Coverage-Optimized) 
01 x 12 y 30 z 36
"""



"""
lotto_ai/gui/app.py je Streamlit aplikacija v4 za Loto 7/39 sa više „stranica“ 
(sidebar st.radio): generator portfolija, wheeling, matematika, test fer igre, istorija. 
Pre ulaska u UI traži lozinku iz st.secrets (get_secret_password / check_password). 
Pri startu poziva init_db(). U bočnoj traci prikazuje sledeće izvlačenje, broj izvlačenja u bazi, 
opciju ažuriranja podataka (scraper) i ručni unos. 
Glavni tok: generisanje portfolija tiketa, Monte Carlo procena, 
poredenje sa nasumičnim baznim portfolijima, čuvanje predikcije u bazi, 
izbor tiketa, označavanje kao odigrano, preuzimanje TXT-a. 
Posebne sekcije: wheeling sa procenom troška i verifikacijom garancije, 
tablica verovatnoća i EV, Kelly u kontekstu lutrije, opisna statistika brojeva, 
dijagnostika fer igre na istorijskim podacima, istorija sa auto-evaluacijom i pregledom metapodataka.


Modeli i tehnike

Portfolio generator: generate_adaptive_portfolio sa strategijama coverage_optimized, hybrid, pure_random; parametrizovani težinski optimizator (parovi, trojke, kazna preklapanja, parnost, zbir, broj MC kandidata) i ograničenja iz konfiguracije.

Feature matrica: build_feature_matrix nad istorijom (isti sloj kao u drugim modulima projekta).

Verovatnosna matematika: match_probability, expected_value_per_ticket, portfolio_expected_value (uz napomenu o aproksimaciji), portfolio_monte_carlo_statistics za konkretan portfolio, compare_portfolio_to_random_baseline za empirijsko poredenje sa generatorom slučajnih portfolija.

Wheeling: generate_abbreviated_wheel, wheel_cost_estimate — kombinatorne garancije uz verifikaciju.

Fer igra: test_lottery_fairness nad DataFrame-om izvlačenja (hi-kvadrat uniformnosti, runs, serijska korelacija, test parova — po strukturi rezultata u UI).

Učenje / praćenje: PredictionTracker (evaluacija, performanse strategije, čuvanje metapodataka uključujući MC i baseline), PlayedTicketsTracker, deserialize_result_payload za stari i novi format ticket_matches (lista vs dict sa empirijskim baseline-om).



Dobre strane (uočljivo u kodu)

Jasan matematički i odgovorni ton: eksplicitno se navodi da se ne predviđa budućnost, već struktura portfolija / wheeling.

Višeslojni prikaz: brza aproksimacija EV-a + Monte Carlo za generisani skup + percentili u odnosu na nasumične portfolije.

Konfigurabilnost kroz konstante (DRAW_DAYS, PRIZE_TABLE, težine optimizatora, broj simulacija).

Lokalizacija dela poruka (dani u nedelji, srpski tekst u UI gde je primenjeno).

Integracija podataka: scraper, ručni unos, cloud upozorenje za trajnost ručnog unosa.

Istorija sa SQL upitima, expanderima i parsiranjem JSON metapodataka; kompatibilnost sa starijim zapisima rezultata.



Slabije strane

Složenost session state-a: mnogo ključeva (mc_stats, baseline_stats, strategija, itd.) — veća šansa za nus-effekte pri rerun tokovima ako se proširuje UI.

Wheeling / garancija: NE daje minimalni broj kombinacija za garanciju.

Baza svih do sad izvucenih kombinacija nije imala sve kombinacije 
(sad ima trenutno svih 4596 kombinacija 30.07.1985.- 10.04.2026.).
"""
