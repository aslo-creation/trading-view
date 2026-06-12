"""
app.py — QUANT TERMINAL · Interface haute densité style Bloomberg (Streamlit).

┌──────────────────────── COMMAND LINE  (GP / WEIF / MKT / HELP <GO>) ───────┐
│ WORLD MARKETS        │  AI PREDICTIVE CONTEXT (PRED)   │  TOP NEWS (BICO)  │
│ (grille + sparklines │  TradingView Advanced Chart     │  sentiment ±10    │
│  ou matrice WEIF)    │  multi-timeframes + volume      │  RATES&MACRO (ECO)│
└─────────────────────────────────────────────────────────────────────────────┘
Backend intégralement conservé : security.py (auth/RBAC/sanitisation),
agents/* (comité IA), core/* (math + signaux), services/* (données, news,
rate limiting). Rien n'est mocké : chaque panneau est câblé aux moteurs réels.

Fix critique vs version précédente : le fragment d'auto-refresh ne déclenche
plus de rerun lors de sa PREMIÈRE exécution (cela créait une boucle infinie
de rechargements — page qui "mouline" sans jamais s'afficher).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

from agents.orchestrator import CommitteeDebrief, CommitteeOrchestrator
from config.security import (
    InvalidTickerError,
    MAX_WATCHLIST_SIZE,
    RedactingFilter,
    assert_logs_are_masked,
    authorize,
    issue_session_token,
    load_environment,
    sanitize_ticker,
    verify_session_token,
)
from core.math_engine import mean_reversion_scan, volatility_break
from core.signal_engine import composite_signal
from services.market_data import fed_funds_implied, fetch_fred, fetch_many, label_of
from services.news_feed import fetch_headlines, headlines_for_agents
from services.rate_limiter import GLOBAL_LIMITER, RateLimitExceeded

# ------------------------------------------------------------------ bootstrap
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(level=logging.INFO)
for handler in logging.getLogger().handlers:
    handler.addFilter(RedactingFilter())

load_dotenv_bridge_done = False
try:
    # En ligne (Streamlit Community Cloud), les secrets arrivent via st.secrets
    # et non via un fichier .env — on les expose comme variables d'environnement
    # AVANT load_environment() pour que tout le backend reste identique.
    for _k, _v in st.secrets.items():
        if isinstance(_v, str) and _k not in os.environ:
            os.environ[_k] = _v
except Exception:  # noqa: BLE001 — pas de secrets.toml en local : normal
    pass

load_environment()

st.set_page_config(page_title="QUANT TERMINAL", page_icon="▮", layout="wide",
                   initial_sidebar_state="collapsed")

CORE_ASSETS = ["SPX", "GOLD", "WTI", "BTC"]
GRID_ASSETS = CORE_ASSETS + ["^TNX"]          # ^TNX = rendement US 10 ans ×10
TV_SYMBOLS = {"GOLD": "TVC:GOLD", "WTI": "TVC:USOIL", "SPX": "SP:SPX",
              "BTC": "BINANCE:BTCUSDT", "^TNX": "TVC:US10Y"}
AGENT_AVATARS = {"Macro Economist": "🌍", "Quantitative Strategist": "📐",
                 "Structurer / Risk Manager": "🛡️"}

TERMINAL_CSS = """
<style>
.stApp { background:#080808; }
section[data-testid="stSidebar"] { background:#0a0a0a; border-right:1px solid #222; }
div[data-testid="stVerticalBlock"] * { border-radius:0 !important; }
.stTabs [data-baseweb="tab-list"] { border-bottom:1px solid #222; }
.term-hdr { color:#f5a623; font:700 11px monospace; letter-spacing:2px;
            border-bottom:1px solid #222; padding:3px 0; margin:2px 0 6px 0; }
.term-box { border:1px solid #222; background:#0d0d0d; padding:8px 10px;
            font-family:monospace; }
.pos { color:#00ff00; } .neg { color:#ff3333; } .amber { color:#f5a623; }
.dim { color:#777; font-size:11px; }
table.wm { width:100%; border-collapse:collapse; font:12px monospace; color:#ccc; }
table.wm td { border-bottom:1px solid #181818; padding:3px 5px; white-space:nowrap; }
.newsline { font:12px monospace; color:#ccc; border-bottom:1px solid #161616;
            padding:4px 0; }
.sbadge { display:inline-block; min-width:30px; text-align:center;
          font:700 11px monospace; padding:0 3px; border:1px solid #222; }
@keyframes flashg { 0%,100%{opacity:1;} 50%{opacity:.35;} }
@keyframes flashr { 0%,100%{opacity:1;} 50%{opacity:.35;} }
.flash-pos { animation:flashg 1.1s infinite; color:#00ff00; border-color:#00ff00; }
.flash-neg { animation:flashr 1.1s infinite; color:#ff3333; border-color:#ff3333; }
.predscore { font:700 26px monospace; }
.cmdhint { color:#555; font:11px monospace; }
</style>
"""
st.markdown(TERMINAL_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------ auth wall
with open("config/auth_config.yaml", encoding="utf-8") as f:
    auth_cfg = yaml.load(f, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    auth_cfg["credentials"], auth_cfg["cookie"]["name"],
    auth_cfg["cookie"]["key"], auth_cfg["cookie"]["expiry_days"],
)
authenticator.login(location="main")

if st.session_state.get("authentication_status") is False:
    try:
        GLOBAL_LIMITER.acquire("auth_attempts",
                               identity=st.session_state.get("username") or "anon")
    except RateLimitExceeded:
        st.error("Trop de tentatives. Verrouillage temporaire.")
        st.stop()
    st.error("Identifiants invalides.")
    st.stop()
elif st.session_state.get("authentication_status") is None:
    st.info("Connectez-vous pour accéder au terminal.")
    st.stop()

username = st.session_state["username"]
role = auth_cfg["credentials"]["usernames"][username].get("role", "viewer")
if "session_token" not in st.session_state:
    st.session_state["session_token"] = issue_session_token(username, role)
claims = verify_session_token(st.session_state["session_token"])
if claims is None:
    st.session_state.clear()
    st.error("Session expirée. Reconnectez-vous.")
    st.stop()

# ------------------------------------------------------------------ sidebar
authenticator.logout(location="sidebar")
st.sidebar.success(f"{claims.username} · {claims.role}")
BEGINNER = st.sidebar.radio("Niveau", ["🎓 Débutant", "⚡ Avancé"],
                            horizontal=True).startswith("🎓")

if "watchlist" not in st.session_state:
    st.session_state.watchlist = list(CORE_ASSETS)
if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = "SPX"
if "left_view" not in st.session_state:
    st.session_state.left_view = "MKT"
if "cmd_msg" not in st.session_state:
    st.session_state.cmd_msg = ""

st.sidebar.divider()
new_ticker = st.sidebar.text_input("Ajouter à la watchlist",
                                   placeholder="AAPL, NVDA, ETH-USD…",
                                   max_chars=15)
if st.sidebar.button("➕ Ajouter") and new_ticker:
    if not authorize(claims, "add_ticker"):
        st.sidebar.error("Permissions insuffisantes.")
    else:
        try:
            GLOBAL_LIMITER.acquire("ticker_mutation", identity=claims.username)
            clean = sanitize_ticker(new_ticker)
            if len(st.session_state.watchlist) >= MAX_WATCHLIST_SIZE:
                st.sidebar.error("Watchlist pleine.")
            elif clean not in st.session_state.watchlist:
                st.session_state.watchlist.append(clean)
                st.sidebar.success(f"{clean} ajouté.")
        except (InvalidTickerError, RateLimitExceeded) as exc:
            st.sidebar.error(str(exc))

removable = [s for s in st.session_state.watchlist if s not in CORE_ASSETS]
if removable:
    to_remove = st.sidebar.selectbox("Retirer", ["—"] + removable)
    if to_remove != "—" and st.sidebar.button("🗑 Retirer"):
        st.session_state.watchlist.remove(to_remove)
        if st.session_state.selected_symbol == to_remove:
            st.session_state.selected_symbol = "SPX"
        st.rerun()

if st.sidebar.button("🔄 Rafraîchir maintenant"):
    st.cache_data.clear()
    st.rerun()
auto_refresh = st.sidebar.toggle("🔁 Actualisation auto", value=True)
refresh_secs = st.sidebar.select_slider(
    "Intervalle", options=[120, 300, 600], value=300,
    format_func=lambda s: f"{s // 60} min") if auto_refresh else 300
st.sidebar.caption("Outil d'analyse — pas un conseil en investissement.")

# ------------------------------------------------------------------ data layer
@st.cache_data(ttl=120, show_spinner="📡 Données de marché…")
def get_market_data(symbols: tuple[str, ...]):
    out, demo = fetch_many(list(symbols))
    return out, demo, datetime.now().strftime("%H:%M:%S")

@st.cache_data(ttl=600, show_spinner="📰 Actualités…")
def get_news():
    return fetch_headlines()

@st.cache_data(ttl=3600, show_spinner=False)
def get_fred():
    return fetch_fred()

@st.cache_data(ttl=1800, show_spinner=False)
def get_fed_implied():
    return fed_funds_implied()

all_symbols = tuple(dict.fromkeys(st.session_state.watchlist + GRID_ASSETS))
data, demo_symbols, fetched_at = get_market_data(all_symbols)
headlines = get_news()
fred, fred_live = get_fred()

st.sidebar.caption(f"📡 MàJ {fetched_at}"
                   + (f" · auto/{refresh_secs // 60} min" if auto_refresh
                      else " · auto OFF"))

# --- Auto-refresh SANS boucle infinie : jamais de rerun au premier passage ---
if auto_refresh and hasattr(st, "fragment"):
    @st.fragment(run_every=timedelta(seconds=refresh_secs))
    def _auto_tick():
        now = time.time()
        last = st.session_state.get("_tick_ts")
        st.session_state["_tick_ts"] = now
        if last is not None and (now - last) >= refresh_secs - 5:
            st.rerun(scope="app")
    _auto_tick()

if demo_symbols:
    st.warning(f"⚠️ DÉMO pour : {', '.join(sorted(demo_symbols))} "
               "(hors-ligne ou symbole introuvable).")

# ------------------------------------------------------------------ helpers UI
def fmt_px(v: float) -> str:
    return f"{v:,.2f}" if abs(v) < 100000 else f"{v:,.0f}"

def sparkline_svg(series: pd.Series, w: int = 110, h: int = 24) -> str:
    s = series.dropna().tail(40)
    if len(s) < 2:
        return ""
    vals = s.to_numpy(dtype=float)
    mn, mx = float(vals.min()), float(vals.max())
    rng = (mx - mn) or 1.0
    pts = " ".join(
        f"{i * w / (len(vals) - 1):.1f},{h - 2 - (v - mn) / rng * (h - 4):.1f}"
        for i, v in enumerate(vals))
    color = "#00ff00" if vals[-1] >= vals[0] else "#ff3333"
    return (f'<svg width="{w}" height="{h}"><polyline fill="none" '
            f'stroke="{color}" stroke-width="1.3" points="{pts}"/></svg>')

def world_markets_html() -> str:
    rows = []
    for sym in GRID_ASSETS + [s for s in st.session_state.watchlist
                              if s not in GRID_ASSETS]:
        df = data.get(sym)
        if df is None or len(df) < 2:
            continue
        last = float(df["close"].iloc[-1])
        chg = (last / float(df["close"].iloc[-2]) - 1) * 100
        name = "US 10Y" if sym == "^TNX" else label_of(sym)
        shown = last / 10 if sym == "^TNX" else last     # ^TNX = yield × 10
        suffix = " %" if sym == "^TNX" else ""
        cls = "pos" if chg >= 0 else "neg"
        try:
            z = mean_reversion_scan(sym, df).z20
            ztxt = f"<span class='{ 'amber' if abs(z) >= 2.5 else 'dim'}'>Z {z:+.1f}</span>"
        except (ValueError, IndexError, KeyError):
            ztxt = "<span class='dim'>—</span>"
        rows.append(
            f"<tr><td class='amber'>{name}</td>"
            f"<td style='text-align:right'>{fmt_px(shown)}{suffix}</td>"
            f"<td class='{cls}' style='text-align:right'>{chg:+.2f}%</td>"
            f"<td>{sparkline_svg(df['close'])}</td><td>{ztxt}</td></tr>")
    return "<table class='wm'>" + "".join(rows) + "</table>"

def weif_heatmap():
    syms = [s for s in st.session_state.watchlist if s in data][:10]
    closes = pd.DataFrame({s: data[s]["close"] for s in syms}).dropna()
    corr = closes.pct_change().dropna().tail(60).corr()
    import plotly.graph_objects as go
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index),
        zmin=-1, zmax=1, text=np.round(corr.values, 2),
        texttemplate="%{text}",
        colorscale=[[0, "#ff3333"], [0.5, "#101010"], [1, "#00ff00"]],
        showscale=False))
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=0, b=0),
                      paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
                      font=dict(family="monospace", size=11, color="#ccc"))
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False})

def tradingview_widget(symbol: str, height: int = 470):
    tv = TV_SYMBOLS.get(symbol, symbol.replace("=F", "").replace("^", ""))
    html = f"""
    <div class="tradingview-widget-container" style="height:{height}px">
      <div id="tvchart" style="height:{height}px"></div>
      <script src="https://s3.tradingview.com/tv.js"></script>
      <script>
      new TradingView.widget({{
        "container_id": "tvchart", "symbol": "{tv}", "interval": "D",
        "autosize": true, "theme": "dark", "style": "1", "locale": "fr",
        "toolbar_bg": "#080808", "withdateranges": true,
        "hide_side_toolbar": false, "allow_symbol_change": false,
        "studies": ["Volume@tv-basicstudies"]
      }});
      </script>
    </div>"""
    components.html(html, height=height + 6)

def pred_panel(symbol: str):
    sig = composite_signal(symbol, data[symbol])
    prob = float(np.clip(50 + sig.score / 2, 5, 95))
    if sig.score >= 15:
        line, cls = f"{prob:.0f}% Prob. LONG Setup", "pos"
    elif sig.score <= -15:
        line, cls = f"{100 - prob:.0f}% Prob. SHORT Setup", "neg"
    else:
        line, cls = "FLAT — pas d'avantage statistique", "amber"
    reasons = "".join(f"<div class='dim'>· {r}</div>" for r in sig.reasons[:4])
    st.markdown(
        f"<div class='term-box'><div class='term-hdr'>AI PREDICTIVE CONTEXT "
        f"· {label_of(symbol)}</div>"
        f"<span class='predscore {cls}'>{line}</span>"
        f"<span class='dim'>  score {sig.score:+.0f}/100</span>{reasons}"
        f"<div class='dim' style='margin-top:4px'>Heuristique explicable "
        f"(retour moyenne 40 / tendance 35 / RSI 25) — pas une certitude. "
        f"Croisez avec le Comité IA.</div></div>",
        unsafe_allow_html=True)

def news_panel():
    st.markdown("<div class='term-hdr'>TOP NEWS · BICO ‹GO›</div>",
                unsafe_allow_html=True)
    if not headlines:
        st.markdown("<span class='dim'>Aucun flux — vérifiez la connexion."
                    "</span>", unsafe_allow_html=True)
        return
    lines = []
    for h in headlines[:14]:
        s = h.sentiment
        if s >= 6:
            badge = f"<span class='sbadge flash-pos'>+{s}</span>"
        elif s <= -6:
            badge = f"<span class='sbadge flash-neg'>{s}</span>"
        elif s > 0:
            badge = f"<span class='sbadge pos'>+{s}</span>"
        elif s < 0:
            badge = f"<span class='sbadge neg'>{s}</span>"
        else:
            badge = "<span class='sbadge dim'>0</span>"
        fire = "🔥" if h.impact == "haute" else ""
        title = (f"<a href='{h.link}' style='color:#ccc;text-decoration:none'>"
                 f"{h.title}</a>" if h.link else h.title)
        lines.append(f"<div class='newsline'>{badge} {fire} {title} "
                     f"<span class='dim'>· {h.source} · {h.age_label}</span></div>")
    st.markdown("".join(lines), unsafe_allow_html=True)
    if BEGINNER:
        st.markdown("<span class='dim'>Score −10/+10 = tonalité automatique du "
                    "titre (lexique) ; clignote aux extrêmes. 🔥 = fort impact."
                    "</span>", unsafe_allow_html=True)

def macro_panel():
    st.markdown("<div class='term-hdr'>RATES & MACRO · ECO ‹GO›"
                + ("" if fred_live else " <span class='dim'>(démo)</span>")
                + "</div>", unsafe_allow_html=True)
    def last(sid):
        s = fred.get(sid)
        return float(s.iloc[-1]) if s is not None and len(s) else None
    def delta(sid):
        s = fred.get(sid)
        return float(s.iloc[-1] - s.iloc[-2]) if s is not None and len(s) > 1 else None
    rows = []
    def add(name, val, d=None, fmt="{:.2f}", unit=" %"):
        if val is None:
            rows.append(f"<tr><td class='amber'>{name}</td>"
                        f"<td class='dim' colspan=2>n/d</td></tr>")
            return
        dtxt = ""
        if d is not None:
            cls = "pos" if d >= 0 else "neg"
            dtxt = f"<td class='{cls}' style='text-align:right'>{d:+.2f}</td>"
        rows.append(f"<tr><td class='amber'>{name}</td>"
                    f"<td style='text-align:right'>{fmt.format(val)}{unit}</td>"
                    f"{dtxt}</tr>")
    add("US 10Y", last("DGS10"), delta("DGS10"))
    add("US 2Y", last("DGS2"), delta("DGS2"))
    y10, y2 = last("DGS10"), last("DGS2")
    add("Spread 10Y−2Y", (y10 - y2) if y10 is not None and y2 is not None else None)
    cpi = fred.get("CPIAUCSL")
    cpi_yoy = (float(cpi.iloc[-1] / cpi.iloc[-13] - 1) * 100
               if cpi is not None and len(cpi) >= 13 else None)
    add("Inflation CPI (1 an)", cpi_yoy)
    eff = last("DFF")
    add("Fed Funds effectif", eff)
    imp = get_fed_implied()
    add("Fed implicite (ZQ fut.)", imp)
    if imp is not None and eff is not None:
        move = imp - eff
        direction = "BAISSE" if move < 0 else "HAUSSE"
        probish = min(abs(move) / 0.25 * 100, 100)
        add(f"Pricé : {direction}", probish, fmt="{:.0f}", unit=" %*")
    st.markdown("<table class='wm'>" + "".join(rows) + "</table>"
                "<span class='dim'>* estimation simplifiée via futures ZQ — "
                "indicative seulement.</span>", unsafe_allow_html=True)

# ------------------------------------------------------------------ command router
def handle_command(raw: str):
    c = raw.strip().upper().replace("<GO>", " ").replace("‹GO›", " ").strip()
    parts = [p for p in c.split() if p and p != "GO"]
    if not parts:
        return
    cmd = parts[0]
    if cmd == "HELP":
        st.session_state.cmd_msg = ("Commandes : [ACTIF] GP = graphique "
                                    "(ex: BTC GP) · WEIF = matrice de "
                                    "corrélations · MKT = world markets · "
                                    "HELP = cette aide")
    elif cmd == "WEIF":
        st.session_state.left_view = "WEIF"
        st.session_state.cmd_msg = "WEIF › matrice de corrélations affichée."
    elif cmd == "MKT":
        st.session_state.left_view = "MKT"
        st.session_state.cmd_msg = "MKT › world markets affiché."
    else:
        try:
            sym = sanitize_ticker(cmd)
        except InvalidTickerError:
            st.session_state.cmd_msg = f"Symbole invalide : {cmd[:15]}"
            return
        if sym not in st.session_state.watchlist and sym not in GRID_ASSETS:
            if not authorize(claims, "add_ticker"):
                st.session_state.cmd_msg = "Permissions insuffisantes."
                return
            if len(st.session_state.watchlist) >= MAX_WATCHLIST_SIZE:
                st.session_state.cmd_msg = "Watchlist pleine."
                return
            st.session_state.watchlist.append(sym)
        st.session_state.selected_symbol = sym
        st.session_state.cmd_msg = f"{sym} GP › graphique chargé."

def _on_command():
    handle_command(st.session_state.get("cmdline", ""))
    st.session_state.cmdline = ""

# ================================================================== TABS
tab_term, tab_ai, tab_admin = st.tabs(["▮ TERMINAL", "🏛 COMITÉ IA", "🔐 ADMIN"])

# ================================================================== TERMINAL
with tab_term:
    c_cmd, c_hint = st.columns([2, 3])
    with c_cmd:
        st.text_input("CMD", key="cmdline", on_change=_on_command,
                      placeholder="BTC GP <GO>   ·   WEIF <GO>   ·   HELP <GO>",
                      label_visibility="collapsed")
    with c_hint:
        msg = st.session_state.cmd_msg or \
            "Tapez une commande puis Entrée — ex : GOLD GP <GO>"
        st.markdown(f"<div class='cmdhint' style='padding-top:8px'>{msg}</div>",
                    unsafe_allow_html=True)

    col_left, col_mid, col_right = st.columns([1.15, 2.3, 1.25])

    with col_left:
        if st.session_state.left_view == "WEIF":
            st.markdown("<div class='term-hdr'>WEIF ‹GO› · CORRÉLATIONS 60J"
                        "</div>", unsafe_allow_html=True)
            weif_heatmap()
            st.markdown("<span class='dim'>Vert +1 = évoluent ensemble · "
                        "Rouge −1 = opposés. Tapez MKT ‹GO› pour revenir."
                        "</span>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='term-hdr'>WORLD MARKETS ‹GO›</div>",
                        unsafe_allow_html=True)
            st.markdown(world_markets_html(), unsafe_allow_html=True)
            st.markdown("<span class='dim'>Z = écart statistique 20j ; ambre "
                        "si |Z| ≥ 2,5 (extrême).</span>", unsafe_allow_html=True)
        if not BEGINNER:
            with st.expander("Scanner détaillé"):
                rows = []
                for sym, df in data.items():
                    try:
                        mr = mean_reversion_scan(sym, df)
                        vb = volatility_break(sym, df["close"])
                        rows.append({"Sym": sym, "Z20": round(mr.z20, 2),
                                     "Z50": round(mr.z50, 2),
                                     "Vol10/60": round(vb.ratio, 2),
                                     "Src": "DÉMO" if sym in demo_symbols
                                            else "LIVE"})
                    except (ValueError, IndexError, KeyError):
                        continue
                st.dataframe(pd.DataFrame(rows), hide_index=True,
                             use_container_width=True)

    with col_mid:
        sel = st.session_state.selected_symbol
        if sel not in data:
            sel = "SPX"
        pred_panel(sel)
        tradingview_widget(sel)
        st.markdown("<span class='dim'>Graphique TradingView temps réel — "
                    "timeframes et indicateurs via sa barre d'outils. Le "
                    "panneau PRED ci-dessus est calculé par NOTRE moteur "
                    "quantitatif sur les données Yahoo.</span>",
                    unsafe_allow_html=True)

    with col_right:
        news_panel()
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        macro_panel()

# ================================================================== COMITÉ IA
with tab_ai:
    st.header("Comité d'investissement IA")
    st.caption("🌍 Macro + 📐 Quant en parallèle ; 🛡️ Risk challenge, "
               "dimensionne (Kelly/VaR) ou veto. Actualités et séries macro "
               "réelles injectées.")
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.warning("Ajoutez ANTHROPIC_API_KEY dans .env pour activer le débat "
                   "IA (sinon : abstention propre).")
    if not authorize(claims, "run_committee"):
        st.warning("Votre rôle ne permet pas de convoquer le comité.")
    else:
        if st.button("🏛 Convoquer le comité", type="primary"):
            try:
                GLOBAL_LIMITER.acquire("llm_committee", identity=claims.username)
            except RateLimitExceeded as exc:
                st.error(str(exc))
                st.stop()
            context = {
                "ohlcv": {s: data[s] for s in CORE_ASSETS if s in data},
                "fred": fred,
                "headlines": headlines_for_agents(headlines),
            }
            async def _run():
                orch = CommitteeOrchestrator()
                try:
                    return await orch.convene(context)
                finally:
                    await orch.aclose()
            with st.spinner("Comité en session…"):
                st.session_state["last_debrief"] = asyncio.run(_run())

        debrief: CommitteeDebrief | None = st.session_state.get("last_debrief")
        if debrief is None:
            st.info("Cliquez sur **Convoquer le comité**.")
        else:
            st.caption(f"Session {debrief.started_at} · "
                       f"Macro {'FRED live' if fred_live else 'démo'} · "
                       f"{len(headlines)} titres injectés")
            for rep in debrief.transcript + (
                    [r for r in debrief.rebuttals] if debrief.rebuttals else []):
                with st.chat_message(rep.agent,
                                     avatar=AGENT_AVATARS.get(rep.agent, "🤖")):
                    st.markdown(f"**{rep.agent}** · confiance {rep.confidence:.0%}")
                    st.markdown(f"*{rep.stance}*")
                    for kp in rep.key_points:
                        st.markdown(f"- {kp}")
            st.divider()
            verdict = debrief.final_verdict
            if verdict.get("verdict") == "sized_trade":
                d = verdict["detail"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Signal", f"{d['side'].upper()} {d['symbol']}")
                c2.metric("Entrée réf.", f"{d['invalidation']['entry_ref']:,.2f}")
                c3.metric("Stop", f"{d['invalidation']['stop']:,.2f}",
                          f"-{d['invalidation']['stop_distance_pct']} %")
                c4.metric("Taille max", f"{d['sizing']['max_position_pct']:.2%}")
                st.success(f"Comité : {verdict['stance']} "
                           f"({verdict['confidence']:.0%})")
            else:
                st.info("VERDICT : NO TRADE / ABSTENTION.")
            st.caption(verdict.get("disclaimer", ""))

# ================================================================== ADMIN
with tab_admin:
    st.header("Posture de sécurité")
    if authorize(claims, "view_secrets_report"):
        report = assert_logs_are_masked()
        if report:
            st.table(pd.DataFrame([{"clé": k, "valeur masquée": v}
                                   for k, v in report.items()]))
        else:
            st.info("Aucune clé optionnelle configurée.")
        st.caption("Limites : " + " | ".join(
            f"{sc} {c}/{int(w)}s"
            for sc, (c, w) in GLOBAL_LIMITER.DEFAULT_POLICIES.items()))
    else:
        st.warning("Rôle admin requis.")
