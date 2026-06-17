"""
NG Natural Gas — IBS + 20-Day Extreme Live Signal Dashboard
  • Yahoo Finance NG=F  (5-min auto-refresh)
  • Entry:  IBS < 0.20  AND  close ≤ 20-day rolling low   (long)
            IBS > 0.80  AND  close ≥ 20-day rolling high  (short)
  • Exit:   IBS > 0.50  (long)  /  IBS < 0.50  (short)
  • Exact IBS and 20-day extreme trigger prices for today, live intraday H/L
  • Manual position tracker with $P&L
  • Roll-adjusted prices: NG futures contract rolls are back-adjusted for continuity
"""
import warnings; warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NG Signal · IBS+20d",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── THEME ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0F172A !important; }
section[data-testid="stSidebar"] { background-color: #1E293B !important; }

[data-testid="metric-container"] {
    background: #1E293B; border: 1px solid #334155;
    border-radius: 8px; padding: 12px 16px;
}
[data-testid="stMetricLabel"] p  { color: #94A3B8 !important; font-size: 11px !important; }
[data-testid="stMetricValue"]    { color: #E2E8F0 !important; }
[data-testid="stMetricDelta"]    { font-size: 11px !important; }

.signal-wrap {
    background:#1E293B; border:1px solid #334155;
    border-radius:8px; padding:18px 20px; text-align:center;
}
.sig-badge {
    display:inline-block; padding:8px 28px; border-radius:6px;
    font-size:28px; font-weight:800; letter-spacing:3px; margin:6px 0;
}
.sig-long  { background:#064e3b; color:#10B981; border:2px solid #10B981; }
.sig-short { background:#450a0a; color:#EF4444; border:2px solid #EF4444; }
.sig-flat  { background:#1E293B; color:#94A3B8; border:2px solid #475569; }
.sig-sub   { color:#94A3B8; font-size:12px; margin-top:6px; }

.tc { background:#1E293B; border:1px solid #334155; border-radius:8px; padding:14px 16px; }
.tc-warn { border-color:#F59E0B !important; background:#1c1500 !important; }
.tc-info { border-color:#38BDF8 !important; background:#0c1a26 !important; }
.tc-lbl  { color:#64748B; font-size:10px; text-transform:uppercase; letter-spacing:1.2px; }
.tc-px   { font-size:22px; font-weight:700; font-family:monospace; color:#E2E8F0; margin:4px 0; }
.tc-note { color:#64748B; font-size:11px; margin-top:3px; }
.g { color:#10B981; } .r { color:#EF4444; } .y { color:#F59E0B; } .b { color:#38BDF8; }

#MainMenu,footer { visibility:hidden; }
.block-container { padding-top:0.6rem; }
hr { border-color:#334155 !important; }
</style>
""", unsafe_allow_html=True)

# 5-minute JS auto-refresh
st.components.v1.html('<script>setTimeout(()=>{window.location.reload();},300000);</script>', height=0)

# ─── ROLL CALENDAR ────────────────────────────────────────────────────────────
def _ng_roll_schedule(start_ts, end_ts):
    """
    Compute Yahoo Finance NG=F roll dates.
    NG futures expire on the 3rd-to-last business day of the delivery month.
    Yahoo switches to the new front-month contract the next trading day.
    """
    roll_dates = []
    current = pd.Timestamp(start_ts).replace(day=1)
    end     = pd.Timestamp(end_ts) + pd.DateOffset(months=2)
    while current <= end:
        m_end  = current + pd.offsets.MonthEnd(0)
        bdays  = pd.bdate_range(current, m_end)
        if len(bdays) >= 3:
            expiry = bdays[-3]
            nxt    = pd.bdate_range(expiry + pd.Timedelta(days=1),
                                    expiry + pd.Timedelta(days=10))
            if len(nxt):
                roll_dates.append(nxt[0].normalize())
        current += pd.DateOffset(months=1)
    return roll_dates


def detect_and_adjust(df):
    """
    Find NG=F contract-roll gaps on the computed expiry schedule and
    back-adjust all prices so returns are continuous across rolls.

    Only adjusts when |gap| > 2 % (filters out tiny basis noise).
    Back-adjustment preserves CURRENT prices; all PRIOR prices are scaled
    so percentage returns within each period are unchanged.

    Returns (adjusted_df, roll_info_list).
    """
    df = df.copy()
    schedule = _ng_roll_schedule(df.index[0], df.index[-1])

    confirmed = []
    for rd in schedule:
        target = None
        for candidate in [rd, rd + pd.tseries.offsets.BusinessDay(1)]:
            if candidate in df.index:
                target = candidate
                break
        if target is None:
            continue
        iloc = df.index.get_loc(target)
        if iloc < 1:
            continue
        ratio = float(df["close"].iloc[iloc] / df["close"].iloc[iloc - 1])
        if abs(ratio - 1.0) < 0.02:
            continue
        confirmed.append((iloc, ratio, target))

    roll_info = [
        {
            "date":      target,
            "gap_pct":   (ratio - 1) * 100,
            "ratio":     ratio,
            "old_close": float(df["close"].iloc[iloc - 1]),
            "new_close": float(df["close"].iloc[iloc]),
        }
        for iloc, ratio, target in confirmed
    ]

    cols = ["open", "high", "low", "close"]
    for iloc, ratio, _ in sorted(confirmed, reverse=True):
        for col in cols:
            df.iloc[:iloc, df.columns.get_loc(col)] *= ratio

    return df, roll_info


# ─── DATA ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_daily():
    raw = yf.download("NG=F", period="4y", auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame(), []
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]
    raw = raw[["open", "high", "low", "close"]].dropna().copy()
    raw.index = pd.to_datetime(raw.index).tz_localize(None)

    df, roll_info = detect_and_adjust(raw)

    df["gap"] = df.index.to_series().diff().dt.days
    df = df[(df["high"] > df["low"]) & (df["gap"] <= 14)].copy()

    # IBS
    df["IBS"] = (df["close"] - df["low"]) / (df["high"] - df["low"])

    # 20-day rolling extreme (entry thresholds)
    df["rmin20"] = df["close"].rolling(20).min()
    df["rmax20"] = df["close"].rolling(20).max()

    return df.dropna(subset=["IBS", "rmin20", "rmax20"]), roll_info


@st.cache_data(ttl=60)
def get_intraday():
    try:
        df5 = yf.download("NG=F", period="1d", interval="5m",
                          auto_adjust=True, progress=False)
        if df5.empty:
            return None
        if isinstance(df5.columns, pd.MultiIndex):
            df5.columns = df5.columns.get_level_values(0)
        df5.columns = [c.lower() for c in df5.columns]
        df5.index = pd.to_datetime(df5.index).tz_localize(None)
        return dict(price=float(df5["close"].iloc[-1]),
                    high=float(df5["high"].max()),
                    low=float(df5["low"].min()),
                    time=df5.index[-1])
    except Exception:
        return None


# ─── SIGNAL ENGINE ────────────────────────────────────────────────────────────
def run_engine(df):
    """
    Flat position model:
      Enter long  when IBS < 0.20  AND  close <= 20d rolling low
      Enter short when IBS > 0.80  AND  close >= 20d rolling high
      Exit long   when IBS > 0.50
      Exit short  when IBS < 0.50
    One trade record per completed entry→exit.
    """
    cv  = df["close"].values
    ib  = df["IBS"].values
    rmn = df["rmin20"].values
    rmx = df["rmax20"].values
    dt  = df.index
    n   = len(df)

    long_mask  = (ib < 0.20) & (cv <= rmn)
    short_mask = (ib > 0.80) & (cv >= rmx)

    trades = []
    eq     = np.ones(n)
    val    = 1.0
    pos    = 0      # 0=flat, 1=long, -1=short
    ei     = None   # entry bar index
    epx    = None   # entry price

    for i in range(n):
        x = cv[i]

        if pos == 0:
            eq[i] = val
            if long_mask[i]:
                pos = 1;  ei = i; epx = x
            elif short_mask[i]:
                pos = -1; ei = i; epx = x
            continue

        ur   = (x / epx - 1) * pos
        eq[i] = val * (1 + ur)

        if i == ei:
            continue

        exit_now = (pos == 1 and ib[i] > 0.50) or (pos == -1 and ib[i] < 0.50)

        if exit_now:
            ret = (x / epx - 1) * pos
            trades.append(dict(
                side="long" if pos == 1 else "short",
                entry_date=dt[ei],
                exit_date=dt[i],
                entry_px=epx,
                exit_px=x,
                return_pct=ret * 100,
                hold=i - ei,
            ))
            val    = val * (1 + ret)
            eq[i]  = val
            pos    = 0; ei = None; epx = None

    return trades, pd.Series(eq, index=dt), dict(
        d=pos,
        epx=epx,
        entry_date=dt[ei] if ei is not None else None,
        val=val,
        last_c=float(cv[-1]),
        last_ib=float(ib[-1]),
        last_dt=dt[-1],
        rmin20=float(rmn[-1]),
        rmax20=float(rmx[-1]),
    )


# ─── TRIGGER PRICES ───────────────────────────────────────────────────────────
def calc_triggers(df, live):
    """
    Exact closing prices that would fire each signal today.

    Long entry requires BOTH:
      IBS < 0.20  →  close ≤ low + 0.20 × range
      close ≤ 20d rolling low  (from prior closes)
    Binding = min of the two (more restrictive ceiling).

    Short entry requires BOTH:
      IBS > 0.80  →  close ≥ low + 0.80 × range
      close ≥ 20d rolling high (from prior closes)
    Binding = max of the two (more restrictive floor).
    """
    h   = live["high"] if live else float(df["high"].iloc[-1])
    l   = live["low"]  if live else float(df["low"].iloc[-1])
    rng = max(h - l, 1e-9)

    ibs_long  = l + 0.20 * rng
    ibs_exit  = l + 0.50 * rng
    ibs_short = l + 0.80 * rng

    rmin20 = float(df["rmin20"].iloc[-1])
    rmax20 = float(df["rmax20"].iloc[-1])

    long_trigger  = min(ibs_long,  rmin20)
    long_bind     = "IBS" if ibs_long <= rmin20 else "20d Low"

    short_trigger = max(ibs_short, rmax20)
    short_bind    = "IBS" if ibs_short >= rmax20 else "20d High"

    return dict(
        h=h, l=l, rng=rng,
        ibs_long=ibs_long, ibs_exit=ibs_exit, ibs_short=ibs_short,
        rmin20=rmin20, rmax20=rmax20,
        long_trigger=long_trigger,   long_bind=long_bind,
        short_trigger=short_trigger, short_bind=short_bind,
    )


# ─── HTML HELPERS ─────────────────────────────────────────────────────────────
def fp(p):
    return f"${p:.3f}" if p is not None else "—"

def delta_span(price, ref, invert=False):
    pct = (price / ref - 1) * 100
    if invert: pct = -pct
    col = "#10B981" if pct >= 0 else "#EF4444"
    arr = "▲" if pct >= 0 else "▼"
    return (f'<span style="color:{col}">{arr}&nbsp;{abs(pct):.2f}%'
            f'&nbsp;&nbsp;(${abs(price-ref):.3f})</span>')

def tcard(label, price, ref, note="", extra_cls=""):
    """Render a trigger/status card as HTML."""
    if price is None:
        body  = '<div class="tc-px">—</div>'
        delta = ""
    else:
        near    = ref is not None and abs(price - ref) / max(ref, 1e-9) < 0.003
        hit_tag = (' <span style="color:#F59E0B;font-weight:700">✓&nbsp;HIT</span>'
                   if near else "")
        body  = f'<div class="tc-px">{fp(price)}{hit_tag}</div>'
        delta = delta_span(price, ref) if ref is not None else ""
    return (f'<div class="tc {extra_cls}">'
            f'<div class="tc-lbl">{label}</div>'
            f'{body}'
            f'<div style="margin:4px 0">{delta}</div>'
            f'<div class="tc-note">{note}</div>'
            f'</div>')

def status_card(label, value, note="", extra_cls=""):
    """Render a plain status card (no delta arrow) as HTML."""
    return (f'<div class="tc {extra_cls}">'
            f'<div class="tc-lbl">{label}</div>'
            f'<div class="tc-px" style="font-size:18px">{value}</div>'
            f'<div class="tc-note">{note}</div>'
            f'</div>')


# ─── LOAD ─────────────────────────────────────────────────────────────────────
df, roll_info = get_daily()
if df.empty:
    st.error("⚠️ Yahoo Finance returned no data for NG=F. Check your connection.")
    st.stop()

live              = get_intraday()
trades, eq_s, st8 = run_engine(df)
trg               = calc_triggers(df, live)

now_px  = live["price"] if live else st8["last_c"]
last_c  = st8["last_c"]; last_ib = st8["last_ib"]
d_cur   = st8["d"]


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📍  My Position")
    has_pos  = st.toggle("Track open position", key="has_pos")
    my_side  = None; my_entry = None; my_qty = 1
    if has_pos:
        my_side  = st.selectbox("Side", ["Long", "Short"])
        my_entry = st.number_input("Entry Price ($)", min_value=0.001,
                                   step=0.001, format="%.3f")
        my_qty   = st.number_input("Contracts", min_value=1, value=1)
        st.caption("NG futures: **$10,000 / point / contract**")

    st.divider()
    st.markdown("### 📋  Signal Rules")
    st.markdown("""
🟢 **Long entry**: IBS < 0.20  AND  close = 20-day low  
🔴 **Short entry**: IBS > 0.80  AND  close = 20-day high  
⚡ **IBS exit**: IBS > 0.50 (long) / IBS < 0.50 (short) → **close full position**  
Both entry conditions must be met on the same close.
""")

    st.divider()
    col_r, col_t = st.columns(2)
    with col_r:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear(); st.rerun()
    with col_t:
        st.caption(f"Updated:\n{datetime.now().strftime('%H:%M:%S')}")

    # Strategy stats 2012+
    st.divider()
    st.markdown("### 📈  Backtest (2012+)")
    eq12 = eq_s[eq_s.index >= "2012"]
    if len(eq12) > 2:
        yrs  = (eq12.index[-1] - eq12.index[0]).days / 365.25
        cagr = (eq12.iloc[-1] ** (1 / yrs) - 1) * 100
        dd   = (eq12 - eq12.cummax()) / eq12.cummax() * 100
        vol  = eq12.pct_change().dropna().std() * np.sqrt(252) * 100
        c1, c2 = st.columns(2)
        with c1:
            st.metric("CAGR",   f"{cagr:.1f}%")
            st.metric("Max DD", f"{dd.min():.1f}%")
        with c2:
            st.metric("Sharpe", f"{cagr/vol:.3f}")
            st.metric("Final",  f"{eq12.iloc[-1]:.1f}×")

    if roll_info:
        st.divider()
        st.markdown("### 🔄  Recent Roll Adjustments")
        st.caption("Prices back-adjusted at each roll to preserve return continuity.")
        for r in sorted(roll_info, key=lambda x: x["date"], reverse=True)[:6]:
            col = "#EF4444" if r["gap_pct"] < 0 else "#10B981"
            st.markdown(
                f'<span style="color:#94A3B8">{r["date"].strftime("%Y-%m-%d")}</span>  '
                f'<span style="color:{col};font-weight:600">{r["gap_pct"]:+.1f}%</span>  '
                f'<span style="color:#64748B">'
                f'{r["old_close"]:.3f} → {r["new_close"]:.3f}</span>',
                unsafe_allow_html=True,
            )


# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("## 🔥  Natural Gas (NG=F) — Live Signal Dashboard")
live_str = f"  ·  Intraday: **${now_px:.3f}**" if live else ""
ts = live["time"].strftime("%H:%M") if live else "—"
n_rolls = len([r for r in roll_info
               if r["date"] >= pd.Timestamp.now() - pd.DateOffset(months=12)])
st.caption(
    f"IBS + 20-Day Extreme  ·  Last settlement: **{st8['last_dt'].date()}**  "
    f"·  Close: **${last_c:.3f}**{live_str}  ·  As of {ts}  "
    f"·  *Auto-refreshes every 5 min*  ·  "
    f"Roll-adjusted ({n_rolls} rolls in 12 mo)"
)
st.divider()


# ─── SIGNAL ROW ───────────────────────────────────────────────────────────────
sig_col, m1, m2, m3, m4 = st.columns([2.2, 1.1, 1.1, 1.0, 1.0])

with sig_col:
    if d_cur == 1:
        cls = "sig-long";  lbl = "LONG";  sub = "Full position open"
    elif d_cur == -1:
        cls = "sig-short"; lbl = "SHORT"; sub = "Full position open"
    else:
        cls = "sig-flat";  lbl = "FLAT";  sub = "No open signal — waiting for entry"
    st.markdown(f"""
<div class="signal-wrap">
  <div style="color:#64748B;font-size:10px;letter-spacing:1.2px;text-transform:uppercase">MODEL SIGNAL</div>
  <div class="sig-badge {cls}">{lbl}</div>
  <div class="sig-sub">{sub}</div>
</div>""", unsafe_allow_html=True)

with m1:
    if d_cur != 0 and st8["entry_date"] is not None:
        days_in  = (pd.Timestamp.now() - st8["entry_date"]).days
        open_ret = ((now_px / st8["epx"] - 1) * 100 if d_cur == 1
                    else (st8["epx"] / now_px - 1) * 100)
        st.metric("Days in Trade", f"{days_in}d", f"{open_ret:+.1f}% open P&L")
    else:
        st.metric("Days in Trade", "—", "Flat")

with m2:
    if d_cur != 0 and st8["epx"]:
        lbl2 = st8["entry_date"].strftime("%b %d, %Y") if st8["entry_date"] else ""
        st.metric("Entry Price", fp(st8["epx"]), lbl2)
    else:
        st.metric("Entry Price", "—", "Flat")

with m3:
    ibs_d = ("🟢 Long zone"  if last_ib < 0.20 else
             "🔴 Short zone" if last_ib > 0.80 else
             "⚡ Exit zone"  if last_ib > 0.50 else "Neutral")
    st.metric("IBS", f"{last_ib:.3f}", ibs_d)

with m4:
    if d_cur == 1:
        level_lbl = "20d High (short trigger)"
        level_val = fp(trg["rmax20"])
        dist = (trg["rmax20"] / now_px - 1) * 100
        level_d = f"{dist:+.1f}% from current"
    elif d_cur == -1:
        level_lbl = "20d Low (long trigger)"
        level_val = fp(trg["rmin20"])
        dist = (trg["rmin20"] / now_px - 1) * 100
        level_d = f"{dist:+.1f}% from current"
    else:
        level_lbl = "20d Low / High"
        level_val = f"{fp(trg['rmin20'])} / {fp(trg['rmax20'])}"
        level_d   = "entry thresholds"
    st.metric(level_lbl, level_val, level_d)


# ─── P&L STRIP ────────────────────────────────────────────────────────────────
if has_pos and my_entry and my_entry > 0:
    st.divider()
    pts     = now_px - my_entry if my_side == "Long" else my_entry - now_px
    pnl_usd = pts * 10000 * my_qty
    pnl_pct = pts / my_entry * 100
    pp1, pp2, pp3, pp4, pp5 = st.columns(5)
    with pp1: st.metric("My Side",  my_side,
                         f"×{my_qty} contract{'s' if my_qty > 1 else ''}")
    with pp2: st.metric("My Entry", fp(my_entry))
    with pp3: st.metric("Current",  fp(now_px))
    with pp4:
        st.metric("P&L (pts)",  f"{pts:+.3f}", f"{pnl_pct:+.2f}%",
                  delta_color="normal" if pts >= 0 else "inverse")
    with pp5:
        st.metric("P&L ($)", f"${pnl_usd:+,.0f}", "$10k per pt per contract",
                  delta_color="normal" if pnl_usd >= 0 else "inverse")

st.divider()


# ─── TRIGGER LEVEL PANEL ──────────────────────────────────────────────────────
src = (f"H&nbsp;=&nbsp;{fp(trg['h'])}&nbsp;&nbsp;"
       f"L&nbsp;=&nbsp;{fp(trg['l'])}&nbsp;&nbsp;"
       f"Range&nbsp;=&nbsp;${trg['rng']:.3f}")
if live:
    src += f"&nbsp;&nbsp;·&nbsp;&nbsp;Current&nbsp;=&nbsp;{fp(now_px)}"

st.markdown(
    f"#### ⚡&nbsp; Signal Triggers&nbsp;&nbsp;"
    f'<span style="color:#64748B;font-size:13px;font-weight:400">({src})</span>',
    unsafe_allow_html=True,
)

tc1, tc2, tc3, tc4 = st.columns(4)

IBS_EXIT_CARD = tcard(
    "IBS EXIT LEVEL · IBS = 0.50",
    trg["ibs_exit"], now_px,
    "Close full position when day's close crosses this level",
    extra_cls="tc-info",
)

# ── FLAT ────────────────────────────────────────────────────────────────────
if d_cur == 0:
    with tc1:
        note_long = (f"IBS ≤ {fp(trg['ibs_long'])}  ·  20d Low ≤ {fp(trg['rmin20'])}"
                     f"  ·  binding: {trg['long_bind']}  ·  needs close ≤")
        st.markdown(tcard(
            "LONG ENTRY — binding",
            trg["long_trigger"], now_px, note_long,
        ), unsafe_allow_html=True)
    with tc2:
        st.markdown(IBS_EXIT_CARD, unsafe_allow_html=True)
    with tc3:
        note_short = (f"IBS ≥ {fp(trg['ibs_short'])}  ·  20d High ≥ {fp(trg['rmax20'])}"
                      f"  ·  binding: {trg['short_bind']}  ·  needs close ≥")
        st.markdown(tcard(
            "SHORT ENTRY — binding",
            trg["short_trigger"], now_px, note_short,
        ), unsafe_allow_html=True)
    with tc4:
        ibs_color = "#10B981" if last_ib < 0.20 else "#EF4444" if last_ib > 0.80 else "#94A3B8"
        st.markdown(
            f'<div class="tc">'
            f'<div class="tc-lbl">MARKET STATUS</div>'
            f'<div style="margin:6px 0 4px">'
            f'  <span style="color:#64748B">IBS:&nbsp;&nbsp;&nbsp;&nbsp;</span>'
            f'  <span style="color:{ibs_color};font-weight:700;font-family:monospace">{last_ib:.3f}</span>'
            f'</div>'
            f'<div style="margin:4px 0">'
            f'  <span style="color:#64748B">20d Low: </span>'
            f'  <span style="color:#10B981;font-weight:700;font-family:monospace">{fp(trg["rmin20"])}</span>'
            f'</div>'
            f'<div style="margin:4px 0">'
            f'  <span style="color:#64748B">20d High:</span>'
            f'  <span style="color:#EF4444;font-weight:700;font-family:monospace">{fp(trg["rmax20"])}</span>'
            f'</div>'
            f'<div class="tc-note">No open position — awaiting entry signal</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── LONG ─────────────────────────────────────────────────────────────────────
elif d_cur == 1:
    with tc1:
        if st8["epx"]:
            open_r = (now_px / st8["epx"] - 1) * 100
            note   = (f"Open P&L: {open_r:+.2f}%  ·  "
                      f"entered {st8['entry_date'].strftime('%b %d') if st8['entry_date'] else '?'}")
            st.markdown(tcard("ENTRY (LONG)", st8["epx"], now_px, note,
                              extra_cls="tc-info"), unsafe_allow_html=True)
        else:
            st.markdown(status_card("ENTRY (LONG)", "open", ""), unsafe_allow_html=True)
    with tc2:
        st.markdown(IBS_EXIT_CARD, unsafe_allow_html=True)
    with tc3:
        note_short = (f"IBS ≥ {fp(trg['ibs_short'])}  ·  20d High ≥ {fp(trg['rmax20'])}"
                      f"  ·  binding: {trg['short_bind']}  ·  needs close ≥")
        st.markdown(tcard(
            "SHORT ENTRY (next signal)",
            trg["short_trigger"], now_px, note_short,
        ), unsafe_allow_html=True)
    with tc4:
        dist20h = (trg["rmax20"] / now_px - 1) * 100
        ibs_color = "#10B981" if last_ib < 0.20 else "#EF4444" if last_ib > 0.80 else "#94A3B8"
        st.markdown(
            f'<div class="tc">'
            f'<div class="tc-lbl">POSITION MONITOR</div>'
            f'<div style="margin:6px 0 4px">'
            f'  <span style="color:#64748B">IBS:&nbsp;&nbsp;&nbsp;&nbsp;</span>'
            f'  <span style="color:{ibs_color};font-weight:700;font-family:monospace">{last_ib:.3f}</span>'
            f'  <span style="color:#64748B;font-size:10px"> (exit >0.50)</span>'
            f'</div>'
            f'<div style="margin:4px 0">'
            f'  <span style="color:#64748B">20d High:</span>'
            f'  <span style="color:#EF4444;font-weight:700;font-family:monospace">{fp(trg["rmax20"])}</span>'
            f'  <span style="color:#64748B;font-size:10px"> ({dist20h:+.1f}%)</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── SHORT ─────────────────────────────────────────────────────────────────────
else:
    with tc1:
        if st8["epx"]:
            open_r = (st8["epx"] / now_px - 1) * 100
            note   = (f"Open P&L: {open_r:+.2f}%  ·  "
                      f"entered {st8['entry_date'].strftime('%b %d') if st8['entry_date'] else '?'}")
            st.markdown(tcard("ENTRY (SHORT)", st8["epx"], now_px, note,
                              extra_cls="tc-info"), unsafe_allow_html=True)
        else:
            st.markdown(status_card("ENTRY (SHORT)", "open", ""), unsafe_allow_html=True)
    with tc2:
        st.markdown(IBS_EXIT_CARD, unsafe_allow_html=True)
    with tc3:
        note_long = (f"IBS ≤ {fp(trg['ibs_long'])}  ·  20d Low ≤ {fp(trg['rmin20'])}"
                     f"  ·  binding: {trg['long_bind']}  ·  needs close ≤")
        st.markdown(tcard(
            "LONG ENTRY (next signal)",
            trg["long_trigger"], now_px, note_long,
        ), unsafe_allow_html=True)
    with tc4:
        dist20l = (trg["rmin20"] / now_px - 1) * 100
        ibs_color = "#10B981" if last_ib < 0.20 else "#EF4444" if last_ib > 0.80 else "#94A3B8"
        st.markdown(
            f'<div class="tc">'
            f'<div class="tc-lbl">POSITION MONITOR</div>'
            f'<div style="margin:6px 0 4px">'
            f'  <span style="color:#64748B">IBS:&nbsp;&nbsp;&nbsp;&nbsp;</span>'
            f'  <span style="color:{ibs_color};font-weight:700;font-family:monospace">{last_ib:.3f}</span>'
            f'  <span style="color:#64748B;font-size:10px"> (exit <0.50)</span>'
            f'</div>'
            f'<div style="margin:4px 0">'
            f'  <span style="color:#64748B">20d Low: </span>'
            f'  <span style="color:#10B981;font-weight:700;font-family:monospace">{fp(trg["rmin20"])}</span>'
            f'  <span style="color:#64748B;font-size:10px"> ({dist20l:+.1f}%)</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.divider()


# ─── CHART ────────────────────────────────────────────────────────────────────
cutoff = pd.Timestamp.now() - pd.DateOffset(months=12)
df12   = df[df.index >= cutoff].copy()

long_entries  = [(t["entry_date"], t["entry_px"]) for t in trades
                 if t["side"] == "long"  and t["entry_date"] >= cutoff]
short_entries = [(t["entry_date"], t["entry_px"]) for t in trades
                 if t["side"] == "short" and t["entry_date"] >= cutoff]
ibs_exits     = [(t["exit_date"],  t["exit_px"])  for t in trades
                 if t["exit_date"] >= cutoff]

BG = "#0F172A"; GR = "#1E293B"

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    vertical_spacing=0.04, row_heights=[0.68, 0.32],
    subplot_titles=["NG=F — Price  (12 months, roll-adjusted)", "IBS"],
)

# Candlestick
fig.add_trace(go.Candlestick(
    x=df12.index, open=df12["open"], high=df12["high"],
    low=df12["low"],  close=df12["close"],
    increasing_line_color="#10B981", decreasing_line_color="#EF4444",
    increasing_fillcolor="#065f46",  decreasing_fillcolor="#7f1d1d",
    name="NG=F", showlegend=True, line_width=1,
), row=1, col=1)

# 20d rolling low / high bands on price chart
fig.add_trace(go.Scatter(
    x=df12.index, y=df12["rmin20"],
    line=dict(color="#10B981", width=1.0, dash="dot"),
    name="20d Low", opacity=0.7,
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=df12.index, y=df12["rmax20"],
    line=dict(color="#EF4444", width=1.0, dash="dot"),
    name="20d High", opacity=0.7,
), row=1, col=1)

# Entry / exit markers
for evts, sym, col, nm in [
    (long_entries,  "triangle-up",   "#10B981", "Long Entry"),
    (short_entries, "triangle-down", "#EF4444", "Short Entry"),
]:
    if evts:
        dd_, pp_ = zip(*evts)
        fig.add_trace(go.Scatter(
            x=list(dd_), y=list(pp_), mode="markers",
            marker=dict(symbol=sym, size=13, color=col,
                        line=dict(color=BG, width=1.5)), name=nm,
        ), row=1, col=1)

if ibs_exits:
    di, pi = zip(*ibs_exits)
    fig.add_trace(go.Scatter(
        x=list(di), y=list(pi), mode="markers",
        marker=dict(symbol="x", size=9, color="#F59E0B",
                    line=dict(width=2)), name="IBS Exit",
    ), row=1, col=1)

# Current price line
fig.add_hline(y=now_px, line_color="#60A5FA", line_width=1, line_dash="dot",
              annotation_text=f"  {fp(now_px)}", annotation_font_color="#60A5FA",
              row=1, col=1)

# Active trigger overlays
if d_cur == 0:
    lt = trg.get("long_trigger"); st_ = trg.get("short_trigger")
    if lt and lt > 0:
        fig.add_hline(y=lt, line_color="#10B981", line_width=0.9, line_dash="dash",
                      annotation_text=f"  Long entry {fp(lt)}",
                      annotation_font_color="#10B981", row=1, col=1)
    if st_ and st_ > 0:
        fig.add_hline(y=st_, line_color="#EF4444", line_width=0.9, line_dash="dash",
                      annotation_text=f"  Short entry {fp(st_)}",
                      annotation_font_color="#EF4444", row=1, col=1)
else:
    fig.add_hline(y=trg["ibs_exit"], line_color="#F59E0B", line_width=0.9, line_dash="dash",
                  annotation_text=f"  IBS exit {fp(trg['ibs_exit'])}",
                  annotation_font_color="#F59E0B", row=1, col=1)

# Roll-date vertical markers
for r in roll_info:
    if r["date"] >= cutoff:
        for row in (1, 2):
            fig.add_vline(x=r["date"], line_color="#334155",
                          line_width=1, line_dash="dot", row=row, col=1)

# IBS panel
fig.add_trace(go.Scatter(x=df12.index, y=df12["IBS"],
    line=dict(color="#86EFAC", width=1.5), name="IBS",
    fill="tozeroy", fillcolor="rgba(134,239,172,0.04)", showlegend=False,
), row=2, col=1)
for lv, lc, lt in [(0.20, "#10B981", "0.20"), (0.50, "#F59E0B", "0.50"), (0.80, "#EF4444", "0.80")]:
    fig.add_hline(y=lv, line_color=lc, line_width=0.7, line_dash="dot",
                  annotation_text=f"  {lt}", annotation_font_color=lc, row=2, col=1)
fig.add_hline(y=last_ib, line_color="#86EFAC", line_width=0.5, line_dash="dash",
              annotation_text=f"  {last_ib:.3f}", annotation_font_color="#86EFAC",
              row=2, col=1)

fig.update_layout(
    height=640, paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(color="#CBD5E1", size=11),
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor=BG,
                bordercolor="#334155", borderwidth=1, font_size=10),
    margin=dict(t=55, b=10, l=55, r=130),
)
for row in range(1, 3):
    fig.update_xaxes(showgrid=True, gridcolor=GR, zeroline=False, row=row, col=1)
    fig.update_yaxes(showgrid=True, gridcolor=GR, zeroline=False, row=row, col=1)
for ann in fig.layout.annotations:
    if ann.text not in ("NG=F — Price  (12 months, roll-adjusted)", "IBS"):
        ann.font.size = 10

st.plotly_chart(fig, use_container_width=True)


# ─── TRADE LOG ────────────────────────────────────────────────────────────────
st.divider()
st.markdown("#### 📋  Trade Log — IBS + 20-Day Extreme (most recent first)")
st.caption(
    "Entry: IBS < 0.20 AND close = 20d low (long) / IBS > 0.80 AND close = 20d high (short)  ·  "
    "Exit: IBS crosses 0.50"
)

recent = list(reversed(trades[-60:]))

if recent:
    rows = []
    for t in recent:
        side_lbl = "🟢 LONG" if t["side"] == "long" else "🔴 SHORT"
        rows.append({
            "Side":      side_lbl,
            "Entry":     pd.Timestamp(t["entry_date"]).strftime("%Y-%m-%d"),
            "Exit":      pd.Timestamp(t["exit_date"]).strftime("%Y-%m-%d"),
            "Entry $":   fp(t["entry_px"]),
            "Exit $":    fp(t["exit_px"]),
            "Return %":  f"{t['return_pct']:+.2f}%",
            "Hold (d)":  int(t["hold"]),
        })

    dft = pd.DataFrame(rows)

    def _cr(v):
        try:
            fv = float(str(v).replace("%", "").replace("+", ""))
            return "color: #10B981; font-weight:600" if fv >= 0 else "color: #EF4444; font-weight:600"
        except Exception:
            return ""

    try:
        styled = dft.style.map(_cr, subset=["Return %"])
    except AttributeError:
        styled = dft.style.applymap(_cr, subset=["Return %"])

    st.dataframe(styled, use_container_width=True, hide_index=True, height=520)
else:
    st.caption("No completed trades in data range.")


# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.divider()
n_tr  = len(trades)
wins  = sum(1 for t in trades if t["return_pct"] > 0)
wpct  = wins / n_tr * 100 if n_tr else 0
n_roll = len(roll_info)
st.caption(
    f"IBS + 20-Day Extreme  ·  Total trades: {n_tr}  ·  Win rate: {wpct:.1f}%  "
    f"·  Roll adjustments applied: {n_roll}  "
    f"·  Prices are back-adjusted for NG=F contract-roll continuity  "
    f"·  Data: Yahoo Finance NG=F  "
    f"·  ⚠️ For research purposes — not financial advice"
)
