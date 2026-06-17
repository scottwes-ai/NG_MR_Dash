"""
NG Natural Gas — Scaled v2 Live Signal Dashboard
  • Yahoo Finance NG=F  (5-min auto-refresh)
  • Scaled v2: full entry → close 2/3 at IBS exit → hold 1/3 to SAR reverse
  • Re-entry: same-direction signal while 1/3 SAR runner is open → add 2/3 back
  • Exact RSI(3)/IBS trigger prices for today, live intraday H/L
  • Manual position tracker with $P&L
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
    page_title="NG Signal · Scaled v2",
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

# ─── DATA ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_daily():
    df = yf.download("NG=F", period="4y", auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df[["open","high","low","close"]].dropna().copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df["gap"] = df.index.to_series().diff().dt.days
    df = df[(df["high"] > df["low"]) & (df["gap"] <= 14)].copy()
    d  = df["close"].diff()
    U  = d.clip(lower=0); D = (-d).clip(lower=0)
    avgD = D.rolling(3).mean().replace(0, np.nan)
    df["RSI"] = (100 - 100 / (1 + U.rolling(3).mean() / avgD)).fillna(100)
    df["IBS"] = (df["close"] - df["low"]) / (df["high"] - df["low"])
    return df.dropna(subset=["RSI","IBS"])

@st.cache_data(ttl=60)
def get_intraday():
    try:
        df5 = yf.download("NG=F", period="1d", interval="5m",
                          auto_adjust=True, progress=False)
        if df5.empty: return None
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
    cv = df["close"].values; ib = df["IBS"].values; rs = df["RSI"].values
    dt = df.index; n = len(df)
    ls = (rs < 20) & (ib < 0.20)
    ss = (rs > 80) & (ib > 0.80)

    sar_tr=[]; ibs_tr=[]; eq=np.ones(n); val=1.0
    d=0; sep=None; si=None; sal=0.0
    iep=None; ii=None; io=False; ib2=0.0; ic=1.0

    def sf(x): return x/sep   if d==1 else sep/x
    def bf(x): return x/iep   if d==1 else iep/x
    def te(x): return sal*sf(x) + ib2*ic*(bf(x) if io else 1.0)

    for i in range(n):
        x = cv[i]
        if d == 0:
            eq[i] = val
            if ls[i] or ss[i]:
                d=1 if ls[i] else -1
                sep=x; si=i; sal=val/3
                iep=x; ii=i; io=True; ib2=val*2/3; ic=1.0
            continue
        if i == si: eq[i]=val; continue
        eq[i] = te(x)
        if io:
            xe = (d==1 and ib[i]>0.50) or (d==-1 and ib[i]<0.50)
            if xe:
                r=bf(x)-1; ic*=(1+r); io=False
                ibs_tr.append(dict(side="long" if d==1 else "short",
                    entry_date=dt[ii], exit_date=dt[i],
                    entry_px=iep, exit_px=x, return_pct=r*100, hold=i-ii))
                iep=None; ii=None
        if not io:
            sm = (d==1 and ls[i]) or (d==-1 and ss[i])
            if sm: iep=x; ii=i; io=True
        rv = (d==1 and ss[i]) or (d==-1 and ls[i])
        if rv:
            if io:
                r=bf(x)-1; ic*=(1+r); io=False
                ibs_tr.append(dict(side="long" if d==1 else "short",
                    entry_date=dt[ii], exit_date=dt[i],
                    entry_px=iep, exit_px=x, return_pct=r*100, hold=i-ii))
                iep=None; ii=None
            fin=te(x)
            sar_tr.append(dict(side="long" if d==1 else "short",
                entry_date=dt[si], exit_date=dt[i],
                entry_px=sep, exit_px=x, return_pct=(fin/val-1)*100, hold=i-si))
            val=fin; eq[i]=val; d=-d
            sep=x; si=i; sal=val/3
            iep=x; ii=i; io=True; ib2=val*2/3; ic=1.0

    return sar_tr, ibs_tr, pd.Series(eq, index=dt), dict(
        d=d, sep=sep, si=dt[si] if si is not None else None,
        io=io, iep=iep, ii=dt[ii] if ii is not None else None,
        val=val, last_c=cv[-1], last_r=rs[-1], last_ib=ib[-1], last_dt=dt[-1],
    )

# ─── TRIGGER PRICES ───────────────────────────────────────────────────────────
def calc_triggers(df, live):
    """
    Exact closing prices that would fire each signal today.
    IBS triggers from today's live H/L.
    RSI triggers from Cutler's RSI algebra (SMA-based).
    """
    cv = df["close"].values; n = len(cv)
    h = live["high"] if live else float(df["high"].iloc[-1])
    l = live["low"]  if live else float(df["low"].iloc[-1])
    rng = max(h - l, 1e-9)
    t = dict(h=h, l=l, rng=rng,
             ibs_long  = l + 0.20*rng,
             ibs_exit  = l + 0.50*rng,
             ibs_short = l + 0.80*rng,
             rsi_long=None, rsi_short=None,
             long_trigger=None, short_trigger=None,
             long_bind="?", short_bind="?")
    if n >= 3:
        d1 = float(cv[-2]-cv[-3]); d2 = float(cv[-1]-cv[-2])
        U1=max(d1,0); D1=max(-d1,0); U2=max(d2,0); D2=max(-d2,0)
        prev = float(cv[-1])
        t["rsi_long"]  = prev + D1+D2 - 4*(U1+U2)
        t["rsi_short"] = prev + 4*(D1+D2) - (U1+U2)
        t["long_trigger"]  = min(t["ibs_long"],  t["rsi_long"])
        t["short_trigger"] = max(t["ibs_short"], t["rsi_short"])
        t["long_bind"]  = "IBS" if t["ibs_long"] <= t["rsi_long"]  else "RSI"
        t["short_bind"] = "IBS" if t["ibs_short"] >= t["rsi_short"] else "RSI"
    return t

# ─── HTML HELPERS ─────────────────────────────────────────────────────────────
def fp(p):   return f"${p:.3f}" if p is not None else "—"

def delta_span(price, ref, invert=False):
    pct = (price/ref-1)*100
    if invert: pct = -pct
    col = "#10B981" if pct >= 0 else "#EF4444"
    arr = "▲" if pct >= 0 else "▼"
    return (f'<span style="color:{col}">{arr}&nbsp;{abs(pct):.2f}%'
            f'&nbsp;&nbsp;(${abs(price-ref):.3f})</span>')

def tcard(label, price, ref, note="", warn=False):
    if price is None:
        body = '<div class="tc-px">—</div>'
        delta = ""
    else:
        near = abs(price-ref)/max(ref,1e-9) < 0.003
        hit_tag = ' <span style="color:#F59E0B;font-weight:700">✓&nbsp;HIT</span>' if near else ""
        body  = f'<div class="tc-px">{fp(price)}{hit_tag}</div>'
        delta = delta_span(price, ref)
    extra = "tc-warn" if warn else ""
    return (f'<div class="tc {extra}">'
            f'<div class="tc-lbl">{label}</div>'
            f'{body}'
            f'<div style="margin:4px 0">{delta}</div>'
            f'<div class="tc-note">{note}</div>'
            f'</div>')

# ─── LOAD ─────────────────────────────────────────────────────────────────────
df = get_daily()
if df.empty:
    st.error("⚠️ Yahoo Finance returned no data for NG=F. Check your connection.")
    st.stop()

live  = get_intraday()
sar_tr, ibs_tr, eq_s, st8 = run_engine(df)
trg   = calc_triggers(df, live)

now_px  = live["price"] if live else st8["last_c"]
last_c  = st8["last_c"]; last_r = st8["last_r"]; last_ib = st8["last_ib"]
d_cur   = st8["d"]; io_cur = st8["io"]

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📍  My Position")
    has_pos = st.toggle("Track open position", key="has_pos")
    my_side=None; my_entry=None; my_qty=1
    if has_pos:
        my_side  = st.selectbox("Side", ["Long","Short"])
        my_entry = st.number_input("Entry Price ($)", min_value=0.001,
                                   step=0.001, format="%.3f")
        my_qty   = st.number_input("Contracts", min_value=1, value=1)
        st.caption("NG futures: **$10,000 / point / contract**")

    st.divider()
    st.markdown("### 📋  Signal Rules")
    st.markdown("""
🟢 **Long entry**: RSI(3) < 20  AND  IBS < 0.20  
🔴 **Short entry**: RSI(3) > 80  AND  IBS > 0.80  
⚡ **IBS exit**: IBS > 0.50 (long) / < 0.50 (short) → **close 2/3**  
↔ **SAR reverse**: opposite signal → **close 1/3 runner**, enter opposite  
↩ **Re-entry**: same-dir signal while partial → **add 2/3 back**
""")

    st.divider()
    col_r, col_t = st.columns(2)
    with col_r:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear(); st.rerun()
    with col_t:
        st.caption(f"Updated:\n{datetime.now().strftime('%H:%M:%S')}")

    st.divider()
    st.markdown("### 📈  Backtest (2012+)")
    eq12 = eq_s[eq_s.index >= "2012"]
    if len(eq12) > 2:
        yrs  = (eq12.index[-1]-eq12.index[0]).days/365.25
        cagr = (eq12.iloc[-1]**(1/yrs)-1)*100
        dd   = (eq12-eq12.cummax())/eq12.cummax()*100
        vol  = eq12.pct_change().dropna().std()*np.sqrt(252)*100
        col1,col2 = st.columns(2)
        with col1:
            st.metric("CAGR",    f"{cagr:.1f}%")
            st.metric("Max DD",  f"{dd.min():.1f}%")
        with col2:
            st.metric("Sharpe",  f"{cagr/vol:.3f}")
            st.metric("Final",   f"{eq12.iloc[-1]:.1f}×")

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("## 🔥  Natural Gas (NG=F) — Live Signal Dashboard")
live_str = f"  ·  Intraday: **${now_px:.3f}**" if live else ""
ts = live["time"].strftime("%H:%M") if live else "—"
st.caption(
    f"Scaled v2 Strategy  ·  Last settlement: **{st8['last_dt'].date()}**  "
    f"·  Close: **${last_c:.3f}**{live_str}  ·  As of {ts}  "
    f"·  *Auto-refreshes every 5 min*"
)
st.divider()

# ─── SIGNAL ROW ───────────────────────────────────────────────────────────────
sig_col, m1, m2, m3, m4 = st.columns([2.2, 1.1, 1.1, 1.0, 1.0])

with sig_col:
    if d_cur == 1:
        cls="sig-long"; lbl="LONG"
        sub="Full Position (2/3 IBS + 1/3 SAR)" if io_cur else "Partial — 1/3 SAR Runner"
    elif d_cur == -1:
        cls="sig-short"; lbl="SHORT"
        sub="Full Position (2/3 IBS + 1/3 SAR)" if io_cur else "Partial — 1/3 SAR Runner"
    else:
        cls="sig-flat"; lbl="FLAT"; sub="No open signal — waiting for entry"
    st.markdown(f"""
<div class="signal-wrap">
  <div style="color:#64748B;font-size:10px;letter-spacing:1.2px;text-transform:uppercase">MODEL SIGNAL</div>
  <div class="sig-badge {cls}">{lbl}</div>
  <div class="sig-sub">{sub}</div>
</div>""", unsafe_allow_html=True)

with m1:
    if d_cur != 0 and st8["si"] is not None:
        days_in = (pd.Timestamp.now()-st8["si"]).days
        sar_ret = (now_px/st8["sep"]-1)*100 if d_cur==1 else (st8["sep"]/now_px-1)*100
        st.metric("Days in SAR", f"{days_in}d", f"{sar_ret:+.1f}% runner P&L")
    else:
        st.metric("Days in SAR", "—", "Flat")

with m2:
    if d_cur != 0 and st8["sep"]:
        entry_lbl = st8["si"].strftime("%b %d, %Y") if st8["si"] else ""
        st.metric("SAR Entry", fp(st8["sep"]), entry_lbl)
    else:
        st.metric("SAR Entry", "—", "Flat")

with m3:
    rsi_d = ("🟢 Long zone" if last_r < 20 else
             "🔴 Short zone" if last_r > 80 else "Neutral")
    st.metric("RSI (3)", f"{last_r:.1f}", rsi_d)

with m4:
    ibs_d = ("🟢 Entry zone"   if last_ib < 0.20 else
             "🔴 Short entry"  if last_ib > 0.80 else
             "⚡ Exit zone"    if last_ib > 0.50 else "Neutral")
    st.metric("IBS", f"{last_ib:.3f}", ibs_d)

# ─── P&L STRIP ────────────────────────────────────────────────────────────────
if has_pos and my_entry and my_entry > 0:
    st.divider()
    pts = now_px-my_entry if my_side=="Long" else my_entry-now_px
    pnl_usd = pts*10000*my_qty
    pnl_pct = pts/my_entry*100
    pp1,pp2,pp3,pp4,pp5 = st.columns(5)
    with pp1: st.metric("My Side",   my_side, f"×{my_qty} contract{'s' if my_qty>1 else ''}")
    with pp2: st.metric("My Entry",  fp(my_entry))
    with pp3: st.metric("Current",   fp(now_px))
    with pp4:
        st.metric("P&L (pts)",   f"{pts:+.3f}",
                  f"{pnl_pct:+.2f}%",
                  delta_color="normal" if pts>=0 else "inverse")
    with pp5:
        st.metric("P&L ($)",  f"${pnl_usd:+,.0f}",
                  "$10k per pt per contract",
                  delta_color="normal" if pnl_usd>=0 else "inverse")

st.divider()

# ─── TRIGGER LEVEL PANEL ──────────────────────────────────────────────────────
src = f"H&nbsp;=&nbsp;{fp(trg['h'])}&nbsp;&nbsp;L&nbsp;=&nbsp;{fp(trg['l'])}&nbsp;&nbsp;Range&nbsp;=&nbsp;${trg['rng']:.3f}"
if live:
    src += f"&nbsp;&nbsp;·&nbsp;&nbsp;Current&nbsp;=&nbsp;{fp(now_px)}"
st.markdown(
    f"#### ⚡&nbsp; Signal Triggers&nbsp;&nbsp;"
    f'<span style="color:#64748B;font-size:13px;font-weight:400">({src})</span>',
    unsafe_allow_html=True
)

tc1,tc2,tc3,tc4 = st.columns(4)

if d_cur == 0:
    with tc1: st.markdown(tcard(
        "LONG ENTRY — binding",
        trg.get("long_trigger"), now_px,
        f"Binding constraint: {trg['long_bind']}  ·  needs close ≤ this price"
    ), unsafe_allow_html=True)
    with tc2: st.markdown(tcard(
        "IBS component  (IBS = 0.20)",
        trg["ibs_long"], now_px,
        f"RSI trigger: {fp(trg.get('rsi_long'))}  ·  both must be met"
    ), unsafe_allow_html=True)
    with tc3: st.markdown(tcard(
        "SHORT ENTRY — binding",
        trg.get("short_trigger"), now_px,
        f"Binding constraint: {trg['short_bind']}  ·  needs close ≥ this price"
    ), unsafe_allow_html=True)
    with tc4: st.markdown(tcard(
        "IBS component  (IBS = 0.80)",
        trg["ibs_short"], now_px,
        f"RSI trigger: {fp(trg.get('rsi_short'))}  ·  both must be met"
    ), unsafe_allow_html=True)

elif d_cur == 1:
    if io_cur:
        with tc1: st.markdown(tcard(
            "IBS EXIT → Close 2/3  (IBS = 0.50)",
            trg["ibs_exit"], now_px,
            "Closes IBS leg; 1/3 SAR runner continues"
        ), unsafe_allow_html=True)
        with tc2: st.markdown(tcard(
            "SAR REVERSE → Close All",
            trg.get("short_trigger"), now_px,
            f"Binding: {trg['short_bind']}  ·  IBS {fp(trg['ibs_short'])}  ·  RSI {fp(trg.get('rsi_short'))}"
        ), unsafe_allow_html=True)
        with tc3:
            if st8["iep"]:
                lr = (now_px/st8["iep"]-1)*100
                note = f"IBS leg P&L: {lr:+.2f}%  ·  entered {st8['ii'].strftime('%b %d') if st8['ii'] else '?'}"
                st.markdown(tcard("IBS Leg Entry", st8["iep"], now_px, note), unsafe_allow_html=True)
        with tc4:
            if st8["sep"]:
                sr = (now_px/st8["sep"]-1)*100
                note = f"SAR runner P&L: {sr:+.2f}%  ·  entered {st8['si'].strftime('%b %d, %Y') if st8['si'] else '?'}"
                st.markdown(tcard("SAR Runner Entry (1/3)", st8["sep"], now_px, note), unsafe_allow_html=True)
    else:
        with tc1: st.markdown(tcard(
            "IBS RE-ENTRY → Add 2/3 Long",
            trg.get("long_trigger"), now_px,
            f"Binding: {trg['long_bind']}  ·  IBS {fp(trg['ibs_long'])}  ·  RSI {fp(trg.get('rsi_long'))}"
        ), unsafe_allow_html=True)
        with tc2: st.markdown(tcard(
            "IBS component  (IBS = 0.20)", trg["ibs_long"], now_px,
            "RSI < 20 also required simultaneously"
        ), unsafe_allow_html=True)
        with tc3: st.markdown(tcard(
            "SAR REVERSE → Close 1/3",
            trg.get("short_trigger"), now_px,
            f"Binding: {trg['short_bind']}  ·  IBS {fp(trg['ibs_short'])}  ·  RSI {fp(trg.get('rsi_short'))}"
        ), unsafe_allow_html=True)
        with tc4:
            if st8["sep"]:
                sr = (now_px/st8["sep"]-1)*100
                dheld = (pd.Timestamp.now()-st8["si"]).days if st8["si"] else 0
                note = f"1/3 P&L: {sr:+.2f}%  ·  {dheld}d held  ·  entered {st8['si'].strftime('%b %d, %Y') if st8['si'] else '?'}"
                st.markdown(tcard(f"SAR Runner (1/3 Long)", st8["sep"], now_px, note), unsafe_allow_html=True)

else:
    if io_cur:
        with tc1: st.markdown(tcard(
            "IBS EXIT → Close 2/3  (IBS = 0.50)",
            trg["ibs_exit"], now_px,
            "Closes IBS short leg; 1/3 SAR runner continues"
        ), unsafe_allow_html=True)
        with tc2: st.markdown(tcard(
            "SAR REVERSE → Close All",
            trg.get("long_trigger"), now_px,
            f"Binding: {trg['long_bind']}  ·  IBS {fp(trg['ibs_long'])}  ·  RSI {fp(trg.get('rsi_long'))}"
        ), unsafe_allow_html=True)
        with tc3:
            if st8["iep"]:
                lr = (st8["iep"]/now_px-1)*100
                note = f"IBS short leg P&L: {lr:+.2f}%  ·  entered {st8['ii'].strftime('%b %d') if st8['ii'] else '?'}"
                st.markdown(tcard("IBS Short Leg Entry", st8["iep"], now_px, note), unsafe_allow_html=True)
        with tc4:
            if st8["sep"]:
                sr = (st8["sep"]/now_px-1)*100
                note = f"SAR short runner P&L: {sr:+.2f}%  ·  entered {st8['si'].strftime('%b %d, %Y') if st8['si'] else '?'}"
                st.markdown(tcard("SAR Short Runner (1/3)", st8["sep"], now_px, note), unsafe_allow_html=True)
    else:
        with tc1: st.markdown(tcard(
            "IBS RE-ENTRY → Add 2/3 Short",
            trg.get("short_trigger"), now_px,
            f"Binding: {trg['short_bind']}  ·  IBS {fp(trg['ibs_short'])}  ·  RSI {fp(trg.get('rsi_short'))}"
        ), unsafe_allow_html=True)
        with tc2: st.markdown(tcard(
            "IBS component  (IBS = 0.80)", trg["ibs_short"], now_px,
            "RSI > 80 also required simultaneously"
        ), unsafe_allow_html=True)
        with tc3: st.markdown(tcard(
            "SAR REVERSE → Close 1/3",
            trg.get("long_trigger"), now_px,
            f"Binding: {trg['long_bind']}  ·  IBS {fp(trg['ibs_long'])}  ·  RSI {fp(trg.get('rsi_long'))}"
        ), unsafe_allow_html=True)
        with tc4:
            if st8["sep"]:
                sr = (st8["sep"]/now_px-1)*100
                dheld = (pd.Timestamp.now()-st8["si"]).days if st8["si"] else 0
                note = f"1/3 P&L: {sr:+.2f}%  ·  {dheld}d held  ·  entered {st8['si'].strftime('%b %d, %Y') if st8['si'] else '?'}"
                st.markdown(tcard(f"SAR Runner (1/3 Short)", st8["sep"], now_px, note), unsafe_allow_html=True)

st.divider()

# ─── CHART ────────────────────────────────────────────────────────────────────
cutoff = pd.Timestamp.now() - pd.DateOffset(months=12)
df12   = df[df.index >= cutoff].copy()

long_e  = [(t["entry_date"],t["entry_px"]) for t in sar_tr if t["side"]=="long"  and t["entry_date"]>=cutoff]
short_e = [(t["entry_date"],t["entry_px"]) for t in sar_tr if t["side"]=="short" and t["entry_date"]>=cutoff]
exits   = [(t["exit_date"], t["exit_px"])  for t in sar_tr                        if t["exit_date"] >=cutoff]
ibs_x   = [(t["exit_date"], t["exit_px"])  for t in ibs_tr                        if t["exit_date"] >=cutoff]

BG="#0F172A"; GR="#1E293B"

fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True,
    vertical_spacing=0.035, row_heights=[0.58,0.21,0.21],
    subplot_titles=["NG=F — Price  (12 months)", "RSI (3)", "IBS"],
)

fig.add_trace(go.Candlestick(
    x=df12.index, open=df12["open"], high=df12["high"],
    low=df12["low"], close=df12["close"],
    increasing_line_color="#10B981", decreasing_line_color="#EF4444",
    increasing_fillcolor="#065f46", decreasing_fillcolor="#7f1d1d",
    name="NG=F", showlegend=True, line_width=1,
), row=1, col=1)

for evts, sym, col, nm in [
    (long_e,  "triangle-up",   "#10B981", "Long Entry"),
    (short_e, "triangle-down", "#EF4444", "Short Entry"),
]:
    if evts:
        dd_, pp_ = zip(*evts)
        fig.add_trace(go.Scatter(x=list(dd_), y=list(pp_), mode="markers",
            marker=dict(symbol=sym, size=13, color=col,
                        line=dict(color=BG, width=1.5)), name=nm), row=1, col=1)

if exits:
    de, pe = zip(*exits)
    fig.add_trace(go.Scatter(x=list(de), y=list(pe), mode="markers",
        marker=dict(symbol="circle-open", size=11, color="#F59E0B",
                    line=dict(width=2)), name="SAR Reverse"), row=1, col=1)

if ibs_x:
    di, pi = zip(*ibs_x)
    fig.add_trace(go.Scatter(x=list(di), y=list(pi), mode="markers",
        marker=dict(symbol="x", size=8, color="#A78BFA",
                    line=dict(width=1.5)), name="IBS Exit"), row=1, col=1)

fig.add_hline(y=now_px, line_color="#60A5FA", line_width=1, line_dash="dot",
              annotation_text=f"  {fp(now_px)}", annotation_font_color="#60A5FA",
              row=1, col=1)

if d_cur == 0:
    lt = trg.get("long_trigger"); st_ = trg.get("short_trigger")
    if lt and lt > 0:
        fig.add_hline(y=lt, line_color="#10B981", line_width=0.9, line_dash="dash",
                      annotation_text=f"  Long entry {fp(lt)}", annotation_font_color="#10B981", row=1, col=1)
    if st_ and st_ > 0:
        fig.add_hline(y=st_, line_color="#EF4444", line_width=0.9, line_dash="dash",
                      annotation_text=f"  Short entry {fp(st_)}", annotation_font_color="#EF4444", row=1, col=1)
else:
    fig.add_hline(y=trg["ibs_exit"], line_color="#F59E0B", line_width=0.9, line_dash="dash",
                  annotation_text=f"  IBS exit {fp(trg['ibs_exit'])}", annotation_font_color="#F59E0B", row=1, col=1)
    rev_px = trg.get("short_trigger" if d_cur==1 else "long_trigger")
    if rev_px and rev_px > 0:
        fig.add_hline(y=rev_px, line_color="#F472B6", line_width=0.9, line_dash="dash",
                      annotation_text=f"  SAR reverse {fp(rev_px)}", annotation_font_color="#F472B6", row=1, col=1)

fig.add_trace(go.Scatter(x=df12.index, y=df12["RSI"],
    line=dict(color="#38BDF8", width=1.5), name="RSI(3)", showlegend=False), row=2, col=1)
fig.add_hrect(y0=0,  y1=20,  fillcolor="#064e3b", opacity=0.20, line_width=0, row=2, col=1)
fig.add_hrect(y0=80, y1=100, fillcolor="#7f1d1d", opacity=0.20, line_width=0, row=2, col=1)
for lv, lc in [(20,"#10B981"),(50,"#475569"),(80,"#EF4444")]:
    fig.add_hline(y=lv, line_color=lc, line_width=0.7, line_dash="dot", row=2, col=1)
fig.add_hline(y=last_r, line_color="#38BDF8", line_width=0.5, line_dash="dash",
              annotation_text=f"  {last_r:.1f}", annotation_font_color="#38BDF8", row=2, col=1)

fig.add_trace(go.Scatter(x=df12.index, y=df12["IBS"],
    line=dict(color="#86EFAC", width=1.5), name="IBS",
    fill="tozeroy", fillcolor="rgba(134,239,172,0.04)", showlegend=False), row=3, col=1)
for lv, lc, lt in [(0.20,"#10B981","0.20"),(0.50,"#F59E0B","0.50"),(0.80,"#EF4444","0.80")]:
    fig.add_hline(y=lv, line_color=lc, line_width=0.7, line_dash="dot",
                  annotation_text=f"  {lt}", annotation_font_color=lc, row=3, col=1)
fig.add_hline(y=last_ib, line_color="#86EFAC", line_width=0.5, line_dash="dash",
              annotation_text=f"  {last_ib:.3f}", annotation_font_color="#86EFAC", row=3, col=1)

fig.update_layout(
    height=680, paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(color="#CBD5E1", size=11),
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor=BG,
                bordercolor="#334155", borderwidth=1, font_size=10),
    margin=dict(t=55, b=10, l=55, r=130),
)
for row in range(1, 4):
    fig.update_xaxes(showgrid=True, gridcolor=GR, zeroline=False, row=row, col=1)
    fig.update_yaxes(showgrid=True, gridcolor=GR, zeroline=False, row=row, col=1)
for ann in fig.layout.annotations:
    if ann.text not in ("NG=F — Price  (12 months)", "RSI (3)", "IBS"):
        ann.font.size = 10

st.plotly_chart(fig, use_container_width=True)

# ─── TRADE LOG ────────────────────────────────────────────────────────────────
st.divider()
tl_col, _ = st.columns([3, 1])
with tl_col:
    st.markdown("#### 📋  Recent SAR Periods")

if sar_tr:
    rows = []
    for t in reversed(sar_tr[-25:]):
        rows.append({
            "Side":       ("🟢 LONG" if t["side"]=="long" else "🔴 SHORT"),
            "Entry":      pd.Timestamp(t["entry_date"]).strftime("%Y-%m-%d"),
            "Exit":       pd.Timestamp(t["exit_date"]).strftime("%Y-%m-%d"),
            "Entry $":    fp(t["entry_px"]),
            "Exit $":     fp(t["exit_px"]),
            "Return":     f"{t['return_pct']:+.2f}%",
            "Hold (days)":int(t["hold"]),
        })
    dft = pd.DataFrame(rows)

    def _cr(v):
        try:
            fv = float(str(v).replace("%","").replace("+",""))
            return "color: #10B981; font-weight:600" if fv >= 0 else "color: #EF4444; font-weight:600"
        except Exception:
            return ""

    try:
        styled = dft.style.map(_cr, subset=["Return"])
    except AttributeError:
        styled = dft.style.applymap(_cr, subset=["Return"])

    st.dataframe(styled, use_container_width=True, hide_index=True, height=440)
else:
    st.caption("No completed SAR periods in data range.")

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.divider()
n_sar = len(sar_tr); n_ibs = len(ibs_tr)
wins  = sum(1 for t in sar_tr if t["return_pct"] > 0)
wpct  = wins/n_sar*100 if n_sar else 0
st.caption(
    f"Scaled v2  ·  SAR periods: {n_sar}  ·  IBS sub-trades: {n_ibs}  "
    f"·  Win rate: {wpct:.1f}%  ·  Data: Yahoo Finance NG=F  "
    f"·  ⚠️ For research purposes — not financial advice"
)
