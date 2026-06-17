"""
NG Natural Gas — IBS + 20-Day Extreme Live Signal Dashboard
  • Yahoo Finance NG=F  (5-min auto-refresh)
  • Entry:  IBS < 0.20  AND  close ≤ 20-day rolling low   (long)
          IBS > 0.80  AND  close ≦ 20-day rolling high  (short)
  • Exit:   IBS > 0.50  (long)  /  IBS < 0.50  (short)
  • Exact IBS and 20-day extreme trigger prices for today, live intraday H/L
  • Manual position tracker with $P&L
  • Roll-adjusted prices: NG futures contract rolls are back-adjusted for continuity
"""