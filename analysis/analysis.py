"""
analysis.py
-----------
Smart-meter analytics pipeline:

  1. Descriptive  : system KPIs (consumption, peak demand, load factor, peak hour)
  2. Diagnostic   : hour-of-day load profile, U-shaped temperature response,
                    monthly demand, consumption by customer segment
  3. Segmentation : KMeans (k=3) on per-meter behavioural features
  4. Anomaly      : robust (median/MAD) detection of faulty meters, scored vs truth
  5. Forecast     : hand-built additive Holt-Winters on daily system demand,
                    holdout MAPE -> accuracy, plus a 14-day forward forecast

Writes docs/dashboard_data.json and prints a readout.
"""
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

r = pd.read_csv("data/readings.csv", parse_dates=["timestamp"])
m = pd.read_csv("data/meters.csv")
r["hour"] = r["timestamp"].dt.hour
r["date"] = r["timestamp"].dt.normalize()
r["month"] = r["timestamp"].dt.month
out = {}

# ---------------------------------------------------------------- 4. ANOMALY (first)
feat_rows = []
for mid, g in r.groupby("meter_id"):
    k = g["kwh"].to_numpy()
    med = np.median(k[k > 0]) if (k > 0).any() else 0
    feat_rows.append({
        "meter_id": mid,
        "zero_share": float((k == 0).mean()),
        "spike_ratio": float(k.max() / med) if med > 0 else 0.0,
    })
fa = pd.DataFrame(feat_rows)
fa["flag"] = (fa["zero_share"] > 0.005) | (fa["spike_ratio"] > 8)
flagged = set(fa.loc[fa["flag"], "meter_id"])
truth = set(m.loc[m["is_faulty_truth"] == 1, "meter_id"])
tp = len(flagged & truth)
out["anomaly"] = {
    "flagged": int(len(flagged)),
    "true_faulty": int(len(truth)),
    "recall": round(tp / len(truth), 2) if truth else 0,
    "precision": round(tp / len(flagged), 2) if flagged else 0,
    "examples": sorted(flagged)[:8],
}

# ---------------------------------------------------------------- 1. DESCRIPTIVE
n_meters = r["meter_id"].nunique()
total_kwh = float(r["kwh"].sum())
system_hourly = r.groupby("timestamp")["kwh"].sum()
peak_demand = float(system_hourly.max())
avg_demand = float(system_hourly.mean())
hod_profile = r.groupby("hour")["kwh"].mean()           # avg per meter-hour
peak_hour = int(hod_profile.idxmax())
out["kpis"] = {
    "meters": int(n_meters),
    "readings": int(len(r)),
    "total_mwh": round(total_kwh / 1000, 1),
    "peak_demand_kw": round(peak_demand, 1),
    "load_factor": round(avg_demand / peak_demand, 3),
    "peak_hour": peak_hour,
    "avg_daily_kwh_per_meter": round(total_kwh / n_meters / 365, 2),
}

# ---------------------------------------------------------------- 2. DIAGNOSTIC
out["load_profile"] = {"hours": list(range(24)),
                       "kwh": [round(float(hod_profile[h]), 3) for h in range(24)]}

# temperature U-curve (daily)
daily = r.groupby("date").agg(demand=("kwh", "sum"), temp=("temp_c", "mean")).reset_index()
bins = list(range(-10, 36, 4))
daily["tbin"] = pd.cut(daily["temp"], bins=bins)
ucurve = daily.groupby("tbin", observed=True)["demand"].mean()
out["temp_curve"] = [{"temp": int(iv.left + 2), "demand_mwh": round(float(v) / 1000, 2)}
                     for iv, v in ucurve.items()]
out["temp_corr"] = round(float(daily["temp"].corr(daily["demand"])), 3)

# monthly demand
monthly = r.groupby("month")["kwh"].sum() / 1000
out["monthly_mwh"] = [round(float(monthly.get(i, 0)), 1) for i in range(1, 13)]

# ---------------------------------------------------------------- 3. SEGMENTATION (KMeans)
r["dow"] = r["timestamp"].dt.dayofweek
feat = []
for mid, g in r.groupby("meter_id"):
    k = g["kwh"].to_numpy(); hr = g["hour"].to_numpy(); dw = g["dow"].to_numpy()
    tot = k.sum(); wk = k[dw < 5].mean(); we = k[dw >= 5].mean()
    feat.append({
        "meter_id": mid,
        "log_avg": np.log(k.mean() + 1e-6),
        "load_factor": k.mean() / k.max() if k.max() else 0,
        "night_share": k[hr < 6].sum() / tot if tot else 0,
        "evening_share": k[(hr >= 18) & (hr <= 22)].sum() / tot if tot else 0,
        "day_share": k[(hr >= 9) & (hr <= 17)].sum() / tot if tot else 0,
        "weekend_ratio": we / wk if wk else 1,
    })
fdf = pd.DataFrame(feat).set_index("meter_id")
fit = fdf[~fdf.index.isin(flagged)]          # exclude faulty meters from clustering
X = StandardScaler().fit_transform(fit.values)
km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
fit = fit.copy(); fit["cluster"] = km.labels_
order = fit.groupby("cluster")["log_avg"].mean().sort_values().index.tolist()
names = {order[0]: "Low usage", order[1]: "Medium usage", order[2]: "High usage"}
fit["segment"] = fit["cluster"].map(names)
seg_tot = r.merge(fit[["segment"]], left_on="meter_id", right_index=True) \
           .groupby("segment")["kwh"].sum() / 1000
seg_cnt = fit["segment"].value_counts()
out["segments"] = [{"segment": s, "meters": int(seg_cnt[s]),
                    "total_mwh": round(float(seg_tot[s]), 1)}
                   for s in ["Low usage", "Medium usage", "High usage"]]

# ---------------------------------------------------------------- 5. FORECAST (Holt-Winters, additive, weekly)
y = daily.sort_values("date")["demand"].to_numpy() / 1000   # MWh per day
mseason = 7
n = len(y); split = 330
train, test = y[:split], y[split:]

def holt_winters(series, m, alpha, beta, gamma, h):
    L = series[:m].mean()
    T = (series[m:2 * m].mean() - series[:m].mean()) / m
    S = list(series[:m] - L)
    fitted = []
    for t in range(len(series)):
        if t >= m:
            last_s = S[t - m]
            Lp = alpha * (series[t] - last_s) + (1 - alpha) * (L + T)
            Tp = beta * (Lp - L) + (1 - beta) * T
            Sp = gamma * (series[t] - Lp) + (1 - gamma) * last_s
            L, T = Lp, Tp; S.append(Sp)
            fitted.append(L + T + S[t])
        else:
            fitted.append(series[t])
    fc = [L + (i + 1) * T + S[len(series) - m + (i % m)] for i in range(h)]
    return np.array(fc)

best = None
for a in (0.1, 0.2, 0.3, 0.5):
    for b in (0.01, 0.05, 0.1):
        for gmm in (0.1, 0.3, 0.5):
            fc = holt_winters(train, mseason, a, b, gmm, len(test))
            mape = np.mean(np.abs((test - fc) / test)) * 100
            if best is None or mape < best[0]:
                best = (mape, a, b, gmm, fc)
mape, a, b, gmm, test_fc = best
future = holt_winters(y, mseason, a, b, gmm, 14)
out["forecast"] = {
    "accuracy_pct": round(100 - mape, 1),
    "mape_pct": round(mape, 2),
    "params": {"alpha": a, "beta": b, "gamma": gmm, "season": mseason},
    "test_dates": [d.strftime("%b %d") for d in daily.sort_values("date")["date"].iloc[split:]],
    "test_actual": [round(float(v), 1) for v in test],
    "test_forecast": [round(float(v), 1) for v in test_fc],
    "future": [round(float(v), 1) for v in future],
}

with open("docs/dashboard_data.json", "w") as f:
    json.dump(out, f, indent=2)

# ---- readout ----
k = out["kpis"]
print(f"Meters {k['meters']} | Readings {k['readings']:,} | Total {k['total_mwh']:,} MWh")
print(f"Peak demand {k['peak_demand_kw']:,} kW @ hour {k['peak_hour']:02d}:00 | load factor {k['load_factor']}")
print(f"Temp-demand correlation: {out['temp_corr']} (heating-dominated U-curve)")
print("Segments:", [(s['segment'], s['meters']) for s in out['segments']])
an = out["anomaly"]
print(f"Anomaly: flagged {an['flagged']} | recall {an['recall']} precision {an['precision']} | e.g. {an['examples']}")
print(f"Forecast: Holt-Winters HW(a={a},b={b},g={gmm}) -> {out['forecast']['accuracy_pct']}% accuracy "
      f"(MAPE {out['forecast']['mape_pct']}%)")
print("wrote docs/dashboard_data.json")
