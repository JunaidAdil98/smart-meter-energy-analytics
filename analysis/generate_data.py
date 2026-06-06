"""
generate_data.py
----------------
Simulates a smart-meter (AMI) dataset for 144 meters at hourly resolution over
2024 -> 1,261,440 readings. Consumption is built with genuine structure so the
analysis recovers real signal:

  - hour-of-day profiles differ by customer type (residential evening peak,
    business daytime peak, industrial flat-high)
  - weekday/weekend effects
  - U-shaped temperature response (heating in winter, cooling in summer)
  - a handful of FAULTY meters (stuck-at-zero dropouts + spikes) for anomaly detection

Writes:
  data/meters.csv          (144 rows: meter metadata + latent type)
  data/readings.csv        (full 1.26M hourly readings -- regenerate locally)
  data/readings_sample.csv (14-day sample, small, committed to the repo)

Synthetic data for a portfolio demo -- NOT real meter data.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(11)
HOURS = pd.date_range("2024-01-01", periods=8760, freq="h")   # 365 days
N_HOURS = len(HOURS)
hod = HOURS.hour.to_numpy()
dow = HOURS.dayofweek.to_numpy()           # 0=Mon .. 6=Sun
doy = HOURS.dayofyear.to_numpy()

# ---- daily temperature: seasonal sine (cold Jan, warm Jul) + noise ----
day_temp = 9 + 13 * -np.cos(2 * np.pi * (np.arange(365)) / 365) \
           + RNG.normal(0, 2.2, 365)       # ~ -4C winter .. 22C summer
temp = day_temp[doy - 1] + 3 * np.sin(2 * np.pi * (hod - 15) / 24)  # daily swing
temp = np.round(temp, 1)

# ---- hour-of-day shape templates (length 24), normalized around 1.0 ----
def norm(a):
    a = np.asarray(a, float); return a / a.mean()

shape = {
    "residential": norm([.45,.4,.38,.37,.4,.5,.75,.95,.8,.65,.6,.62,
                         .65,.6,.58,.6,.7,.95,1.35,1.5,1.4,1.15,.85,.6]),
    "business":    norm([.3,.28,.27,.27,.3,.4,.6,.9,1.3,1.5,1.55,1.5,
                         1.45,1.5,1.5,1.45,1.3,1.0,.7,.5,.42,.38,.34,.32]),
    "industrial":  norm([.85,.83,.82,.82,.85,.95,1.1,1.2,1.25,1.25,1.25,1.2,
                         1.2,1.25,1.25,1.2,1.15,1.1,1.05,1.0,.98,.95,.9,.88]),
}
# weekend multiplier by type (Mon..Sun)
weekend = {
    "residential": np.array([1,1,1,1,1,1.08,1.10]),
    "business":    np.array([1,1,1,1,1,0.55,0.42]),
    "industrial":  np.array([1,1,1,1,1,0.80,0.72]),
}
# temperature response: U-shaped, heating-dominant
def temp_mult(t, heat, cool):
    cold = np.clip(16 - t, 0, None)
    hot = np.clip(t - 23, 0, None)
    return 1 + heat * cold + cool * hot

types = (["residential"] * 90) + (["business"] * 35) + (["industrial"] * 19)
RNG.shuffle(types)
faulty_ids = set(RNG.choice(144, size=6, replace=False).tolist())

meters, frames = [], []
for i, mtype in enumerate(types):
    mid = f"M{2000 + i}"
    base = {"residential": RNG.uniform(0.4, 1.1),
            "business": RNG.uniform(1.5, 4.0),
            "industrial": RNG.uniform(8.0, 22.0)}[mtype]
    heat = {"residential": 0.055, "business": 0.030, "industrial": 0.015}[mtype]
    cool = {"residential": 0.030, "business": 0.045, "industrial": 0.020}[mtype]

    load = (base
            * shape[mtype][hod]
            * weekend[mtype][dow]
            * temp_mult(temp, heat, cool)
            * RNG.normal(1.0, 0.08, N_HOURS))
    load = np.clip(load, 0.01, None)

    faulty = i in faulty_ids
    if faulty:
        # stuck-at-zero dropout for a random ~10-day window
        s = int(RNG.integers(0, N_HOURS - 240))
        load[s:s + int(RNG.integers(120, 240))] = 0.0
        # a few extreme spikes
        spikes = RNG.choice(N_HOURS, size=8, replace=False)
        load[spikes] *= RNG.uniform(6, 12, size=8)

    meters.append({"meter_id": mid, "customer_type": mtype,
                   "base_load_kw": round(base, 3), "is_faulty_truth": int(faulty)})
    frames.append(pd.DataFrame({"meter_id": mid, "timestamp": HOURS,
                                "kwh": np.round(load, 4), "temp_c": temp}))

readings = pd.concat(frames, ignore_index=True)
meters_df = pd.DataFrame(meters)

meters_df.to_csv("data/meters.csv", index=False)
readings.to_csv("data/readings.csv", index=False)
# committed sample: first 14 days, all meters
sample = readings[readings["timestamp"] < "2024-01-15"]
sample.to_csv("data/readings_sample.csv", index=False)

print(f"meters   : {len(meters_df)}")
print(f"readings : {len(readings):,}")
print(f"total consumption: {readings['kwh'].sum()/1000:,.1f} MWh")
print(f"faulty meters (ground truth): {sorted(m['meter_id'] for m in meters if m['is_faulty_truth'])}")
