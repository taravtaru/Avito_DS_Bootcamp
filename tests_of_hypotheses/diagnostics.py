"""
4 Diagnostics — локализация проблемы по сегментам и дням.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from lightgbm import LGBMClassifier
import lightgbm as lgb
from pathlib import Path

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
ROOT = Path(".")
DATA_DIR = ROOT / "data"

import importlib.util
spec = importlib.util.spec_from_file_location("solve", ROOT / "solve.py")
solve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solve)

# ── Подготовка ──
train, test, events = solve.load_data()
train_ev = solve.build_event_features(train, events)
train = train.merge(train_ev, on="lead_id", how="left")
train = solve.build_tabular_features(train)

train["cat_source_x_center"] = train["lead_source"] + "_" + train["call_center"]
train["cat_region_x_segment"] = train["region"] + "_" + train["car_segment"]
train["cat_channel_x_tenure"] = train["lead_channel"] + "_" + train["user_tenure_bucket"]

te_cols = solve.CATEGORICAL_FEATURES + [
    "cat_source_x_center", "cat_region_x_segment", "cat_channel_x_tenure",
]
gm = train["target"].mean()
for col in te_cols:
    agg = train.groupby(col)["target"].agg(["mean", "count"])
    agg["te"] = (agg["count"] * agg["mean"] + 10 * gm) / (agg["count"] + 10)
    train[f"te_{col}"] = train[col].map(agg["te"]).fillna(gm)

all_feats = solve.get_feature_columns(train)
combo_cats = ["cat_source_x_center", "cat_region_x_segment", "cat_channel_x_tenure"]
all_feats = [c for c in all_feats if c not in combo_cats]
feature_cols = solve.select_features(all_feats)
num_feats = [c for c in feature_cols if c not in solve.CATEGORICAL_FEATURES]

train_part, valid_part = solve.make_time_split(train, val_days=4)
X_tr = train_part[num_feats].copy()
y_tr = train_part["target"].values
X_vl = valid_part[num_feats].copy()
y_vl = valid_part["target"].values
vl_dates = pd.to_datetime(valid_part["assignment_date"]).dt.date.values

pos, neg = int(y_tr.sum()), len(y_tr) - int(y_tr.sum())
model = LGBMClassifier(
    n_estimators=2000, learning_rate=0.08, max_depth=4, num_leaves=59,
    min_child_samples=10, reg_alpha=8.46, reg_lambda=0.22,
    subsample=0.98, colsample_bytree=0.84,
    scale_pos_weight=neg/max(pos,1), random_state=RANDOM_STATE, verbose=-1,
)
model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])

scores = model.predict_proba(X_vl)[:, 1]
valid_part = valid_part.copy()
valid_part["pred_score"] = scores
valid_part["y"] = y_vl
valid_part["date"] = vl_dates

sorted_dates = sorted(np.unique(vl_dates))

# ══════════════════════════════════════════════════════════════════════
# ДИАГНОСТИКА 1: AP по lead_source x день
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DIAG 1: AP by lead_source x day")
print("=" * 70)

sources = sorted(valid_part["lead_source"].unique())
header = f"{'lead_source':<12}" + "".join(f"{'  ' + str(d):<16}" for d in sorted_dates) + "   OVERALL"
print(header)
print("-" * len(header))

for src in sources:
    row = f"{src:<12}"
    for date in sorted_dates:
        mask = (valid_part["date"] == date) & (valid_part["lead_source"] == src)
        sub = valid_part[mask]
        if sub["y"].sum() == 0 or len(sub) < 5:
            row += f"{'  -':>16}"
        else:
            ap = average_precision_score(sub["y"], sub["pred_score"])
            n_pos = int(sub["y"].sum())
            row += f"  {ap:.4f} ({n_pos:>3})"
    # Overall
    mask_all = valid_part["lead_source"] == src
    sub_all = valid_part[mask_all]
    if sub_all["y"].sum() > 0:
        ap_all = average_precision_score(sub_all["y"], sub_all["pred_score"])
        row += f"  {ap_all:.4f} ({int(sub_all['y'].sum()):>3})"
    print(row)

# Total row
row = f"{'ALL':<12}"
for date in sorted_dates:
    mask = valid_part["date"] == date
    sub = valid_part[mask]
    ap = average_precision_score(sub["y"], sub["pred_score"])
    row += f"  {ap:.4f} ({int(sub['y'].sum()):>3})"
ap_all = average_precision_score(valid_part["y"], valid_part["pred_score"])
row += f"  {ap_all:.4f} ({int(valid_part['y'].sum()):>3})"
print(row)

# ══════════════════════════════════════════════════════════════════════
# ДИАГНОСТИКА 2: AP по activity level x день
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DIAG 2: AP by activity level x day")
print("=" * 70)

# Определяем activity по медиане evt_total_all
median_activity = valid_part["evt_total_all"].median()
valid_part["activity_level"] = np.where(
    valid_part["evt_total_all"].fillna(0) >= median_activity, "high", "low"
)

header = f"{'activity':<12}" + "".join(f"{'  ' + str(d):<16}" for d in sorted_dates) + "   OVERALL"
print(header)
print("-" * len(header))

for level in ["high", "low"]:
    row = f"{level:<12}"
    for date in sorted_dates:
        mask = (valid_part["date"] == date) & (valid_part["activity_level"] == level)
        sub = valid_part[mask]
        if sub["y"].sum() == 0:
            row += f"{'  -':>16}"
        else:
            ap = average_precision_score(sub["y"], sub["pred_score"])
            n_pos = int(sub["y"].sum())
            n_tot = len(sub)
            row += f"  {ap:.4f} ({n_pos:>3})"
    mask_all = valid_part["activity_level"] == level
    sub_all = valid_part[mask_all]
    ap_all = average_precision_score(sub_all["y"], sub_all["pred_score"])
    row += f"  {ap_all:.4f} ({int(sub_all['y'].sum()):>3})"
    print(row)

# Также по call_center
print("\n--- AP by call_center x day ---")
header = f"{'call_center':<12}" + "".join(f"{'  ' + str(d):<16}" for d in sorted_dates) + "   OVERALL"
print(header)
print("-" * len(header))

for cc in sorted(valid_part["call_center"].unique()):
    row = f"{cc:<12}"
    for date in sorted_dates:
        mask = (valid_part["date"] == date) & (valid_part["call_center"] == cc)
        sub = valid_part[mask]
        if sub["y"].sum() == 0:
            row += f"{'  -':>16}"
        else:
            ap = average_precision_score(sub["y"], sub["pred_score"])
            row += f"  {ap:.4f} ({int(sub['y'].sum()):>3})"
    mask_all = valid_part["call_center"] == cc
    sub_all = valid_part[mask_all]
    ap_all = average_precision_score(sub_all["y"], sub_all["pred_score"])
    row += f"  {ap_all:.4f} ({int(sub_all['y'].sum()):>3})"
    print(row)

# ══════════════════════════════════════════════════════════════════════
# ДИАГНОСТИКА 3: Состав позитивов 21 апреля vs остальные
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DIAG 3: Positive composition — 04-21 vs other days")
print("=" * 70)

apr21 = valid_part[(valid_part["date"] == sorted_dates[2]) & (valid_part["y"] == 1)]
others = valid_part[(valid_part["date"] != sorted_dates[2]) & (valid_part["y"] == 1)]

print(f"\n04-21 positives: {len(apr21)}, Others positives: {len(others)}")

for cat in ["lead_source", "call_center", "region", "car_segment", "user_tenure_bucket", "price_bucket"]:
    print(f"\n--- {cat} ---")
    d21 = apr21[cat].value_counts(normalize=True).round(3)
    doth = others[cat].value_counts(normalize=True).round(3)
    all_vals = sorted(set(d21.index) | set(doth.index))
    print(f"  {'value':<20} {'04-21':>8} {'others':>8} {'delta':>8}")
    for v in all_vals:
        v21 = d21.get(v, 0)
        voth = doth.get(v, 0)
        delta = v21 - voth
        flag = " <<<" if abs(delta) > 0.05 else ""
        print(f"  {str(v):<20} {v21:>8.3f} {voth:>8.3f} {delta:>+8.3f}{flag}")

# Числовые характеристики позитивов
print(f"\n--- Numeric features of positives ---")
num_compare = [
    "evt_n_unique_ctx", "evt_favorite_72h", "evt_chat_open_72h",
    "evt_call_click_72h", "seller_page_views_14d", "search_views_90d",
    "evt_recency_hours", "evt_total_all", "evt_n_unique_ctx_72h",
    "user_active_days_30d", "evt_fav_per_view_72h",
]
num_compare = [f for f in num_compare if f in valid_part.columns]

print(f"  {'feature':<30} {'04-21 pos':>10} {'other pos':>10} {'delta%':>8}")
for f in num_compare:
    m21 = apr21[f].mean()
    moth = others[f].mean()
    pct = (m21 - moth) / (abs(moth) + 1e-10) * 100
    flag = " <<<" if abs(pct) > 15 else ""
    print(f"  {f:<30} {m21:>10.3f} {moth:>10.3f} {pct:>+7.1f}%{flag}")

# ══════════════════════════════════════════════════════════════════════
# ДИАГНОСТИКА 4: Распределение скоров позитивов по сегментам
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DIAG 4: Score distribution of POSITIVES by segment")
print("=" * 70)

pos_df = valid_part[valid_part["y"] == 1].copy()

# По lead_source
print("\n--- By lead_source ---")
for src in sources:
    sub = pos_df[pos_df["lead_source"] == src]
    if len(sub) == 0:
        continue
    q = sub["pred_score"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    low_conf = (sub["pred_score"] < 0.3).mean()
    print(f"  {src:<8} n={len(sub):>4}  "
          f"p10={q[0.1]:.3f}  p25={q[0.25]:.3f}  median={q[0.5]:.3f}  "
          f"p75={q[0.75]:.3f}  p90={q[0.9]:.3f}  "
          f"low_conf(<0.3)={low_conf*100:.1f}%")

# По activity level
print("\n--- By activity level ---")
for level in ["high", "low"]:
    sub = pos_df[pos_df["activity_level"] == level]
    if len(sub) == 0:
        continue
    q = sub["pred_score"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    low_conf = (sub["pred_score"] < 0.3).mean()
    print(f"  {level:<8} n={len(sub):>4}  "
          f"p10={q[0.1]:.3f}  p25={q[0.25]:.3f}  median={q[0.5]:.3f}  "
          f"p75={q[0.75]:.3f}  p90={q[0.9]:.3f}  "
          f"low_conf(<0.3)={low_conf*100:.1f}%")

# По lead_source x activity
print("\n--- By lead_source x activity ---")
for src in sources:
    for level in ["high", "low"]:
        sub = pos_df[(pos_df["lead_source"] == src) & (pos_df["activity_level"] == level)]
        if len(sub) < 5:
            continue
        q = sub["pred_score"].quantile([0.1, 0.25, 0.5])
        low_conf = (sub["pred_score"] < 0.3).mean()
        print(f"  {src:<8} {level:<6} n={len(sub):>4}  "
              f"p10={q[0.1]:.3f}  p25={q[0.25]:.3f}  median={q[0.5]:.3f}  "
              f"low_conf(<0.3)={low_conf*100:.1f}%")

# Скоры позитивов per day per source
print("\n--- Positive scores by day x lead_source ---")
header = f"{'source':<10}" + "".join(f"{'  ' + str(d):<22}" for d in sorted_dates)
print(header)
for src in sources:
    row = f"{src:<10}"
    for date in sorted_dates:
        sub = pos_df[(pos_df["date"] == date) & (pos_df["lead_source"] == src)]
        if len(sub) < 3:
            row += f"{'  -':>22}"
        else:
            row += f"  med={sub['pred_score'].median():.3f} low={sub['pred_score'].lt(0.3).mean()*100:.0f}%"
    print(row)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
