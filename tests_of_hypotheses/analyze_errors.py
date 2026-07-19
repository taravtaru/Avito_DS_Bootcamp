"""
Error Analysis & Day-level Investigation
==========================================
Зачем: понять, почему AP проседает на 04-20 и 04-21, 
и что характерно для ошибок модели.
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
TARGET = "target"

# Импортируем pipeline из solve.py
import importlib.util
spec = importlib.util.spec_from_file_location("solve", ROOT / "solve.py")
solve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solve)

print("=" * 70)
print("ERROR ANALYSIS & DAY-LEVEL INVESTIGATION")
print("=" * 70)

# ── 1. Подготовка данных (как в solve.py) ──
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
# Для анализа используем простое TE (без OOF — не критично).
global_mean = train[TARGET].mean()
for col in te_cols:
    agg = train.groupby(col)[TARGET].agg(["mean", "count"])
    agg["te"] = (agg["count"] * agg["mean"] + 10 * global_mean) / (agg["count"] + 10)
    train[f"te_{col}"] = train[col].map(agg["te"]).fillna(global_mean)

all_feats = solve.get_feature_columns(train)
combo_cats = ["cat_source_x_center", "cat_region_x_segment", "cat_channel_x_tenure"]
all_feats = [c for c in all_feats if c not in combo_cats]
feature_cols = solve.select_features(all_feats)
num_feats = [c for c in feature_cols if c not in solve.CATEGORICAL_FEATURES]

# Split
train_part, valid_part = solve.make_time_split(train, val_days=4)
X_tr = train_part[num_feats].copy()
y_tr = train_part[TARGET].values
X_vl = valid_part[num_feats].copy()
y_vl = valid_part[TARGET].values
vl_dates = pd.to_datetime(valid_part["assignment_date"]).dt.date.values

# Быстрая модель для анализа
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

print("\n" + "=" * 70)
print("1. DAY-LEVEL ANALYSIS: ПОЧЕМУ 04-20 и 04-21 ПРОСЕДАЮТ?")
print("=" * 70)

# Статистика по дням
for date in sorted(np.unique(vl_dates)):
    mask = vl_dates == date
    day_df = valid_part[mask]
    y_day = y_vl[mask]
    s_day = scores[mask]
    ap = average_precision_score(y_day, s_day)
    
    print(f"\n--- {date} (AP={ap:.5f}) ---")
    print(f"  Total: {len(day_df)}, Positives: {int(y_day.sum())} ({y_day.mean()*100:.1f}%)")
    
    # Распределение скоров
    print(f"  Score stats: mean={s_day.mean():.4f}, std={s_day.std():.4f}, "
          f"median={np.median(s_day):.4f}")
    
    # Скоры для target=1 и target=0
    pos_scores = s_day[y_day == 1]
    neg_scores = s_day[y_day == 0]
    print(f"  Pos scores: mean={pos_scores.mean():.4f}, median={np.median(pos_scores):.4f}")
    print(f"  Neg scores: mean={neg_scores.mean():.4f}, median={np.median(neg_scores):.4f}")
    print(f"  Score separation: {pos_scores.mean() - neg_scores.mean():.4f}")
    
    # Ключевые фичи по дням
    key_feats = ["evt_n_unique_ctx", "evt_favorite_72h", "seller_page_views_14d",
                 "search_views_90d", "evt_chat_open_72h"]
    for f in key_feats:
        if f in day_df.columns:
            pos_mean = day_df.loc[y_day == 1, f].mean()
            neg_mean = day_df.loc[y_day == 0, f].mean()
            all_mean = day_df[f].mean()
            print(f"  {f}: all={all_mean:.3f}, pos={pos_mean:.3f}, neg={neg_mean:.3f}")

print("\n" + "=" * 70)
print("2. КАТЕГОРИАЛЬНОЕ РАСПРЕДЕЛЕНИЕ ПО ДНЯМ")
print("=" * 70)

for cat in ["lead_source", "call_center", "region", "car_segment"]:
    print(f"\n--- {cat} ---")
    for date in sorted(np.unique(vl_dates)):
        mask = vl_dates == date
        day_df = valid_part[mask]
        dist = day_df[cat].value_counts(normalize=True).head(5)
        target_rate = day_df.groupby(cat)[TARGET].mean()
        print(f"  {date}: {dict(dist.round(3))}")

print("\n" + "=" * 70)
print("3. TRAIN vs VALIDATION DISTRIBUTION SHIFT")
print("=" * 70)

# Проверяем drift ключевых фичей между train и каждым днём валидации
key_feats = ["evt_n_unique_ctx", "evt_favorite_72h", "seller_page_views_14d",
             "search_views_90d", "evt_chat_open_72h", "te_cat_source_x_center",
             "evt_recency_hours", "evt_mean_src_slot"]

train_means = train_part[key_feats].mean()
train_stds = train_part[key_feats].std()

for f in key_feats:
    print(f"\n  {f}:")
    tr_m = train_means[f]
    print(f"    Train: mean={tr_m:.4f}")
    for date in sorted(np.unique(vl_dates)):
        mask = vl_dates == date
        vl_m = valid_part.loc[mask, f].mean()
        drift = (vl_m - tr_m) / (train_stds[f] + 1e-10)
        flag = " <<<" if abs(drift) > 0.3 else ""
        print(f"    {date}: mean={vl_m:.4f} (drift={drift:+.3f}){flag}")

print("\n" + "=" * 70)
print("4. ERROR ANALYSIS: FALSE NEGATIVES & FALSE POSITIVES")
print("=" * 70)

# Определяем ошибки при threshold = median score для positives
threshold = np.median(scores[y_vl == 1])
print(f"Threshold (median of positive scores): {threshold:.4f}")

# False negatives: target=1, но score < threshold
fn_mask = (y_vl == 1) & (scores < threshold)
fn_df = valid_part[fn_mask]
print(f"\nFalse Negatives (target=1, low score): {fn_mask.sum()}")

# False positives: target=0, но score > threshold  
fp_mask = (y_vl == 0) & (scores > threshold)
fp_df = valid_part[fp_mask]
print(f"False Positives (target=0, high score): {fp_mask.sum()}")

# True positives
tp_mask = (y_vl == 1) & (scores >= threshold)
tp_df = valid_part[tp_mask]
print(f"True Positives: {tp_mask.sum()}")

# Сравним характеристики FN vs TP
compare_feats = [
    "evt_n_unique_ctx", "evt_favorite_72h", "evt_chat_open_72h",
    "evt_call_click_72h", "seller_page_views_14d", "search_views_90d",
    "evt_recency_hours", "evt_total_all", "evt_ctx_entropy",
    "evt_n_unique_ctx_72h", "evt_fav_per_view_72h",
    "user_active_days_30d", "user_age_days",
]
compare_feats = [f for f in compare_feats if f in valid_part.columns]

print(f"\n{'Feature':<35} {'TP mean':>10} {'FN mean':>10} {'FP mean':>10} {'Neg mean':>10}")
print("-" * 80)
for f in compare_feats:
    tp_m = tp_df[f].mean()
    fn_m = fn_df[f].mean()
    fp_m = fp_df[f].mean()
    neg_m = valid_part.loc[y_vl == 0, f].mean()
    print(f"  {f:<33} {tp_m:>10.3f} {fn_m:>10.3f} {fp_m:>10.3f} {neg_m:>10.3f}")

# FN по категориям
print(f"\nFalse Negatives по lead_source:")
print(fn_df["lead_source"].value_counts(normalize=True).head(10).to_string())
print(f"\nTrue Positives по lead_source:")
print(tp_df["lead_source"].value_counts(normalize=True).head(10).to_string())

print(f"\nFalse Negatives по call_center:")
print(fn_df["call_center"].value_counts(normalize=True).head(10).to_string())
print(f"\nTrue Positives по call_center:")
print(tp_df["call_center"].value_counts(normalize=True).head(10).to_string())

print("\n" + "=" * 70)
print("5. PER-DAY: ОШИБКИ НА 'ПЛОХИХ' ДНЯХ (04-20, 04-21)")
print("=" * 70)

good_dates = [np.unique(vl_dates)[0], np.unique(vl_dates)[3]]  # 04-19, 04-22
bad_dates = [np.unique(vl_dates)[1], np.unique(vl_dates)[2]]   # 04-20, 04-21

good_mask = np.isin(vl_dates, good_dates)
bad_mask = np.isin(vl_dates, bad_dates)

good_df = valid_part[good_mask]
bad_df = valid_part[bad_mask]
good_y = y_vl[good_mask]
bad_y = y_vl[bad_mask]
good_s = scores[good_mask]
bad_s = scores[bad_mask]

print(f"\nGood days (19, 22): {good_mask.sum()} samples, {int(good_y.sum())} pos ({good_y.mean()*100:.1f}%)")
print(f"Bad days  (20, 21): {bad_mask.sum()} samples, {int(bad_y.sum())} pos ({bad_y.mean()*100:.1f}%)")

# Сравнение TP score separation
for label, mask, y, s in [("Good", good_mask, good_y, good_s), ("Bad", bad_mask, bad_y, bad_s)]:
    pos_s = s[y == 1]
    neg_s = s[y == 0]
    print(f"\n{label} days:")
    print(f"  Pos score: mean={pos_s.mean():.4f}, std={pos_s.std():.4f}")
    print(f"  Neg score: mean={neg_s.mean():.4f}, std={neg_s.std():.4f}")
    print(f"  Separation: {pos_s.mean() - neg_s.mean():.4f}")

# Фичи: хорошие vs плохие дни
print(f"\n{'Feature':<35} {'Good pos':>10} {'Good neg':>10} {'Bad pos':>10} {'Bad neg':>10}")
print("-" * 80)
for f in compare_feats:
    gp = good_df.loc[good_y == 1, f].mean()
    gn = good_df.loc[good_y == 0, f].mean()
    bp = bad_df.loc[bad_y == 1, f].mean()
    bn = bad_df.loc[bad_y == 0, f].mean()
    flag = " <<<" if abs(gp - bp) / (abs(gp) + 1e-10) > 0.2 else ""
    print(f"  {f:<33} {gp:>10.3f} {gn:>10.3f} {bp:>10.3f} {bn:>10.3f}{flag}")

print("\n" + "=" * 70)
print("6. WEEKDAY EFFECT (train)")
print("=" * 70)

train["date"] = pd.to_datetime(train["assignment_date"])
train["weekday"] = train["date"].dt.day_name()
train["weekday_num"] = train["date"].dt.dayofweek

wd_stats = train.groupby(["assignment_date", "weekday"]).agg(
    n=("target", "size"),
    pos=("target", "sum"),
    rate=("target", "mean"),
).reset_index()

print("\nPer-date stats:")
for _, row in wd_stats.iterrows():
    print(f"  {row['assignment_date']} ({row['weekday'][:3]}): "
          f"n={int(row['n'])}, pos={int(row['pos'])}, rate={row['rate']:.4f}")

# Weekday summary
wd_summary = train.groupby("weekday_num").agg(
    weekday=("weekday", "first"),
    n=("target", "size"),
    rate=("target", "mean"),
).reset_index()
print("\nWeekday summary:")
for _, row in wd_summary.iterrows():
    print(f"  {row['weekday'][:3]}: n={int(row['n'])}, rate={row['rate']:.4f}")

# Валидация — какие дни недели?
print("\nValidation dates weekday:")
for date in sorted(np.unique(vl_dates)):
    d = pd.Timestamp(date)
    print(f"  {date}: {d.day_name()}")

print("\n" + "=" * 70)
print("7. TARGET RATE TREND OVER TIME")
print("=" * 70)

daily_rates = train.groupby("assignment_date").agg(
    n=("target", "size"),
    rate=("target", "mean"),
).reset_index()

for _, row in daily_rates.iterrows():
    bar = "#" * int(row["rate"] * 100)
    print(f"  {row['assignment_date']}: rate={row['rate']:.4f} n={int(row['n'])} {bar}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
