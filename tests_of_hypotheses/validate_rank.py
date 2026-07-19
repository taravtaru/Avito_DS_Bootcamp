"""
Validate Ranking Objectives (lambdarank / pairwise)
===================================================
Ablation study for ranking vs classification on 3 time folds.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker
from xgboost import XGBClassifier, XGBRanker
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from pathlib import Path

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
ROOT = Path(".")
DATA_DIR = ROOT / "data"
TARGET = "target"

import importlib.util
spec = importlib.util.spec_from_file_location("solve", ROOT / "solve.py")
solve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solve)

# ──────────────────────────────────────────────────────────────────────
# 1. Loading Base Data
# ──────────────────────────────────────────────────────────────────────
print("Loading and building base features...")
train, test, events = solve.load_data()
train_ev = solve.build_event_features(train, events)
train = train.merge(train_ev, on="lead_id", how="left")
train = solve.build_tabular_features(train)

all_cols = set(train.columns)
exclude = solve.FEATURES_TO_DROP | set(solve.CATEGORICAL_FEATURES) | {
    "second_event_type", "last_event_type"
}

# Just the basic v5 features that were proven good
bad_v4 = {
    "last_event_was_favorite", "last_2_events",
    "evt_n_transitions", "evt_transition_rate", "evt_acceleration",
    "has_recent_favorite_24h", "has_recent_favorite_72h",
    "has_recent_call_click_24h", "has_recent_call_click_72h",
    "has_recent_chat_open_24h", "has_recent_chat_open_72h",
    "seller_quality_proxy", "item_fav_rate_90d", "item_contact_rate_90d",
    "seller_views_share_7d", "seller_views_share_14d", "seller_views_share_30d"
}
for et in solve.EVENT_TYPES:
    bad_v4.add(f"evt_hours_since_last_{et}")

base_v5_feats = [c for c in all_cols if c not in exclude and c not in bad_v4 and c not in solve.NON_FEATURE_COLUMNS]

train["date_dt"] = pd.to_datetime(train["assignment_date"])
train["date_str"] = train["date_dt"].dt.date.astype(str)

# IMPORTANT: Rankers require data to be sorted by query ID (date)
train = train.sort_values("date_dt").reset_index(drop=True)

dates = train["date_dt"].dt.date
ordered_dates = sorted(dates.unique())

folds = [
    ("Fold 1 (15-16)", ordered_dates[-8], ordered_dates[-7]),
    ("Fold 2 (17-18)", ordered_dates[-6], ordered_dates[-5]),
    ("Fold 3 (19-22)", ordered_dates[-4], ordered_dates[-1])
]

# Hyperparameters (same as in v5)
lgbm_params = solve.LGBM_PARAMS.copy()
lgbm_params["n_estimators"] = 1500

xgb_params = solve.XGBOOST_PARAMS.copy()
xgb_params["n_estimators"] = 1500

def get_group_counts(df):
    return df.groupby("date_str", sort=False).size().values

def get_qids(df):
    date_to_id = {d: i for i, d in enumerate(df["date_str"].unique())}
    return df["date_str"].map(date_to_id).values


def train_eval_lgbm_cls(X_tr, y_tr, X_vl, y_vl, dates_vl):
    pos = int(y_tr.sum())
    neg = len(y_tr) - pos
    params = lgbm_params.copy()
    params["scale_pos_weight"] = neg / max(pos, 1)
    
    model = LGBMClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    preds = model.predict_proba(X_vl)[:, 1]
    return solve.daily_average_precision(y_vl, preds, dates_vl)

def train_eval_lgbm_rank(X_tr, y_tr, X_vl, y_vl, dates_vl, df_tr, df_vl):
    params = lgbm_params.copy()
    params["objective"] = "lambdarank"
    params["metric"] = "map"
    
    # Scale pos weight isn't standard for lambdarank, but we can leave it out
    if "scale_pos_weight" in params: del params["scale_pos_weight"]
    
    group_tr = get_group_counts(df_tr)
    group_vl = get_group_counts(df_vl)
    
    model = LGBMRanker(**params)
    model.fit(
        X_tr, y_tr, group=group_tr,
        eval_set=[(X_vl, y_vl)], eval_group=[group_vl],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
    )
    preds = model.predict(X_vl)
    return solve.daily_average_precision(y_vl, preds, dates_vl)

def train_eval_xgb_cls(X_tr, y_tr, X_vl, y_vl, dates_vl):
    pos = int(y_tr.sum())
    neg = len(y_tr) - pos
    params = xgb_params.copy()
    params["scale_pos_weight"] = neg / max(pos, 1)
    params["early_stopping_rounds"] = 100
    
    model = XGBClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
    preds = model.predict_proba(X_vl)[:, 1]
    return solve.daily_average_precision(y_vl, preds, dates_vl)

def train_eval_xgb_rank(X_tr, y_tr, X_vl, y_vl, dates_vl, df_tr, df_vl):
    params = xgb_params.copy()
    params["objective"] = "rank:pairwise"
    if "scale_pos_weight" in params: del params["scale_pos_weight"]
    params["early_stopping_rounds"] = 100
    
    qid_tr = get_qids(df_tr)
    qid_vl = get_qids(df_vl)
    
    model = XGBRanker(**params)
    model.fit(
        X_tr, y_tr, qid=qid_tr,
        eval_set=[(X_vl, y_vl)], eval_qid=[qid_vl],
        verbose=False
    )
    preds = model.predict(X_vl)
    return solve.daily_average_precision(y_vl, preds, dates_vl)


print("\n" + "="*60 + "\nTIME-BASED CV: CLS vs RANK\n" + "="*60)

results = {"LGBM_CLS": [], "LGBM_RANK": [], "XGB_CLS": [], "XGB_RANK": []}

for fold_name, val_start, val_end in folds:
    print(f"\n--- {fold_name} ---")
    tr_mask = dates < val_start
    vl_mask = (dates >= val_start) & (dates <= val_end)
    
    df_tr, df_vl = train.loc[tr_mask], train.loc[vl_mask]
    X_t, y_t = df_tr[base_v5_feats], df_tr[TARGET].values
    X_v, y_v = df_vl[base_v5_feats], df_vl[TARGET].values
    d_v = df_vl["date_dt"].dt.date.values
    
    # LGBM
    dap_lgb_cls = train_eval_lgbm_cls(X_t, y_t, X_v, y_v, d_v)
    print(f"LGBM Classify: {dap_lgb_cls:.4f}")
    results["LGBM_CLS"].append(dap_lgb_cls)
    
    dap_lgb_rnk = train_eval_lgbm_rank(X_t, y_t, X_v, y_v, d_v, df_tr, df_vl)
    print(f"LGBM Rank (lambdarank): {dap_lgb_rnk:.4f}")
    results["LGBM_RANK"].append(dap_lgb_rnk)
    
    # XGB
    dap_xgb_cls = train_eval_xgb_cls(X_t, y_t, X_v, y_v, d_v)
    print(f"XGB Classify: {dap_xgb_cls:.4f}")
    results["XGB_CLS"].append(dap_xgb_cls)
    
    dap_xgb_rnk = train_eval_xgb_rank(X_t, y_t, X_v, y_v, d_v, df_tr, df_vl)
    print(f"XGB Rank (pairwise): {dap_xgb_rnk:.4f}")
    results["XGB_RANK"].append(dap_xgb_rnk)

print("\n" + "="*60 + "\nSUMMARY (Mean Daily AP across 3 folds)\n" + "="*60)
for k, v in results.items():
    print(f"{k}: Mean={np.mean(v):.4f} | Min={np.min(v):.4f} | Vals={np.round(v, 4)}")
