import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score

# Конфигурация
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = Path("data")
TARGET = "target"

ID_COLUMNS = {"lead_id", "user_id"}
TIME_COLUMNS = {"assignment_ts", "assignment_date"}
NON_FEATURE_COLUMNS = ID_COLUMNS | TIME_COLUMNS | {TARGET}

CATEGORICAL_FEATURES = [
    "lead_source", "call_center", "region",
    "car_segment", "lead_channel", "user_tenure_bucket", "price_bucket",
    "recency_bucket"
]

EVENT_WINDOWS_HOURS = [24, 72, 168]
EVENT_TYPES = ["item_view", "search", "favorite", "chat_open", "call_click"]
CTX_SEQ_VALUES = ["c01", "c02", "c04", "c06", "c08"]

def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    events = pd.read_csv(DATA_DIR / "events.csv")
    
    train["assignment_ts"] = pd.to_datetime(train["assignment_ts"])
    test["assignment_ts"] = pd.to_datetime(test["assignment_ts"])
    events["event_ts"] = pd.to_datetime(events["event_ts"])
    
    print(f"Train: {train.shape}, Test: {test.shape}, Events: {events.shape}")
    print(f"Target rate: {train[TARGET].mean():.4f}")
    return train, test, events

train, test, events = load_data()

def build_event_features(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    # Оставляем только те события, которые произошли до выдачи лида
    merged = df[["lead_id", "assignment_ts"]].merge(events, on="lead_id", how="left")
    merged = merged[merged["event_ts"] < merged["assignment_ts"]].copy()
    merged["hours_before"] = (merged["assignment_ts"] - merged["event_ts"]).dt.total_seconds() / 3600

    result = pd.DataFrame({"lead_id": df["lead_id"].values})

    # 1. Счетчики по временным окнам и типам событий
    for wh in EVENT_WINDOWS_HOURS:
        wn = f"{wh}h"
        we = merged[merged["hours_before"] <= wh]
        total = we.groupby("lead_id").size().rename(f"evt_total_{wn}")
        result = result.merge(total, on="lead_id", how="left")
        
        for et in EVENT_TYPES:
            c = we[we["event_type"] == et].groupby("lead_id").size().rename(f"evt_{et}_{wn}")
            result = result.merge(c, on="lead_id", how="left")

    # 2. Общие агрегации (уникальность, позиция, недавность)
    agg = merged.groupby("lead_id").agg(
        evt_total_all=("event_ts", "size"),
        evt_n_unique_types=("event_type", "nunique"),
        evt_n_unique_ctx=("ctx_seq", "nunique"),
        evt_mean_src_slot=("src_slot", "mean"),
        evt_std_src_slot=("src_slot", "std"),
        evt_recency_hours=("hours_before", "min"),
    )
    result = result.merge(agg, on="lead_id", how="left")

    # 3. Распределение по контекстам (ctx_seq) и энтропия
    for cv in CTX_SEQ_VALUES:
        c = merged[merged["ctx_seq"] == cv].groupby("lead_id").size().rename(f"evt_ctx_{cv}_count")
        result = result.merge(c, on="lead_id", how="left")
        result[f"evt_ctx_{cv}_share"] = result[f"evt_ctx_{cv}_count"] / result["evt_total_all"].replace(0, np.nan)

    ctx_cols = [f"evt_ctx_{v}_count" for v in CTX_SEQ_VALUES]
    ctx_df = result[ctx_cols].fillna(0)
    ctx_total = ctx_df.sum(axis=1).replace(0, np.nan)
    ctx_probs = ctx_df.div(ctx_total, axis=0)
    result["evt_ctx_entropy"] = -(ctx_probs * np.log(ctx_probs + 1e-10)).sum(axis=1)

    # 4. Поведенческие ratios (доля контактов, доля избранного)
    for wh in EVENT_WINDOWS_HOURS:
        wn = f"{wh}h"
        fav, view = f"evt_favorite_{wn}", f"evt_item_view_{wn}"
        if fav in result.columns and view in result.columns:
            result[f"evt_fav_per_view_{wn}"] = result[fav] / result[view].replace(0, np.nan)
            
        ch, ca, tot = f"evt_chat_open_{wn}", f"evt_call_click_{wn}", f"evt_total_{wn}"
        if all(c in result.columns for c in [ch, ca, tot]):
            result[f"evt_contact_share_{wn}"] = (result[ch].fillna(0) + result[ca].fillna(0)) / result[tot].replace(0, np.nan)

    # 5. Интенсивность (events per active day)
    ad = merged.assign(ed=merged["event_ts"].dt.date).groupby("lead_id")["ed"].nunique().rename("evt_active_days")
    result = result.merge(ad, on="lead_id", how="left")
    result["evt_intensity"] = result["evt_total_all"] / result["evt_active_days"].replace(0, np.nan)

    for et in EVENT_TYPES:
        col = f"evt_{et}_168h"
        if col in result.columns:
            result[f"evt_{et}_share_7d"] = result[col] / result["evt_total_168h"].replace(0, np.nan)

    for hours, name in [(72, "72h"), (24, "24h")]:
        w = merged[merged["hours_before"] <= hours]
        nu = w.groupby("lead_id")["ctx_seq"].nunique().rename(f"evt_n_unique_ctx_{name}")
        result = result.merge(nu, on="lead_id", how="left")

    return result

print("Добавляем event-признаки...")
train_ev = build_event_features(train, events)
test_ev = build_event_features(test, events)
train = train.merge(train_ev, on="lead_id", how="left")
test = test.merge(test_ev, on="lead_id", how="left")

def build_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    # --- 1. Индикаторы целевых сегментов ---
    result["is_crm"] = (result["lead_source"] == "CRM").astype(int)
    result["is_model"] = (result["lead_source"] == "Model").astype(int)
    result["is_perf"] = (result["lead_source"] == "Perf").astype(int)

    # Определяем "холодных" пользователей (меньше 12 событий за все время - примерная медиана)
    evt_total = result.get("evt_total_all", pd.Series(0, index=result.index)).fillna(0)
    result["is_low_activity"] = (evt_total < 12).astype(int)

    recency = result.get("evt_recency_hours", pd.Series(np.nan, index=result.index))
    result["recency_bucket"] = pd.cut(
        recency, bins=[-1, 6, 24, 72, float("inf")],
        labels=["0-6h", "6-24h", "24-72h", "72h+"],
    ).astype(str).replace("nan", "no_events")

    result["has_recent_event_24h"] = (recency <= 24).astype(int).fillna(0)
    result["has_recent_event_72h"] = (recency <= 72).astype(int).fillna(0)
    result["has_seller_views"] = (result.get("seller_page_views_14d", 0) > 0).astype(int)

    spv7 = result.get("seller_page_views_7d", pd.Series(0))
    spv14 = result.get("seller_page_views_14d", pd.Series(0))
    result["seller_views_trend"] = spv7 / (spv14 + 1)

    # --- 2. Важнейшие пересечения (Interactions) ---
    result["crm_x_low_activity"] = result["is_crm"] * result["is_low_activity"]
    result["crm_x_recency"] = result["is_crm"] * recency.fillna(999)
    result["crm_x_seller_views"] = result["is_crm"] * result.get("seller_page_views_14d", 0)
    result["crm_x_has_recent_24h"] = result["is_crm"] * result["has_recent_event_24h"]

    result["low_act_x_seller_views"] = result["is_low_activity"] * result.get("seller_page_views_14d", 0)
    result["low_act_x_item_views"] = result["is_low_activity"] * result.get("item_views_90d", 0)
    result["low_act_x_search_views"] = result["is_low_activity"] * result.get("search_views_90d", 0)
    result["low_act_x_seller_inv"] = result["is_low_activity"] * result.get("seller_inventory_count", 0)

    # --- 3. Базовые отношения (Ratios) ---
    for window in ["30d", "90d"]:
        assigned = f"leadgen_prev_assigned_{window}"
        answered = f"leadgen_prev_answered_{window}"
        positive = f"leadgen_prev_positive_{window}"
        if assigned in result.columns and answered in result.columns:
            result[f"ratio_answered_assigned_{window}"] = result[answered] / result[assigned].replace(0, np.nan)
        if assigned in result.columns and positive in result.columns:
            result[f"ratio_positive_assigned_{window}"] = result[positive] / result[assigned].replace(0, np.nan)

    # --- 4. Тренды (Trends) ---
    trend_pfx = [
        "item_views", "item_favorites", "detail_expands", "photo_swipes",
        "seller_page_views", "search_views", "user_contacts", "chat_opens",
        "call_clicks", "active_days_auto",
    ]
    for prefix in trend_pfx:
        for wl, ws in [("30d", "14d"), ("90d", "30d")]:
            cl, cs = f"{prefix}_{wl}", f"{prefix}_{ws}"
            if cl in result.columns and cs in result.columns:
                result[f"trend_{prefix}_{wl}_vs_{ws}"] = result[cl] - result[cs]

    # --- 5. Относительная активность (3d vs 30d) ---
    for prefix in ["item_views", "search_views", "user_contacts", "chat_opens", "call_clicks"]:
        cs, cl = f"{prefix}_3d", f"{prefix}_30d"
        if cs in result.columns and cl in result.columns:
            result[f"rel_{prefix}_3d_to_30d"] = result[cs] / result[cl].replace(0, np.nan)

    if "user_active_days_30d" in result.columns and "prior_assignments_30d" in result.columns:
        result["interact_activity_x_assignments"] = result["user_active_days_30d"] * result["prior_assignments_30d"]
    if "seller_inventory_count" in result.columns and "seller_response_rate_30d" in result.columns:
        result["interact_seller_inv_x_response"] = result["seller_inventory_count"] * result["seller_response_rate_30d"]

    return result

print("Добавляем табличные признаки и сегменты...")
train = build_tabular_features(train)
test = build_tabular_features(test)

FEATURES_TO_DROP = {
    "rel_search_views_7d_to_90d", "similar_item_clicks_30d",
    "seller_response_rate_30d", "photo_swipes_30d",
    "trend_photo_swipes_30d_vs_14d", "trend_call_clicks_30d_vs_14d",
    "rel_call_clicks_7d_to_90d", "rel_chat_opens_7d_to_90d",
    "item_price_log", "chat_opens_90d", "evt_mean_price",
    "trend_item_favorites_7d_vs_3d",
    "ratio_positive_assigned_3d", "leadgen_prev_assigned_90d",
    "rel_user_contacts_7d_to_30d", "ratio_positive_assigned_7d",
    "trend_search_views_14d_vs_7d", "active_days_auto_14d",
    "rel_call_clicks_7d_to_30d", "evt_span_hours", "trend_chat_opens_14d_vs_7d",
    "seller_page_views_3d", "assignment_weekday", "similar_item_clicks_3d",
    "saved_search_matches_7d", "active_days_auto_3d",
    "item_favorites_14d", "search_views_3d", "item_favorites_3d",
    "total_contact_actions_7d", "leadgen_prev_assigned_7d",
    "active_days_auto_7d", "call_clicks_90d", "user_contacts_14d",
    "ratio_answered_assigned_14d", "rel_item_views_7d_to_90d",
    "trend_item_views_14d_vs_7d", "trend_item_views_90d_vs_30d",
    "leadgen_prev_positive_14d", "ratio_positive_answered_90d",
    "ratio_answered_assigned_1d", "ratio_positive_assigned_1d",
    "ratio_positive_answered_1d", "ratio_positive_answered_7d",
    "interact_price_x_age", "ratio_answered_assigned_3d",
    "ratio_positive_assigned_14d", "ratio_positive_answered_14d",
    "lead_channel", "is_weekend",
    "query_refinements_14d", "trend_detail_expands_30d_vs_14d",
    "trend_active_days_auto_14d_vs_7d", "leadgen_prev_assigned_3d",
    "chat_opens_14d", "total_activity_7d", "detail_expands_1d",
    "evt_item_view_168h", "user_contacts_1d", "leadgen_prev_positive_1d",
    "query_refinements_3d", "query_refinements_30d",
    "leadgen_prev_positive_90d",
}

def get_feature_columns(df):
    exclude = NON_FEATURE_COLUMNS | {"assignment_date_date"}
    all_feats = [c for c in df.columns if c not in exclude]
    selected = [c for c in all_feats if c not in FEATURES_TO_DROP]
    print(f"Отобрано признаков: {len(selected)} (исключено {len(all_feats)-len(selected)})")
    return selected

feature_cols = get_feature_columns(train)

def daily_average_precision(y_true, y_score, dates):
    daily_aps = []
    for date in np.unique(dates):
        mask = dates == date
        if y_true[mask].sum() == 0: continue
        daily_aps.append(average_precision_score(y_true[mask], y_score[mask]))
    return np.mean(daily_aps) if daily_aps else 0.0

def make_time_split(df, val_days=4):
    dates = pd.to_datetime(df["assignment_date"]).dt.date
    ordered = sorted(dates.unique())
    cutoff = ordered[-val_days]
    tr = df[dates < cutoff].copy()
    vl = df[dates >= cutoff].copy()
    print(f"Train: {tr.shape[0]} строк ({ordered[0]} -> {ordered[-val_days-1]})")
    print(f"Valid: {vl.shape[0]} строк ({cutoff} -> {ordered[-1]})")
    return tr, vl

train_part, valid_part = make_time_split(train, val_days=4)

X_tr = train_part[feature_cols].copy()
y_tr = train_part[TARGET].values
X_vl = valid_part[feature_cols].copy()
y_vl = valid_part[TARGET].values
vl_dates = pd.to_datetime(valid_part["assignment_date"]).dt.date.values

cb_cat_feats = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
num_feats = [c for c in feature_cols if c not in CATEGORICAL_FEATURES]
X_tr_num = train_part[num_feats].copy()
X_vl_num = valid_part[num_feats].copy()

# --- Гиперпараметры (Optuna) ---
CATBOOST_PARAMS = {
    "iterations": 2000, "learning_rate": 0.06647, "depth": 4, 
    "l2_leaf_reg": 26.277, "min_data_in_leaf": 31, "random_strength": 3.431, 
    "bagging_temperature": 0.540, "random_seed": RANDOM_STATE, 
    "verbose": 500, "eval_metric": "AUC", "auto_class_weights": "Balanced", 
    "early_stopping_rounds": 100
}

LGBM_PARAMS = {
    "n_estimators": 2000, "learning_rate": 0.13106, "max_depth": 5, 
    "num_leaves": 60, "min_child_samples": 26, "reg_alpha": 7.765, 
    "reg_lambda": 9.132, "subsample": 0.750, "colsample_bytree": 0.810, 
    "random_state": RANDOM_STATE, "verbose": -1, "n_jobs": -1
}

XGBOOST_PARAMS = {
    "n_estimators": 2000, "learning_rate": 0.04839, "max_depth": 3, 
    "min_child_weight": 10, "reg_alpha": 0.074, "reg_lambda": 0.004, 
    "subsample": 0.868, "colsample_bytree": 0.886, "gamma": 0.015, 
    "random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": -1, 
    "eval_metric": "logloss"
}

def train_catboost(X_tr, y_tr, X_vl=None, y_vl=None, cat_features=None):
    actual_cats = [c for c in (cat_features or []) if c in X_tr.columns]
    for col in actual_cats:
        X_tr[col] = X_tr[col].fillna("missing").astype(str)
        if X_vl is not None: X_vl[col] = X_vl[col].fillna("missing").astype(str)
    params = CATBOOST_PARAMS.copy()
    params["cat_features"] = actual_cats if actual_cats else None
    model = CatBoostClassifier(**params)
    eval_set = Pool(X_vl, y_vl, cat_features=actual_cats if actual_cats else None) if X_vl is not None else None
    model.fit(X_tr, y_tr, eval_set=eval_set, verbose=500)
    return model

def train_lgbm(X_tr, y_tr, X_vl=None, y_vl=None):
    pos, neg = int(y_tr.sum()), len(y_tr) - int(y_tr.sum())
    params = LGBM_PARAMS.copy()
    params["scale_pos_weight"] = neg / max(pos, 1)
    model = LGBMClassifier(**params)
    fit_params = dict(X=X_tr, y=y_tr)
    if X_vl is not None:
        fit_params["eval_set"] = [(X_vl, y_vl)]
        fit_params["callbacks"] = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
    model.fit(**fit_params)
    return model

def train_xgboost(X_tr, y_tr, X_vl=None, y_vl=None):
    pos, neg = int(y_tr.sum()), len(y_tr) - int(y_tr.sum())
    params = XGBOOST_PARAMS.copy()
    params["scale_pos_weight"] = neg / max(pos, 1)
    if X_vl is not None: params["early_stopping_rounds"] = 100
    model = XGBClassifier(**params)
    fit_params = dict(X=X_tr, y=y_tr, verbose=False)
    if X_vl is not None: fit_params["eval_set"] = [(X_vl, y_vl)]
    model.fit(**fit_params)
    return model

print("--- Обучение CatBoost ---")
cb_model = train_catboost(X_tr.copy(), y_tr, X_vl.copy(), y_vl, cb_cat_feats)
cb_scores = cb_model.predict_proba(X_vl)[:, 1]

print("\n--- Обучение LightGBM ---")
lgbm_model = train_lgbm(X_tr_num.copy(), y_tr, X_vl_num.copy(), y_vl)
lgbm_scores = lgbm_model.predict_proba(X_vl_num)[:, 1]

print("\n--- Обучение XGBoost ---")
xgb_model = train_xgboost(X_tr_num.copy(), y_tr, X_vl_num.copy(), y_vl)
xgb_scores = xgb_model.predict_proba(X_vl_num)[:, 1]

def optimize_3model_weights(scores_list, y_true, dates, n_steps=21):
    best_dap, best_w = -1.0, [1/3, 1/3, 1/3]
    for w1 in np.linspace(0, 1, n_steps):
        for w2 in np.linspace(0, 1 - w1, n_steps):
            w3 = max(1 - w1 - w2, 0)
            blended = w1 * scores_list[0] + w2 * scores_list[1] + w3 * scores_list[2]
            dap = daily_average_precision(y_true, blended, dates)
            if dap > best_dap:
                best_dap, best_w = dap, [w1, w2, w3]
    print(f"Оптимальные веса: CB={best_w[0]:.2f}, LGBM={best_w[1]:.2f}, XGB={best_w[2]:.2f}")
    print(f"Лучший Daily AP ансамбля: {best_dap:.5f}")
    return best_w

weights = optimize_3model_weights([cb_scores, lgbm_scores, xgb_scores], y_vl, vl_dates)

ens_val = weights[0]*cb_scores + weights[1]*lgbm_scores + weights[2]*xgb_scores
print("\nDaily AP breakdown:")
for date in sorted(np.unique(vl_dates)):
    mask = vl_dates == date
    ap = average_precision_score(y_vl[mask], ens_val[mask])
    print(f"  {date}: AP={ap:.5f}")
