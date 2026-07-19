"""
Feature engineering experiments against solve.py's exact validation protocol.

Every variant uses the original XGBoost parameters and the original fixed
ensemble weights (CB=0.4, LGBM=0.3, XGB=0.3). Only features are changed.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from xgboost import XGBClassifier

import solve


RANK_COLUMNS = [
    "item_views_90d", "search_views_90d", "photo_swipes_30d",
    "user_active_days_30d", "item_favorites_90d", "user_contacts_30d",
    "seller_page_views_30d", "active_days_auto_90d", "evt_total_all",
    "evt_recency_hours", "assignment_hour", "car_age_years",
    "mileage_km_log", "item_price_log",
]


def daily_ap(y, pred, dates):
    return solve.daily_average_precision(y, pred, dates)


def per_day_ap(y, pred, dates):
    return {
        str(date): daily_ap(y[dates == date], pred[dates == date], dates[dates == date])
        for date in sorted(np.unique(dates))
    }


def add_rank_features(df):
    daily = pd.DataFrame({"lead_id": df["lead_id"]})
    segment = pd.DataFrame({"lead_id": df["lead_id"]})
    for col in RANK_COLUMNS:
        if col not in df.columns:
            continue
        ascending = col != "evt_recency_hours"
        daily[f"day_rank_{col}"] = df.groupby("assignment_date")[col].rank(
            pct=True, ascending=ascending,
        )
        segment[f"day_source_rank_{col}"] = df.groupby(
            ["assignment_date", "lead_source"]
        )[col].rank(pct=True, ascending=ascending)
    return daily, segment


def build_price_affinity(df, events):
    leads = df[["lead_id", "assignment_ts", "item_price_log"]].rename(
        columns={"item_price_log": "current_price"},
    )
    event_data = events[["lead_id", "event_ts", "item_price_log"]].rename(
        columns={"item_price_log": "event_price"},
    )
    merged = leads.merge(event_data, on="lead_id", how="left")
    merged = merged[merged["event_ts"] < merged["assignment_ts"]].copy()
    merged["hours_before"] = (
        merged["assignment_ts"] - merged["event_ts"]
    ).dt.total_seconds() / 3600
    priced = merged.dropna(subset=["event_price"])

    agg = priced.groupby("lead_id")["event_price"].agg(
        price_hist_mean="mean",
        price_hist_median="median",
        price_hist_std="std",
        price_hist_min="min",
        price_hist_max="max",
        price_hist_count="count",
    )
    last = (
        priced.sort_values("event_ts").groupby("lead_id").tail(1)
        .set_index("lead_id")["event_price"].rename("price_hist_last")
    )
    recent = (
        priced[priced["hours_before"] <= 168]
        .groupby("lead_id")["event_price"].mean().rename("price_hist_mean_7d")
    )
    priced["price_close_025"] = (
        (priced["event_price"] - priced["current_price"]).abs() <= 0.25
    ).astype(float)
    close = priced.groupby("lead_id")["price_close_025"].mean().rename("price_hist_close_share")
    result = agg.join([last, recent, close], how="outer").reset_index()
    result = df[["lead_id", "item_price_log"]].merge(result, on="lead_id", how="left")
    for reference in ["mean", "median", "last", "mean_7d"]:
        hist_col = f"price_hist_{reference}"
        result[f"price_gap_{reference}"] = result["item_price_log"] - result[hist_col]
        result[f"price_abs_gap_{reference}"] = result[f"price_gap_{reference}"].abs()
    result["price_gap_z"] = result["price_gap_mean"] / result["price_hist_std"].clip(lower=0.05)
    return result.drop(columns="item_price_log")


def build_src_slot_features(df, events):
    leads = df[["lead_id", "assignment_ts"]]
    merged = leads.merge(events[["lead_id", "event_ts", "src_slot"]], on="lead_id", how="left")
    merged = merged[merged["event_ts"] < merged["assignment_ts"]].copy()
    merged["hours_before"] = (
        merged["assignment_ts"] - merged["event_ts"]
    ).dt.total_seconds() / 3600
    valid = merged.dropna(subset=["src_slot"])
    result = valid.groupby("lead_id")["src_slot"].agg(
        slot_median="median", slot_min="min", slot_max="max",
    )
    last = (
        valid.sort_values("event_ts").groupby("lead_id").tail(1)
        .set_index("lead_id")["src_slot"].rename("slot_last")
    )
    result = result.join(last, how="outer")
    for cutoff in [3, 5, 10]:
        share = (valid["src_slot"] <= cutoff).groupby(valid["lead_id"]).mean()
        result[f"slot_top{cutoff}_share"] = share
    reciprocal = (1.0 / valid["src_slot"].clip(lower=1)).groupby(valid["lead_id"]).mean()
    result["slot_mean_reciprocal"] = reciprocal
    for hours in [24, 72, 168]:
        result[f"slot_mean_{hours}h"] = (
            valid[valid["hours_before"] <= hours].groupby("lead_id")["src_slot"].mean()
        )
    result["slot_recent_vs_all"] = result["slot_mean_72h"] - result["slot_median"]
    return result.reset_index()


def build_ctx_event_crosses(df, events):
    leads = df[["lead_id", "assignment_ts"]]
    merged = leads.merge(
        events[["lead_id", "event_ts", "event_type", "ctx_seq"]],
        on="lead_id", how="left",
    )
    merged = merged[merged["event_ts"] < merged["assignment_ts"]].copy()
    merged["ctx_seq"] = merged["ctx_seq"].fillna("missing")
    merged["is_contact"] = merged["event_type"].isin(
        ["favorite", "chat_open", "call_click"],
    ).astype(int)
    merged["is_chat"] = (merged["event_type"] == "chat_open").astype(int)
    merged["is_call"] = (merged["event_type"] == "call_click").astype(int)
    grouped = merged.groupby(["lead_id", "ctx_seq"]).agg(
        total=("event_type", "size"),
        contact=("is_contact", "sum"),
        chat=("is_chat", "sum"),
        call=("is_call", "sum"),
    )
    wide = grouped.unstack(fill_value=0)
    wide.columns = [f"ctxevent_{metric}_{ctx}" for metric, ctx in wide.columns]
    result = wide.reset_index()
    for ctx in sorted(merged["ctx_seq"].unique()):
        total = f"ctxevent_total_{ctx}"
        contact = f"ctxevent_contact_{ctx}"
        if total in result and contact in result:
            result[f"ctxevent_contact_share_{ctx}"] = result[contact] / (result[total] + 1.0)
    return result


def build_consistency_features(df):
    result = pd.DataFrame({"lead_id": df["lead_id"]})
    mapping = {
        "item_view": "item_views",
        "search": "search_views",
        "favorite": "item_favorites",
        "chat_open": "chat_opens",
        "call_click": "call_clicks",
    }
    for event_name, tabular_name in mapping.items():
        for hours, window in [(24, "1d"), (72, "3d"), (168, "7d")]:
            raw_col = f"evt_{event_name}_{hours}h"
            tab_col = f"{tabular_name}_{window}"
            if raw_col not in df or tab_col not in df:
                continue
            result[f"consistency_{event_name}_{window}"] = (
                np.log1p(df[raw_col]) - np.log1p(df[tab_col])
            )
            result[f"raw_share_{event_name}_{window}"] = (
                df[raw_col] / (df[tab_col] + 1.0)
            )
    return result


def build_funnel_features(df):
    result = pd.DataFrame({"lead_id": df["lead_id"]})
    for window in ["7d", "14d", "30d", "90d"]:
        views = df[f"item_views_{window}"]
        result[f"funnel_favorite_per_view_{window}"] = df[f"item_favorites_{window}"] / (views + 5.0)
        result[f"funnel_detail_per_view_{window}"] = df[f"detail_expands_{window}"] / (views + 5.0)
        result[f"funnel_photo_per_view_{window}"] = df[f"photo_swipes_{window}"] / (views + 5.0)
        result[f"funnel_contact_per_view_{window}"] = df[f"user_contacts_{window}"] / (views + 5.0)
        result[f"funnel_seller_per_view_{window}"] = df[f"seller_page_views_{window}"] / (views + 5.0)
        result[f"funnel_call_per_chat_{window}"] = df[f"call_clicks_{window}"] / (df[f"chat_opens_{window}"] + 2.0)
    return result


def build_disjoint_features(df):
    result = pd.DataFrame({"lead_id": df["lead_id"]})
    prefixes = [
        "item_views", "search_views", "photo_swipes",
        "item_favorites", "user_contacts", "active_days_auto",
    ]
    for prefix in prefixes:
        recent3 = df[f"{prefix}_3d"] / 3.0
        old3_30 = (df[f"{prefix}_30d"] - df[f"{prefix}_3d"]).clip(lower=0) / 27.0
        recent30 = df[f"{prefix}_30d"] / 30.0
        old30_90 = (df[f"{prefix}_90d"] - df[f"{prefix}_30d"]).clip(lower=0) / 60.0
        result[f"disjoint_short_{prefix}"] = np.log1p(recent3) - np.log1p(old3_30)
        result[f"disjoint_long_{prefix}"] = np.log1p(recent30) - np.log1p(old30_90)
    return result


def build_car_features(df):
    result = pd.DataFrame({"lead_id": df["lead_id"]})
    result["car_log_mileage_per_year"] = df["mileage_km_log"] - np.log1p(df["car_age_years"])
    result["car_price_minus_mileage"] = df["item_price_log"] - df["mileage_km_log"]
    result["seller_inventory_log"] = np.log1p(df["seller_inventory_count"])
    result["seller_quality_log"] = (
        df["seller_response_rate_30d"] * np.log1p(df["seller_inventory_count"])
    )
    result["day_segment_price_rank"] = df.groupby(
        ["assignment_date", "car_segment"]
    )["item_price_log"].rank(pct=True)
    result["day_segment_mileage_rank"] = df.groupby(
        ["assignment_date", "car_segment"]
    )["mileage_km_log"].rank(pct=True)
    return result


def main():
    np.random.seed(solve.RANDOM_STATE)
    train, _, events = solve.load_data()
    base_events = solve.build_event_features(train, events)
    train = train.merge(base_events, on="lead_id", how="left")
    train = solve.build_tabular_features(train)

    features = solve.select_features(solve.get_feature_columns(train))
    cats = [c for c in solve.CATEGORICAL_FEATURES if c in features]
    nums = [c for c in features if c not in cats]
    train_part, valid_part = solve.make_time_split(train, val_days=4)
    train_mask = train.index.isin(train_part.index)
    valid_mask = train.index.isin(valid_part.index)
    y_tr = train.loc[train_mask, solve.TARGET].to_numpy()
    y_vl = train.loc[valid_mask, solve.TARGET].to_numpy()
    dates = pd.to_datetime(train.loc[valid_mask, "assignment_date"]).dt.date.to_numpy()

    # Baseline predictions, with exactly solve.py's hyperparameters.
    X_cb_tr = train.loc[train_mask, features].copy()
    X_cb_vl = train.loc[valid_mask, features].copy()
    for col in cats:
        X_cb_tr[col] = X_cb_tr[col].fillna("missing").astype(str)
        X_cb_vl[col] = X_cb_vl[col].fillna("missing").astype(str)
    cb_params = solve.CATBOOST_PARAMS.copy()
    cb_params.update(cat_features=cats, verbose=0)
    cb = CatBoostClassifier(**cb_params)
    cb.fit(X_cb_tr, y_tr, eval_set=(X_cb_vl, y_vl), verbose=False)
    cb_pred = cb.predict_proba(X_cb_vl)[:, 1]

    X_tr = train.loc[train_mask, nums].copy()
    X_vl = train.loc[valid_mask, nums].copy()
    pos = int(y_tr.sum())
    scale = (len(y_tr) - pos) / pos
    lgb_params = solve.LGBM_PARAMS.copy()
    lgb_params["scale_pos_weight"] = scale
    lgb_model = LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_tr, y_tr, eval_set=[(X_vl, y_vl)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    lgb_pred = lgb_model.predict_proba(X_vl)[:, 1]

    def train_xgb(frame, added_columns):
        columns = nums + added_columns
        xtr = frame.loc[train_mask, columns].replace([np.inf, -np.inf], np.nan)
        xvl = frame.loc[valid_mask, columns].replace([np.inf, -np.inf], np.nan)
        params = solve.XGBOOST_PARAMS.copy()
        params.update(scale_pos_weight=scale, early_stopping_rounds=100)
        model = XGBClassifier(**params)
        model.fit(xtr, y_tr, eval_set=[(xvl, y_vl)], verbose=False)
        return model.predict_proba(xvl)[:, 1], model.best_iteration + 1

    xgb_base, base_iteration = train_xgb(train, [])
    baseline = 0.4 * cb_pred + 0.3 * lgb_pred + 0.3 * xgb_base
    baseline_score = daily_ap(y_vl, baseline, dates)
    print(f"BASELINE: {baseline_score:.6f}; XGB iter={base_iteration}; days={per_day_ap(y_vl, baseline, dates)}")

    print("Building candidate feature groups...")
    daily_rank, source_rank = add_rank_features(train)
    groups = {
        "daily_rank": daily_rank,
        "source_rank": source_rank,
        "price_affinity": build_price_affinity(train, events),
        "src_slot": build_src_slot_features(train, events),
        "ctx_event_cross": build_ctx_event_crosses(train, events),
        "event_consistency": build_consistency_features(train),
        "funnel": build_funnel_features(train),
        "disjoint": build_disjoint_features(train),
        "car_economics": build_car_features(train),
    }

    rows = []
    saved_predictions = {
        "y": y_vl,
        "dates": dates.astype(str),
        "cb": cb_pred,
        "lgb": lgb_pred,
        "xgb_base": xgb_base,
    }
    for name, group in groups.items():
        candidate = train.merge(group, on="lead_id", how="left", suffixes=("", "_candidate"))
        added = [c for c in group.columns if c != "lead_id" and c not in nums]
        pred, iteration = train_xgb(candidate, added)
        individual = daily_ap(y_vl, pred, dates)
        fixed_blend = 0.4 * cb_pred + 0.3 * lgb_pred + 0.3 * pred
        blend_score = daily_ap(y_vl, fixed_blend, dates)
        days = per_day_ap(y_vl, fixed_blend, dates)
        rows.append({
            "feature_group": name,
            "n_added": len(added),
            "xgb_ap": individual,
            "fixed_blend_ap": blend_score,
            "delta_vs_baseline": blend_score - baseline_score,
            "xgb_iteration": iteration,
            **days,
        })
        saved_predictions[name] = pred
        print(
            f"{name:<20} +{len(added):>2} XGB={individual:.6f} "
            f"blend={blend_score:.6f} delta={blend_score-baseline_score:+.6f} "
            f"days={days}"
        )

    results = pd.DataFrame(rows).sort_values("fixed_blend_ap", ascending=False)
    results.to_csv("feature_experiment_results.csv", index=False)
    np.savez_compressed("feature_experiment_predictions.npz", **saved_predictions)
    print("\nRESULTS")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
