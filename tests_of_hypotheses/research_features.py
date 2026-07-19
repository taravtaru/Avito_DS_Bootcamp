"""
Feature Importance & Noise Research
=====================================

Исследуем:
  1. Feature importance (CatBoost gain, LightGBM gain).
  2. Noise test: добавляем случайные признаки, чтобы определить порог "шума".
  3. Permutation importance по Daily AP.
  4. Корреляции фичей с таргетом.
  5. Группировка: какие группы признаков наиболее важны.

Результат — рекомендации по удалению/сохранению фичей.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import average_precision_score

# ── Повторно используем функции из solve.py ──
from solve import (
    load_data,
    build_event_features,
    build_tabular_features,
    target_encode_oof,
    daily_average_precision,
    make_time_split,
    get_feature_columns,
    RANDOM_STATE, TARGET, CATEGORICAL_FEATURES, NON_FEATURE_COLUMNS,
)

np.random.seed(RANDOM_STATE)


# ──────────────────────────────────────────────────────────────────────
# Подготовка данных (повторяет pipeline из solve.py)
# ──────────────────────────────────────────────────────────────────────

def prepare_data():
    """Подготавливает данные: загрузка + feature engineering."""
    print("=" * 70)
    print("FEATURE IMPORTANCE & NOISE RESEARCH")
    print("=" * 70)

    train, test, events = load_data()

    # Event features.
    train_events = build_event_features(train, events)
    test_events = build_event_features(test, events)
    train = train.merge(train_events, on="lead_id", how="left")
    test = test.merge(test_events, on="lead_id", how="left")

    # Tabular features.
    train = build_tabular_features(train)
    test = build_tabular_features(test)

    # Target encoding.
    train["cat_source_x_center"] = train["lead_source"] + "_" + train["call_center"]
    test["cat_source_x_center"] = test["lead_source"] + "_" + test["call_center"]
    train["cat_region_x_segment"] = train["region"] + "_" + train["car_segment"]
    test["cat_region_x_segment"] = test["region"] + "_" + test["car_segment"]
    train["cat_channel_x_tenure"] = train["lead_channel"] + "_" + train["user_tenure_bucket"]
    test["cat_channel_x_tenure"] = test["lead_channel"] + "_" + test["user_tenure_bucket"]

    te_columns = CATEGORICAL_FEATURES + [
        "cat_source_x_center", "cat_region_x_segment", "cat_channel_x_tenure",
    ]
    train, test = target_encode_oof(train, test, te_columns)

    return train, test


def get_feature_lists(feature_cols):
    """Классифицирует фичи по группам для анализа."""
    groups = {
        "event_features": [c for c in feature_cols if c.startswith("evt_")],
        "ratio_features": [c for c in feature_cols if c.startswith("ratio_")],
        "trend_features": [c for c in feature_cols if c.startswith("trend_")],
        "relative_features": [c for c in feature_cols if c.startswith("rel_")],
        "interaction_features": [c for c in feature_cols if c.startswith("interact_")],
        "target_encoding": [c for c in feature_cols if c.startswith("te_")],
        "aggregate_totals": [c for c in feature_cols if c.startswith("total_")],
        "categorical_raw": [c for c in feature_cols if c in CATEGORICAL_FEATURES],
        "original_numeric": [],  # Заполним ниже.
    }

    # Все, что не попало в другие группы.
    assigned = set()
    for g_cols in groups.values():
        assigned.update(g_cols)
    groups["original_numeric"] = [c for c in feature_cols if c not in assigned]

    return groups


# ──────────────────────────────────────────────────────────────────────
# 1. Noise Test — добавляем шумовые признаки
# ──────────────────────────────────────────────────────────────────────

def noise_test(X_train, y_train, X_val, y_val, n_noise=5):
    """
    Добавляет n_noise случайных (шумовых) признаков к данным.
    Обучает LightGBM и сравнивает важности реальных фичей с шумовыми.

    Фичи с важностью ниже максимума шумовых — кандидаты на удаление.
    """
    print("\n" + "=" * 70)
    print("NOISE TEST")
    print("=" * 70)

    X_tr = X_train.copy()
    X_vl = X_val.copy()

    noise_cols = []
    for i in range(n_noise):
        col_name = f"__noise_{i}__"
        noise_cols.append(col_name)
        X_tr[col_name] = np.random.randn(len(X_tr))
        X_vl[col_name] = np.random.randn(len(X_vl))

    # Обучаем LightGBM (без категориальных — чтобы все фичи были числовые).
    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=7,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=-1,
    )
    model.fit(
        X_tr, y_train,
        eval_set=[(X_vl, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    # Важности.
    importance = pd.Series(
        model.feature_importances_,
        index=X_tr.columns,
    ).sort_values(ascending=False)

    # Порог шума.
    noise_importances = importance[noise_cols]
    noise_threshold = noise_importances.max()
    noise_mean = noise_importances.mean()

    print(f"\nNoise features importance: {noise_importances.values}")
    print(f"Noise threshold (max): {noise_threshold}")
    print(f"Noise mean: {noise_mean:.1f}")

    # Фичи ниже порога шума.
    below_noise = importance[
        (importance <= noise_threshold) & (~importance.index.isin(noise_cols))
    ]

    print(f"\nFeatures BELOW noise threshold ({len(below_noise)}):")
    for feat, imp in below_noise.items():
        print(f"  {feat}: {imp}")

    # Фичи выше порога.
    above_noise = importance[
        (importance > noise_threshold) & (~importance.index.isin(noise_cols))
    ]
    print(f"\nFeatures ABOVE noise threshold ({len(above_noise)})")

    return importance, noise_threshold, below_noise.index.tolist()


# ──────────────────────────────────────────────────────────────────────
# 2. Model Feature Importance (CatBoost + LightGBM)
# ──────────────────────────────────────────────────────────────────────

def model_importance(X_train, y_train, X_val, y_val, feature_cols):
    """Обучает CatBoost и LightGBM, извлекает feature importance."""
    print("\n" + "=" * 70)
    print("MODEL FEATURE IMPORTANCE")
    print("=" * 70)

    # ── CatBoost ──
    cat_features = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    X_tr_cb = X_train[feature_cols].copy()
    X_vl_cb = X_val[feature_cols].copy()
    for col in cat_features:
        X_tr_cb[col] = X_tr_cb[col].fillna("missing").astype(str)
        X_vl_cb[col] = X_vl_cb[col].fillna("missing").astype(str)

    cb = CatBoostClassifier(
        iterations=1000, learning_rate=0.03, depth=7, l2_leaf_reg=5,
        random_seed=RANDOM_STATE, verbose=0, auto_class_weights="Balanced",
        cat_features=cat_features, early_stopping_rounds=100,
    )
    cb.fit(X_tr_cb, y_train,
           eval_set=Pool(X_vl_cb, y_val, cat_features=cat_features))

    cb_importance = pd.Series(
        cb.get_feature_importance(), index=feature_cols,
    ).sort_values(ascending=False)

    print("\n--- CatBoost Top-30 ---")
    for i, (feat, imp) in enumerate(cb_importance.head(30).items()):
        print(f"  {i+1:2d}. {feat:50s} {imp:.2f}")

    print("\n--- CatBoost Bottom-20 ---")
    for feat, imp in cb_importance.tail(20).items():
        print(f"      {feat:50s} {imp:.4f}")

    # ── LightGBM ──
    lgbm_features = [c for c in feature_cols if c not in CATEGORICAL_FEATURES]
    X_tr_lgbm = X_train[lgbm_features].copy()
    X_vl_lgbm = X_val[lgbm_features].copy()

    lgbm = LGBMClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=7, num_leaves=63,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
    )
    lgbm.fit(
        X_tr_lgbm, y_train,
        eval_set=[(X_vl_lgbm, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    lgbm_importance = pd.Series(
        lgbm.feature_importances_, index=lgbm_features,
    ).sort_values(ascending=False)

    print("\n--- LightGBM Top-30 ---")
    for i, (feat, imp) in enumerate(lgbm_importance.head(30).items()):
        print(f"  {i+1:2d}. {feat:50s} {imp}")

    print("\n--- LightGBM Bottom-20 ---")
    for feat, imp in lgbm_importance.tail(20).items():
        print(f"      {feat:50s} {imp}")

    return cb_importance, lgbm_importance


# ──────────────────────────────────────────────────────────────────────
# 3. Permutation Importance по Daily AP
# ──────────────────────────────────────────────────────────────────────

def permutation_importance_daily_ap(
    model, X_val, y_val, dates, feature_cols, n_repeats=3, top_k=40,
):
    """
    Permutation importance: перемешиваем каждый признак по одному
    и замеряем падение Daily AP.

    Более надежно, чем gain-based importance.
    """
    print("\n" + "=" * 70)
    print("PERMUTATION IMPORTANCE (by Daily AP)")
    print("=" * 70)

    baseline_scores = model.predict_proba(X_val)[:, 1]
    baseline_dap = daily_average_precision(y_val, baseline_scores, dates)
    print(f"Baseline Daily AP: {baseline_dap:.5f}")

    importances = {}

    for col in feature_cols:
        drops = []
        for r in range(n_repeats):
            X_shuffled = X_val.copy()
            X_shuffled[col] = np.random.permutation(X_shuffled[col].values)
            shuffled_scores = model.predict_proba(X_shuffled)[:, 1]
            shuffled_dap = daily_average_precision(y_val, shuffled_scores, dates)
            drops.append(baseline_dap - shuffled_dap)

        importances[col] = {
            "mean_drop": np.mean(drops),
            "std_drop": np.std(drops),
        }

    # Сортируем по среднему падению (больше = важнее).
    perm_imp = pd.DataFrame(importances).T.sort_values("mean_drop", ascending=False)

    print(f"\nTop-{top_k} by permutation importance (Daily AP drop):")
    for i, (feat, row) in enumerate(perm_imp.head(top_k).iterrows()):
        marker = " ***" if row["mean_drop"] > 0.005 else ""
        print(f"  {i+1:2d}. {feat:50s} drop={row['mean_drop']:+.5f} +/- {row['std_drop']:.5f}{marker}")

    # Фичи с отрицательным влиянием (модель лучше без них!).
    negative = perm_imp[perm_imp["mean_drop"] < 0]
    if len(negative) > 0:
        print(f"\nFeatures with NEGATIVE importance ({len(negative)} -- model is BETTER without them):")
        for feat, row in negative.iterrows():
            print(f"  {feat:50s} drop={row['mean_drop']:+.5f}")

    return perm_imp


# ──────────────────────────────────────────────────────────────────────
# 4. Корреляция с таргетом
# ──────────────────────────────────────────────────────────────────────

def target_correlations(X_train, y_train, feature_cols):
    """Считает корреляцию каждого признака с таргетом."""
    print("\n" + "=" * 70)
    print("FEATURE-TARGET CORRELATIONS")
    print("=" * 70)

    numeric_cols = [
        c for c in feature_cols
        if c not in CATEGORICAL_FEATURES and pd.api.types.is_numeric_dtype(X_train[c])
    ]

    correlations = X_train[numeric_cols].corrwith(
        pd.Series(y_train, index=X_train.index)
    ).sort_values(key=abs, ascending=False)

    print("\nTop-30 correlated with target:")
    for i, (feat, corr) in enumerate(correlations.head(30).items()):
        print(f"  {i+1:2d}. {feat:50s} corr={corr:+.4f}")

    print("\nBottom-20 (weakest correlation):")
    for feat, corr in correlations.tail(20).items():
        print(f"      {feat:50s} corr={corr:+.4f}")

    return correlations


# ──────────────────────────────────────────────────────────────────────
# 5. Группировой анализ
# ──────────────────────────────────────────────────────────────────────

def group_analysis(importance_series, feature_cols):
    """Анализирует средний importance по группам фичей."""
    print("\n" + "=" * 70)
    print("GROUP IMPORTANCE ANALYSIS")
    print("=" * 70)

    groups = get_feature_lists(feature_cols)

    print(f"\n{'Group':<30s} {'Count':>5s} {'Mean Imp':>10s} {'Sum Imp':>10s} {'Top Feature':<40s}")
    print("-" * 100)

    for group_name, group_cols in groups.items():
        valid_cols = [c for c in group_cols if c in importance_series.index]
        if not valid_cols:
            continue
        mean_imp = importance_series[valid_cols].mean()
        sum_imp = importance_series[valid_cols].sum()
        top_feat = importance_series[valid_cols].idxmax()
        print(f"  {group_name:<28s} {len(valid_cols):>5d} {mean_imp:>10.2f} {sum_imp:>10.2f} {top_feat:<40s}")


# ──────────────────────────────────────────────────────────────────────
# 6. Межфичовые корреляции (детекция мультиколлинеарности)
# ──────────────────────────────────────────────────────────────────────

def multicollinearity_check(X_train, feature_cols, threshold=0.95):
    """Находит пары фичей с корреляцией > threshold."""
    print("\n" + "=" * 70)
    print(f"MULTICOLLINEARITY CHECK (threshold={threshold})")
    print("=" * 70)

    numeric_cols = [
        c for c in feature_cols
        if c not in CATEGORICAL_FEATURES and pd.api.types.is_numeric_dtype(X_train[c])
    ]

    corr_matrix = X_train[numeric_cols].corr().abs()

    # Верхний треугольник.
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    high_corr_pairs = []
    for col in upper.columns:
        for idx in upper.index:
            val = upper.loc[idx, col]
            if pd.notna(val) and val > threshold:
                high_corr_pairs.append((idx, col, val))

    high_corr_pairs.sort(key=lambda x: -x[2])

    print(f"\nHighly correlated pairs (>{threshold}): {len(high_corr_pairs)}")
    for f1, f2, corr in high_corr_pairs[:30]:
        print(f"  {f1:45s} <-> {f2:45s}  r={corr:.4f}")

    if len(high_corr_pairs) > 30:
        print(f"  ... and {len(high_corr_pairs) - 30} more pairs")

    return high_corr_pairs


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    # Подготовка данных.
    train, test = prepare_data()

    # Time split.
    train_part, valid_part = make_time_split(train, val_days=4)

    # Feature columns.
    feature_cols = get_feature_columns(train)
    cat_combo_cols = ["cat_source_x_center", "cat_region_x_segment", "cat_channel_x_tenure"]
    feature_cols = [c for c in feature_cols if c not in cat_combo_cols]

    X_train = train_part[feature_cols].copy()
    y_train = train_part[TARGET].values
    X_val = valid_part[feature_cols].copy()
    y_val = valid_part[TARGET].values
    val_dates = pd.to_datetime(valid_part["assignment_date"]).dt.date.values

    # ── 1. Model Importance ──
    cb_imp, lgbm_imp = model_importance(X_train, y_train, X_val, y_val, feature_cols)

    # ── 2. Group Analysis (на основе CatBoost importance) ──
    group_analysis(cb_imp, feature_cols)

    # ── 3. Noise Test ──
    lgbm_features = [c for c in feature_cols if c not in CATEGORICAL_FEATURES]
    X_train_lgbm = X_train[lgbm_features].copy()
    X_val_lgbm = X_val[lgbm_features].copy()
    _, noise_threshold, noisy_features = noise_test(
        X_train_lgbm, y_train, X_val_lgbm, y_val, n_noise=5,
    )

    # ── 4. Корреляции ──
    correlations = target_correlations(X_train, y_train, feature_cols)

    # ── 5. Мультиколлинеарность ──
    high_corr_pairs = multicollinearity_check(X_train, feature_cols, threshold=0.95)

    # ── 6. Permutation Importance (LightGBM, т.к. быстрее) ──
    print("\nTraining LightGBM for permutation importance...")
    lgbm_for_perm = LGBMClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=7, num_leaves=63,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
    )
    lgbm_for_perm.fit(
        X_val_lgbm.copy(), y_val,  # Нет, учим на train, предсказываем на val.
    )
    # Переобучаем на train.
    lgbm_for_perm.fit(
        X_train_lgbm, y_train,
        eval_set=[(X_val_lgbm, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    perm_imp = permutation_importance_daily_ap(
        lgbm_for_perm, X_val_lgbm, y_val, val_dates, lgbm_features,
        n_repeats=3, top_k=40,
    )

    # ── SUMMARY ──
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    print(f"\nTotal features: {len(feature_cols)}")
    print(f"Features below noise: {len(noisy_features)}")

    # Фичи с негативным permutation importance.
    negative_perm = perm_imp[perm_imp["mean_drop"] < -0.001].index.tolist()
    print(f"Features with negative permutation importance: {len(negative_perm)}")
    for f in negative_perm:
        print(f"  - {f}")

    print(f"Highly correlated pairs (>0.95): {len(high_corr_pairs)}")

    # Рекомендация: объединяем noisy + negative perm.
    candidates_to_drop = set(noisy_features) | set(negative_perm)
    print(f"\nCandidate features to DROP: {len(candidates_to_drop)}")
    for f in sorted(candidates_to_drop):
        print(f"  - {f}")

    # Топ-20 фичей по permutation importance.
    print(f"\nTop-20 MOST IMPORTANT features (permutation):")
    for i, (feat, row) in enumerate(perm_imp.head(20).iterrows()):
        print(f"  {i+1:2d}. {feat}")


if __name__ == "__main__":
    main()
