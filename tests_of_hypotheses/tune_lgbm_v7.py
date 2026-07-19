"""Тюнинг двух потоков LightGBM для уже найденного ансамбля v7.

Скрипт использует сохраненные предсказания лучших CatBoost/XGBoost из
``tune_final_v7.py``. Поэтому повторно обучаются только быстрые LightGBM.
Исследования Optuna сохраняются в SQLite и могут быть продолжены.
"""

import json
import warnings

warnings.filterwarnings("ignore")

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from optuna.trial import TrialState

import solve
from improvement_experiments import per_day_ap
from tune_final_v7 import build_data, optimize_simplex


N_TRIALS_BASE = 20
N_TRIALS_ENHANCED = 20
REFERENCE_WEIGHTS = np.array([0.30, 0.20, 0.15, 0.35])


def make_daily_ap_metric(dates):
    """Метрика ранней остановки, совпадающая с локальной целевой метрикой."""

    def daily_ap_metric(y_true, y_pred):
        value = solve.daily_average_precision(y_true, y_pred, dates)
        return "daily_ap", value, True

    return daily_ap_metric


def suggest_lgbm_params(trial):
    """Умеренное пространство поиска вокруг исходной сильной конфигурации."""

    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.025, 0.16, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "num_leaves": trial.suggest_int("num_leaves", 16, 96),
        "min_child_samples": trial.suggest_int("min_child_samples", 12, 75),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 0.20, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.10, 20.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.10, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.70, 1.0),
        "subsample_freq": trial.suggest_categorical("subsample_freq", [0, 1]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.25),
        "max_bin": trial.suggest_categorical("max_bin", [127, 255, 511]),
    }


def original_trial_params():
    """Исходные параметры как контрольная точка внутри каждого study."""

    params = solve.LGBM_PARAMS
    return {
        "learning_rate": float(params["learning_rate"]),
        "max_depth": int(params["max_depth"]),
        "num_leaves": int(params["num_leaves"]),
        "min_child_samples": int(params["min_child_samples"]),
        "min_child_weight": 1e-3,
        "reg_alpha": float(params["reg_alpha"]),
        "reg_lambda": float(params["reg_lambda"]),
        "subsample": float(params["subsample"]),
        "subsample_freq": 0,
        "colsample_bytree": float(params["colsample_bytree"]),
        "min_split_gain": 0.0,
        "max_bin": 255,
    }


def model_params(params, scale, n_estimators=2500):
    """Добавляет общие технические параметры LightGBM."""

    return {
        **params,
        "n_estimators": int(n_estimators),
        "scale_pos_weight": scale,
        "random_state": solve.RANDOM_STATE,
        "verbose": -1,
        "n_jobs": -1,
        # Отключаем стандартный logloss: early stopping идет по Daily AP.
        "metric": "None",
    }


def fit_with_early_stopping(params, X_train, y_train, X_valid, y_valid, dates, scale):
    model = LGBMClassifier(**model_params(params, scale))
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=make_daily_ap_metric(dates),
        callbacks=[
            lgb.early_stopping(150, first_metric_only=True, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return model


def fit_exact_iterations(params, iterations, X_train, y_train, scale):
    """Воспроизводит лучшую итерацию trial без повторной ранней остановки."""

    model = LGBMClassifier(**model_params(params, scale, n_estimators=iterations))
    model.fit(X_train, y_train)
    return model


def rank_predictions_by_day(predictions, dates):
    """Приводит шкалы моделей к дневным процентильным рангам."""

    ranked = np.empty_like(predictions, dtype=float)
    for day in np.unique(dates):
        mask = dates == day
        frame = pd.DataFrame(predictions[mask])
        ranked[mask] = frame.rank(method="average", pct=True).to_numpy()
    return ranked


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    with open("tuning_final_result.json", encoding="utf-8") as file:
        previous_result = json.load(file)
    cached = np.load("tuned_validation_predictions.npz")

    train, base_features, cats, base_nums, added = build_data()
    del base_features, cats
    enhanced_nums = base_nums + added
    train_part, valid_part = solve.make_time_split(train, val_days=4)

    y_train = train_part[solve.TARGET].to_numpy()
    y_valid = valid_part[solve.TARGET].to_numpy()
    dates = pd.to_datetime(valid_part["assignment_date"]).dt.date.to_numpy()
    date_strings = np.asarray([str(value) for value in dates])

    # Защита от случайного смешивания предсказаний от другого разбиения.
    if not np.array_equal(y_valid, cached["y"]):
        raise RuntimeError("Target в кэше не совпадает с текущим validation split")
    if not np.array_equal(date_strings, cached["dates"].astype(str)):
        raise RuntimeError("Даты в кэше не совпадают с текущим validation split")

    fixed_cb = cached["catboost"]
    original_base = cached["lgbm_base"]
    original_enhanced = cached["lgbm_enhanced"]
    fixed_xgb = cached["xgboost"]
    fixed_matrix = np.column_stack([fixed_cb, original_base, original_enhanced, fixed_xgb])
    reference_pred = fixed_matrix @ REFERENCE_WEIGHTS
    reference_score = solve.daily_average_precision(y_valid, reference_pred, dates)

    positives = int(y_train.sum())
    scale = (len(y_train) - positives) / positives
    X_base_train = train_part[base_nums]
    X_base_valid = valid_part[base_nums]
    X_enh_train = train_part[enhanced_nums]
    X_enh_valid = valid_part[enhanced_nums]

    print(f"Reference tuned ensemble: {reference_score:.6f}")
    print(f"Tuning base LightGBM ({N_TRIALS_BASE} persistent trials)...")

    def base_objective(trial):
        params = suggest_lgbm_params(trial)
        model = fit_with_early_stopping(
            params, X_base_train, y_train, X_base_valid, y_valid, dates, scale,
        )
        pred = model.predict_proba(X_base_valid)[:, 1]
        blend = (
            REFERENCE_WEIGHTS[0] * fixed_cb
            + REFERENCE_WEIGHTS[1] * pred
            + REFERENCE_WEIGHTS[2] * original_enhanced
            + REFERENCE_WEIGHTS[3] * fixed_xgb
        )
        score = solve.daily_average_precision(y_valid, blend, dates)
        trial.set_user_attr("best_iteration", int(model.best_iteration_))
        trial.set_user_attr("days", {
            key: float(value) for key, value in per_day_ap(y_valid, blend, dates).items()
        })
        print(f"LGB-base trial {trial.number:02d}: {score:.6f} iter={model.best_iteration_}")
        return score

    base_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=44),
        storage="sqlite:///tuning_lgbm_base_v7.db",
        study_name="lgbm_base_v7",
        load_if_exists=True,
    )
    if not any(trial.state == TrialState.COMPLETE for trial in base_study.trials):
        base_study.enqueue_trial(original_trial_params())
    base_study.optimize(base_objective, n_trials=N_TRIALS_BASE)

    print(f"Tuning enhanced LightGBM ({N_TRIALS_ENHANCED} persistent trials)...")

    def enhanced_objective(trial):
        params = suggest_lgbm_params(trial)
        model = fit_with_early_stopping(
            params, X_enh_train, y_train, X_enh_valid, y_valid, dates, scale,
        )
        pred = model.predict_proba(X_enh_valid)[:, 1]
        blend = (
            REFERENCE_WEIGHTS[0] * fixed_cb
            + REFERENCE_WEIGHTS[1] * original_base
            + REFERENCE_WEIGHTS[2] * pred
            + REFERENCE_WEIGHTS[3] * fixed_xgb
        )
        score = solve.daily_average_precision(y_valid, blend, dates)
        trial.set_user_attr("best_iteration", int(model.best_iteration_))
        trial.set_user_attr("days", {
            key: float(value) for key, value in per_day_ap(y_valid, blend, dates).items()
        })
        print(f"LGB-enh trial {trial.number:02d}: {score:.6f} iter={model.best_iteration_}")
        return score

    enhanced_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=45),
        storage="sqlite:///tuning_lgbm_enhanced_v7.db",
        study_name="lgbm_enhanced_v7",
        load_if_exists=True,
    )
    if not any(trial.state == TrialState.COMPLETE for trial in enhanced_study.trials):
        enhanced_study.enqueue_trial(original_trial_params())
    enhanced_study.optimize(enhanced_objective, n_trials=N_TRIALS_ENHANCED)

    best_base_params = dict(base_study.best_params)
    best_base_iterations = int(base_study.best_trial.user_attrs["best_iteration"])
    best_base_model = fit_exact_iterations(
        best_base_params, best_base_iterations, X_base_train, y_train, scale,
    )
    tuned_base = best_base_model.predict_proba(X_base_valid)[:, 1]

    best_enhanced_params = dict(enhanced_study.best_params)
    best_enhanced_iterations = int(enhanced_study.best_trial.user_attrs["best_iteration"])
    best_enhanced_model = fit_exact_iterations(
        best_enhanced_params, best_enhanced_iterations, X_enh_train, y_train, scale,
    )
    tuned_enhanced = best_enhanced_model.predict_proba(X_enh_valid)[:, 1]

    # Проверяем варианты независимо: новый поток может оказаться полезен только один.
    candidates = {
        "original_both": np.column_stack(
            [fixed_cb, original_base, original_enhanced, fixed_xgb]
        ),
        "tuned_base": np.column_stack(
            [fixed_cb, tuned_base, original_enhanced, fixed_xgb]
        ),
        "tuned_enhanced": np.column_stack(
            [fixed_cb, original_base, tuned_enhanced, fixed_xgb]
        ),
        "tuned_both": np.column_stack(
            [fixed_cb, tuned_base, tuned_enhanced, fixed_xgb]
        ),
    }

    best = None
    print("Optimizing final weights (step=0.025)...")
    for candidate_name, raw_matrix in candidates.items():
        for blend_mode, matrix in (
            ("raw", raw_matrix),
            ("per_day_rank", rank_predictions_by_day(raw_matrix, dates)),
        ):
            score, weights = optimize_simplex(matrix, y_valid, dates, steps=40)
            print(f"{candidate_name:16s} {blend_mode:12s}: {score:.6f} weights={weights}")
            if best is None or score > best["score"]:
                best = {
                    "score": float(score),
                    "candidate": candidate_name,
                    "blend_mode": blend_mode,
                    "weights": weights,
                    "matrix": matrix,
                    "raw_matrix": raw_matrix,
                }

    final_pred = best["matrix"] @ best["weights"]
    final_days = {
        key: float(value) for key, value in per_day_ap(y_valid, final_pred, dates).items()
    }
    print(
        f"FINAL: {best['score']:.6f}, candidate={best['candidate']}, "
        f"mode={best['blend_mode']}, weights={best['weights']}, days={final_days}"
    )

    base_study.trials_dataframe().to_csv("tuning_lgbm_base_trials.csv", index=False)
    enhanced_study.trials_dataframe().to_csv("tuning_lgbm_enhanced_trials.csv", index=False)

    result = {
        "reference_score": float(reference_score),
        "validation_score": best["score"],
        "validation_days": final_days,
        "candidate": best["candidate"],
        "blend_mode": best["blend_mode"],
        "model_order": ["catboost", "lgbm_base", "lgbm_enhanced", "xgboost"],
        "weights": [float(value) for value in best["weights"]],
        "lgbm_base_search_score": float(base_study.best_value),
        "lgbm_base_params": best_base_params,
        "lgbm_base_iterations": best_base_iterations,
        "lgbm_enhanced_search_score": float(enhanced_study.best_value),
        "lgbm_enhanced_params": best_enhanced_params,
        "lgbm_enhanced_iterations": best_enhanced_iterations,
        "catboost_params": previous_result["catboost_params"],
        "catboost_iterations": previous_result["catboost_iterations"],
        "xgboost_params": previous_result["xgboost_params"],
        "xgboost_iterations": previous_result["xgboost_iterations"],
    }
    with open("tuning_lgbm_final_result.json", "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    np.savez_compressed(
        "tuned_lgbm_validation_predictions.npz",
        y=y_valid,
        dates=date_strings,
        catboost=fixed_cb,
        lgbm_base_original=original_base,
        lgbm_base_tuned=tuned_base,
        lgbm_enhanced_original=original_enhanced,
        lgbm_enhanced_tuned=tuned_enhanced,
        xgboost=fixed_xgb,
        final=final_pred,
    )


if __name__ == "__main__":
    main()
