
import os
import json
import copy
import traceback
import importlib.util
from datetime import datetime

import numpy as np
import pandas as pd
import optuna

BASE_SCRIPT_PATH = "train_tcga.py"
NPZ_PATH = "../../datasets/tcga/tcga.npz"
TCGA_DATA_DIR = "../../datasets/tcga"

OUT_DIR = "overnight_optuna_true_mise"
os.makedirs(OUT_DIR, exist_ok=True)

N_TRIALS = 40
TUNE_RUNS_PER_TRIAL = 3
FINAL_RUNS = 5
OPTUNA_TIMEOUT_SEC = None
OPTUNA_N_JOBS = 1
STUDY_NAME = "tcga_artemis_true_mise"
STUDY_DB = os.path.join(OUT_DIR, "optuna_study.sqlite3")

DOSE_GRID_SIZE = 65
BENCHMARK_BATCH_SIZE = 32
BENCHMARK_SEED = 909
BENCHMARK_ASSIGNMENT_BIAS = 10.0

RUN_COMPONENT_ABLATION = True
RUN_PAIRING_ABLATION = True
RUN_SENSITIVITY = False


def load_module_from_path(py_path: str, module_name: str = "artemis_tcga_base"):
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = load_module_from_path(BASE_SCRIPT_PATH)
mod.NPZ_PATH = NPZ_PATH
mod.TCGA_DATA_DIR = TCGA_DATA_DIR
mod.OUT_DIR = OUT_DIR
mod.SAVE_RESULTS = True


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def trial_to_row(trial):
    row = {
        "number": trial.number,
        "value": trial.value,
        "state": str(trial.state),
        "datetime_start": str(trial.datetime_start),
        "datetime_complete": str(trial.datetime_complete),
    }
    for k, v in trial.params.items():
        row[f"param_{k}"] = v
    for k, v in trial.user_attrs.items():
        row[f"user_{k}"] = v
    return row


def save_study_csv(study, out_csv):
    rows = [trial_to_row(t) for t in study.trials]
    pd.DataFrame(rows).to_csv(out_csv, index=False, sep=";")


def build_search_space(trial, base_params):
    p = copy.deepcopy(base_params)

    p["lr"] = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
    p["main_weight_decay"] = trial.suggest_float("main_weight_decay", 1e-6, 5e-4, log=True)
    p["lr_treat_clf"] = trial.suggest_float("lr_treat_clf", 1e-4, 1e-3, log=True)
    p["clf_weight_decay"] = trial.suggest_float("clf_weight_decay", 1e-7, 1e-4, log=True)

    p["batch_size"] = trial.suggest_categorical("batch_size", [128, 256])
    p["latent_dim"] = trial.suggest_categorical("latent_dim", [64, 128])
    p["hidden_dim"] = trial.suggest_categorical("hidden_dim", [128, 256, 384])

    p["alpha"] = trial.suggest_float("alpha", 0.01, 0.15)
    p["margin"] = trial.suggest_float("margin", 0.2, 1.0)
    p["warmup_epochs"] = trial.suggest_categorical("warmup_epochs", [5, 10, 20, 30])
    p["pair_update_freq"] = trial.suggest_categorical("pair_update_freq", [3, 5, 10])
    p["pair_threshold_percentile"] = trial.suggest_categorical("pair_threshold_percentile", [20.0, 30.0, 40.0])

    p["lambda_mi_pos"] = trial.suggest_float("lambda_mi_pos", 0.0, 0.08)
    p["mi_start_epoch"] = trial.suggest_categorical("mi_start_epoch", [10, 20, 30, 40])
    p["mi_pos_min_count"] = trial.suggest_categorical("mi_pos_min_count", [8, 12, 16])

    p["dose_loss_weight"] = trial.suggest_float("dose_loss_weight", 0.01, 0.10)
    p["treat_clf_steps"] = trial.suggest_categorical("treat_clf_steps", [3, 5, 8])

    p["encoder_dropout"] = trial.suggest_categorical("encoder_dropout", [0.10, 0.15, 0.20])
    p["clf_dropout"] = trial.suggest_categorical("clf_dropout", [0.10, 0.15, 0.20])
    p["clip_norm"] = trial.suggest_categorical("clip_norm", [1.0, 2.0, 5.0])

    p["epochs"] = 120
    p["patience"] = 20

    p["use_contrastive"] = True
    p["use_local_mi"] = True
    p["use_dynamic_update"] = True
    p["pair_mode"] = "dynamic_ite"
    p["feature_k"] = 20

    return p


def objective_factory(bundle, true_grid_all, dose_grid):
    def objective(trial):
        params = build_search_space(trial, mod.BEST_PARAMS)

        split_scores = []
        split_epochs = []
        trial_rows = []

        for run_id in range(TUNE_RUNS_PER_TRIAL):
            split = mod.split_tcga_bundle(bundle, seed=100 + run_id, true_grid_all=true_grid_all)
            res = mod.train_single_run(
                split=split,
                run_seed=1000 + trial.number * 100 + run_id,
                hyperparams=params,
                model_name=f"optuna_trial_{trial.number}",
                dose_grid=dose_grid,
            )

            split_scores.append(float(res["sqrt_mise"]))
            split_epochs.append(float(res["epochs"]))

            trial_rows.append({
                "trial": trial.number,
                "run_id": run_id,
                "sqrt_mise": float(res["sqrt_mise"]),
                "mise": float(res["mise"]),
                "best_val_sqrt_mise": float(res["best_val_sqrt_mise"]),
                "epochs": float(res["epochs"]),
                "test_factual_rmse_norm": float(res["test_factual_rmse_norm"]),
            })

            trial.report(float(np.mean(split_scores)), step=run_id)

        pd.DataFrame(trial_rows).to_csv(
            os.path.join(OUT_DIR, f"trial_{trial.number:03d}_per_run.csv"),
            index=False,
            sep=";",
        )

        mean_score = float(np.mean(split_scores))
        std_score = float(np.std(split_scores))
        mean_epochs = float(np.mean(split_epochs))

        trial.set_user_attr("mean_sqrt_mise", mean_score)
        trial.set_user_attr("std_sqrt_mise", std_score)
        trial.set_user_attr("mean_epochs", mean_epochs)

        return mean_score

    return objective


def build_mi_ablation_configs(best_params):
    configs = {}

    p = copy.deepcopy(best_params)
    configs["full_model"] = p

    p = copy.deepcopy(best_params)
    p["use_local_mi"] = False
    configs["no_local_mi"] = p

    p = copy.deepcopy(best_params)
    p["lambda_mi_pos"] = 0.0
    configs["mi_weight_zero"] = p

    p = copy.deepcopy(best_params)
    p["mi_start_epoch"] = max(best_params.get("epochs", 120) + 1, 999)
    configs["mi_never_starts"] = p

    p = copy.deepcopy(best_params)
    p["treat_clf_steps"] = 1
    configs["mi_single_clf_step"] = p

    return configs


def run_final_evaluation(bundle, true_grid_all, dose_grid, best_params):
    summary_paths = []

    mod.run_experiment_group(
        "final_eval_true_mise",
        {"full_model": copy.deepcopy(best_params)},
        bundle,
        true_grid_all,
        dose_grid,
        FINAL_RUNS,
    )
    summary_paths.append(os.path.join(OUT_DIR, "final_eval_true_mise_aggregate.csv"))

    if RUN_COMPONENT_ABLATION:
        component_configs = mod.build_component_ablation_configs(best_params)
        mod.run_experiment_group(
            "component_ablation_true_mise_final",
            component_configs,
            bundle,
            true_grid_all,
            dose_grid,
            FINAL_RUNS,
        )
        summary_paths.append(os.path.join(OUT_DIR, "component_ablation_true_mise_final_aggregate.csv"))

    if RUN_PAIRING_ABLATION:
        pairing_configs = mod.build_pairing_ablation_configs(best_params)
        mod.run_experiment_group(
            "pairing_ablation_true_mise_final",
            pairing_configs,
            bundle,
            true_grid_all,
            dose_grid,
            FINAL_RUNS,
        )
        summary_paths.append(os.path.join(OUT_DIR, "pairing_ablation_true_mise_final_aggregate.csv"))

    mi_configs = build_mi_ablation_configs(best_params)
    mod.run_experiment_group(
        "mi_ablation_true_mise_final",
        mi_configs,
        bundle,
        true_grid_all,
        dose_grid,
        FINAL_RUNS,
    )
    summary_paths.append(os.path.join(OUT_DIR, "mi_ablation_true_mise_final_aggregate.csv"))

    if RUN_SENSITIVITY:
        sensitivity_configs = mod.build_sensitivity_configs(best_params)
        mod.run_experiment_group(
            "sensitivity_true_mise_final",
            sensitivity_configs,
            bundle,
            true_grid_all,
            dose_grid,
            FINAL_RUNS,
        )
        summary_paths.append(os.path.join(OUT_DIR, "sensitivity_true_mise_final_aggregate.csv"))

    return summary_paths


def main():
    print(f"[{now_str()}] Loading data ...")
    bundle = mod.load_tcga_mitnet_npz(NPZ_PATH)

    print(f"[{now_str()}] Building benchmark and true response grid cache ...")
    _, dose_grid, true_grid_all = mod.build_benchmark_and_true_grid_all(
        data_dir=TCGA_DATA_DIR,
        tcga_num_features=bundle["X"].shape[1],
        num_treatments=bundle["K"],
        dose_grid_size=DOSE_GRID_SIZE,
        batch_size=BENCHMARK_BATCH_SIZE,
        seed=BENCHMARK_SEED,
        strength_of_assignment_bias=BENCHMARK_ASSIGNMENT_BIAS,
        total_n=bundle["X"].shape[0],
    )

    print(f"[{now_str()}] Starting Optuna study ...")
    sampler = optuna.samplers.TPESampler(seed=2026, multivariate=True, group=True)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="minimize",
        sampler=sampler,
        storage=f"sqlite:///{STUDY_DB}",
        load_if_exists=True,
    )

    objective = objective_factory(bundle, true_grid_all, dose_grid)

    def callback(study, trial):
        save_study_csv(study, os.path.join(OUT_DIR, "optuna_trials.csv"))
        if study.best_trial is not None:
            best_payload = {
                "best_value": study.best_value,
                "best_params": study.best_params,
                "best_trial_number": study.best_trial.number,
                "best_user_attrs": study.best_trial.user_attrs,
            }
            save_json(best_payload, os.path.join(OUT_DIR, "best_optuna_result.json"))

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        timeout=OPTUNA_TIMEOUT_SEC,
        n_jobs=OPTUNA_N_JOBS,
        callbacks=[callback],
        gc_after_trial=True,
        show_progress_bar=False,
    )

    print(f"[{now_str()}] Optuna finished.")
    print(f"Best true sqrt_MISE: {study.best_value:.6f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    best_params = copy.deepcopy(mod.BEST_PARAMS)
    best_params.update(study.best_params)
    best_params["use_contrastive"] = True
    best_params["use_local_mi"] = True
    best_params["use_dynamic_update"] = True
    best_params["pair_mode"] = "dynamic_ite"
    best_params["feature_k"] = 20

    save_json(best_params, os.path.join(OUT_DIR, "best_params_for_final.json"))

    print(f"[{now_str()}] Starting final evaluation and ablations ...")
    summary_paths = run_final_evaluation(bundle, true_grid_all, dose_grid, best_params)

    print(f"[{now_str()}] Done.")
    print("Saved summary files:")
    for p in summary_paths:
        print(" -", p)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        err_path = os.path.join(OUT_DIR, "overnight_runner_exception.txt")
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(repr(e))
            f.write("\n\n")
            f.write(traceback.format_exc())
        raise
