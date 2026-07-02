from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.model_selection import cross_val_score

from src.baselines.data_loading import load_dataset_for_baseline
from src.baselines.featurizers import transform_features


def _resolve_output_filename(cfg) -> str:
    dataset_name = str(cfg.dataset.name)
    k = int(cfg.kmer_size)
    if dataset_name == "polymerase":
        type_dataset = str(cfg.dataset.type_dataset).lower()
        return f"tune_C_kmers_{type_dataset}_k_{k}_logistic.csv"
    return f"tune_C_kmers_k_{k}_logistic.csv"


@hydra.main(config_path="../src/conf", config_name="tune_kmers_logistic", version_base=None)
def main(cfg) -> None:
    if str(cfg.featurizer.name) != "kmer_count":
        raise ValueError("This tuner expects featurizer=kmer_count.")

    seed = int(cfg.random_state)
    data = load_dataset_for_baseline(cfg, seed=seed)
    X_train, X_test = transform_features(
        cfg.featurizer, data["X_train_raw"], data["X_test_raw"]
    )
    y_train = np.asarray(data["y_train"]).ravel()
    y_test = np.asarray(data["y_test"]).ravel()

    c_values = [float(c) for c in cfg.C_values]
    rows = []

    for c in c_values:
        print(f"\nTesting C = {c}")
        model = LogisticRegression(
            C=c,
            max_iter=int(cfg.max_iter),
            random_state=seed,
            solver=str(cfg.solver),
            penalty=str(cfg.penalty),
        )

        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=int(cfg.cv_folds),
            scoring=str(cfg.cv_scoring),
        )
        mean_cv_auprc = float(cv_scores.mean())
        std_cv_auprc = float(cv_scores.std())

        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        y_scores_test = model.predict_proba(X_test)[:, 1]

        coefficients = model.coef_[0]
        n_nonzero_coefs = int((np.abs(coefficients) > 1e-10).sum())
        n_total_features = int(len(coefficients))

        test_accuracy = float(accuracy_score(y_test, y_pred_test))
        test_f1 = float(f1_score(y_test, y_pred_test))
        test_auprc = float(average_precision_score(y_test, y_scores_test))

        rows.append(
            {
                "C": c,
                "cv_auprc_mean": mean_cv_auprc,
                "cv_auprc_std": std_cv_auprc,
                "test_accuracy": test_accuracy,
                "test_f1": test_f1,
                "test_auprc": test_auprc,
                "n_nonzero_coefs": n_nonzero_coefs,
                "n_total_features": n_total_features,
            }
        )

        print(f"  CV AUPRC: {mean_cv_auprc:.4f} (+/- {std_cv_auprc:.4f})")
        print(f"  Test AUPRC: {test_auprc:.4f}")
        print(f"  Non-zero coefs: {n_nonzero_coefs} / {n_total_features}")

    df_results = pd.DataFrame(rows)
    best_idx = int(df_results["cv_auprc_mean"].idxmax())
    best_c = float(df_results.loc[best_idx, "C"])

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / str(cfg.output_dir_template)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = cfg.get("output_filename") or _resolve_output_filename(cfg)
    output_path = output_dir / str(output_filename)
    df_results.to_csv(output_path, index=False)

    print(f"\nSaved tuning results to: {output_path}")
    print(f"\n{'=' * 50}")
    print(f"Best C based on CV AUPRC: {best_c}")
    print(f"CV AUPRC: {df_results.loc[best_idx, 'cv_auprc_mean']:.4f}")
    print(f"Test AUPRC: {df_results.loc[best_idx, 'test_auprc']:.4f}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
