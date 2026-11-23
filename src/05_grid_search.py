"""CatBoost 하이퍼파라미터 격자 탐색 (5-fold CV, F1 기준)

SMOTE-ENN은 Pipeline 안에서 학습 폴드에만 적용
조합 240개 x 5-fold, 실행 시간 김 (Colab CPU 기준 1시간 이상)
실행: python src/05_grid_search.py
"""
import json

import pandas as pd
from catboost import CatBoostClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from data import RESULTS, SEED, load_train_test, make_sampler

OUT = RESULTS / "05_grid_search"

PARAM_GRID = {
    "model__depth": [4, 6, 8, 10],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "model__l2_leaf_reg": [1, 3, 5, 7, 9],
    "model__iterations": [200, 500, 800],
}


def build_pipeline():
    """SMOTE-ENN + CatBoost (병렬은 GridSearchCV에서 처리)"""
    model = CatBoostClassifier(random_state=SEED, eval_metric="Logloss", verbose=0, thread_count=1)
    return Pipeline([("sampler", make_sampler()), ("model", model)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X_train, _, y_train, _ = load_train_test()

    grid = GridSearchCV(build_pipeline(), PARAM_GRID, scoring={"F1": "f1", "Recall": "recall"},
                        refit="F1", cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
                        n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    cols = ["params", "mean_test_F1", "mean_test_Recall", "rank_test_F1"]
    pd.DataFrame(grid.cv_results_)[cols].sort_values("rank_test_F1").to_csv(
        OUT / "cv_results.csv", index=False)

    best = {k.replace("model__", ""): v for k, v in grid.best_params_.items()}
    result = {"best_params": best, "best_cv_f1": round(grid.best_score_, 4)}
    (OUT / "best_params.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()
