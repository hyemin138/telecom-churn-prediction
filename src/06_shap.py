"""SHAP 해석: 최종 CatBoost 모델의 변수별 이탈 기여도

실행: python src/06_shap.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier

from data import RESULTS, load_train_test

OUT = RESULTS / "06_shap"
MODEL_PATH = RESULTS / "04_models" / "catboost_final.cbm"  # 04_models.py 실행 후 생성
SAMPLE_IDX = 777  # 개별 고객 예시


def shap_table(shap_values, X_test):
    """변수별 평균 |SHAP|와 영향 방향 (변수 값과 SHAP 값의 상관, 양수면 이탈 증가)"""
    mean_abs = np.abs(shap_values).mean(axis=0)
    direction = [np.corrcoef(X_test[c], shap_values[:, i])[0, 1] if X_test[c].std() > 0 else 0
                 for i, c in enumerate(X_test.columns)]
    table = pd.DataFrame({"mean_abs_shap": mean_abs, "direction": direction}, index=X_test.columns)
    return table.sort_values("mean_abs_shap", ascending=False).round(4)


def save_plot(name):
    """현재 그래프 저장"""
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=130, bbox_inches="tight")
    plt.close("all")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, X_test, _, _ = load_train_test()
    model = CatBoostClassifier().load_model(str(MODEL_PATH))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, show=False)
    save_plot("summary_beeswarm.png")
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    save_plot("summary_bar.png")
    shap.force_plot(explainer.expected_value, shap_values[SAMPLE_IDX], X_test.iloc[SAMPLE_IDX],
                    matplotlib=True, show=False)
    save_plot(f"force_plot_{SAMPLE_IDX}.png")

    table = shap_table(shap_values, X_test)
    table.to_csv(OUT / "shap_importance.csv")
    print(table.head(10).to_string())


if __name__ == "__main__":
    main()
