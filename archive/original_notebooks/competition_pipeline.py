# Colab 원본 1111.ipynb, 대회 제출 버전 (참고용, 정리본은 src/)
# 33개 변수(하위 4개 미제거) 버전, 출력: CatBoost 0.8375 / F1 0.7271 / Recall 0.8155 / AUC 0.9106

# 데이터 로딩

import numpy as np
import pandas as pd

telco = pd.read_csv('telco-2.csv')

pd.set_option('display.max_columns', None)

# 우편번호 → County(주)

import pgeocode

nomi = pgeocode.Nominatim('us')

def county(zip_code):
    info = nomi.query_postal_code(str(zip_code))
    return info.county_name

telco['County'] = telco['Zip Code'].apply(county)

# County에 결측치 4개 존재  - 직접 찾아서 대체

telco[telco['County'].isna()]

telco.loc[
    (telco['Latitude'] == 38.244806) & (telco['Longitude'] == -120.417301),
    'County'
] = 'Calaveras County'

# 최종 변수 선정

cols = ['Customer ID', 'Under 30', 'Senior Citizen',
        'Dependents', 'Country', 'State', 'Zip Code', 'Latitude','City',
        'Longitude', 'Quarter', 'Referred a Friend', 'Churn Score',
        'Churn Category', 'Churn Reason', 'Customer Status', 'Satisfaction Score']

# 'Churn Score', 'Churn Category', 'Churn Reason', 'Customer Status' → 누수 데이터, 'Satisfaction Score' → 사후 결과 변수
# 'Customer ID' → 비관련 변수, 'Under 30', 'Senior
# Citizen','Dependents', 'Referred a Friend' → 중복 변수
# 'Country', 'State', 'Quarter' → 상수형 변수
# 'Latitude', 'Longitude', 'Zip Code', 'City' → 고유값이 너무 큼 → 고차원 문제 발생
# 'City' 상위 범주인 'County'로 파생변수 활용
telco = telco.drop(columns=cols)
telco.info()

# 결측값 처리

telco['Offer'] = telco['Offer'].fillna('-')
telco['Internet Type'] = telco['Internet Type'].fillna('-')

# 인코딩 + 데이터 분리

y = (telco['Churn Label'] == 'Yes').astype(int)
X = telco.drop(columns=['Churn Label'])


cats = X.select_dtypes(include=['object']).columns.tolist()
X = pd.get_dummies(X, columns=cats, dtype=int)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# (샘플링)

smote = SMOTE(sampling_strategy=0.7, k_neighbors=3, random_state=42)
enn = ENN(n_neighbors=3, kind_sel='mode')
smoteenn = SMOTEENN(smote=smote, enn=enn, random_state=42)


X_train_sme, y_train_sme = smoteenn.fit_resample(X_train, y_train)

# 모델링 (더블랜덤포레스트)

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

class DoubleRandomForest:
    def __init__(self, n_estimators=500, max_features='sqrt', max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n, p = X.shape

        for b in range(self.n_estimators):

            Xb, yb = resample(X, y)

            # 2 노드별 부트스트랩 반영할 결정트리 생성
            tree = DecisionTreeClassifier(
                max_features=self.max_features,
                max_depth=self.max_depth,
                random_state=self.random_state + b if self.random_state else None
            )
            tree.fit(Xb, yb)
            self.trees.append(tree)

    def predict_proba(self, X):
        probs = np.mean([tree.predict_proba(X) for tree in self.trees], axis=0)
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

drf = DoubleRandomForest(n_estimators=500, max_depth=10 ,random_state=42)
drf.fit(X_train_sme, y_train_sme)

y_pred = drf.predict(X_test)
y_proba = drf.predict_proba(X_test)[:, 1]

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")


# 모델링 (XGBoost)

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

xgb = XGBClassifier(
    random_state=42,
    eval_metric='logloss'
)

xgb.fit(X_train_sme, y_train_sme)

y_pred  = xgb.predict(X_test)
y_proba = xgb.predict_proba(X_test)[:, 1]

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")

# 모델링 (CatBoost) (최적파라미터)

from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score


cat = CatBoostClassifier(
    depth=8,
    iterations=800,
    l2_leaf_reg=3,
    learning_rate=0.03,
    random_state=42,
    eval_metric='Logloss'
)

cat.fit(X_train_sme, y_train_sme)

y_pred  = cat.predict(X_test)
y_proba = cat.predict_proba(X_test)[:, 1]

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")

# 모델링 (베이스모델)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_sme, y_train_sme)

y_pred  = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")

# 베이스모델로 샘플링 비교 → SMOTEENN 채택

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, recall_score, accuracy_score, roc_auc_score

# 샘플링 (Default, SMOTE, SMOTEENN)
samplers = {
    "None": "passthrough",
    "SMOTE": SMOTE(random_state=42),
    "SMOTEENN": SMOTEENN(random_state=42)
}

# 기본 모델
rf = RandomForestClassifier(random_state=42)

# 교차검증 설정
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# 지표 확인
scoring = {
    "f1": make_scorer(f1_score),
    "recall": make_scorer(recall_score),
    "accuracy": make_scorer(accuracy_score),
    "roc_auc": "roc_auc"
}


#  샘플링 별로 교차검증 수행
results = []
for name, sampler in samplers.items():
    pipe = Pipeline([("sampler", sampler), ("model", rf)])
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

    results.append({
        "Sampler": name,
        "F1": np.mean(scores["test_f1"]),
        "Recall": np.mean(scores["test_recall"]),
        "Accuracy": np.mean(scores["test_accuracy"]),
        "ROC_AUC": np.mean(scores["test_roc_auc"])
    })

# 결과 확인
result = pd.DataFrame(results).sort_values("F1", ascending=False)
print(result)

# XAI

from catboost import Pool
import shap

explainer = shap.TreeExplainer(cat)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)

import shap

explainer = shap.TreeExplainer(cat)
shap_values = explainer.shap_values(X_test)

i = 777 # 본인이 원하는 데이터(행) 넘버 적으면 됨!!
shap.initjs()
shap.force_plot(
    explainer.expected_value,
    shap_values[i],
    X_test.iloc[i],
    matplotlib=True
)

# CatBoost 하이퍼파라미터 서치 (격자 탐색)

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, recall_score
from catboost import CatBoostClassifier

cat = CatBoostClassifier(
    random_state=42,
    eval_metric='Logloss',
    verbose=0
)


param_grid = {
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'iterations': [200, 500, 800],
}

# 다중 스코어러 정의
scoring = {
    'F1': make_scorer(f1_score),
    'Recall': make_scorer(recall_score)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=cat,
    param_grid=param_grid,
    scoring=scoring,
    refit='F1',
    cv=cv,
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train_sme, y_train_sme)

print("Best Params:", grid.best_params_)
print("Best F1 Score:", grid.best_score_)

# 각 스코어별 결과 보기
cv_results = grid.cv_results_
print("Recall mean:", cv_results['mean_test_Recall'][grid.best_index_])

# 'depth': 10, 'iterations': 800, 'l2_leaf_reg': 3, 'learning_rate': 0.05
