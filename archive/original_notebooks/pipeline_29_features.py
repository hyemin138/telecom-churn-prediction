# Colab 원본 "통계자료분석 전처리 & eda" (참고용)
# 29개 변수(하위 4개 제거) 버전, 출력: CatBoost 0.8318 / F1 0.7182 / Recall 0.8075 / AUC 0.9091

import numpy as np
import pandas as pd

telco = pd.read_csv('/content/telco.csv')

pd.set_option('display.max_columns', None)

# 우편번호 → County(주)

# %pip install pgeocode

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

# 최종변수 선택

cols = ['Customer ID', 'Under 30', 'Senior Citizen',
        'Dependents', 'Country', 'State', 'Zip Code', 'Latitude','City',
        'Longitude', 'Quarter', 'Referred a Friend', 'Churn Score',
        'Churn Category', 'Churn Reason', 'Customer Status', 'Satisfaction Score',
        'Unlimited Data','Device Protection Plan','Internet Service','Phone Service']

# 'Churn Score', 'Churn Category', 'Churn Reason', 'Customer Status' → 누수 데이터, 'Satisfaction Score' → 사후 결과 변수
# 'Customer ID' → 비관련 변수, 'Under 30', 'Senior
# Citizen','Dependents', 'Referred a Friend' → 중복 변수
# 'Country', 'State', 'Quarter' → 상수형 변수
# 'Latitude', 'Longitude', 'Zip Code', 'City' → 고유값이 너무 큼 → 고차원 문제 발생
# 'City' 상위 범주인 'County'로 파생변수 활용
#'Unlimited Data','Device Protection Plan','Internet Service','Phone Service' → 피처중요도 하위 4개 삭제
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

# 샘플링

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours as ENN
from imblearn.combine import SMOTEENN

smote = SMOTE(sampling_strategy=0.7, k_neighbors=3, random_state=42)
enn = ENN(n_neighbors=3, kind_sel='mode')
smoteenn = SMOTEENN(smote=smote, enn=enn, random_state=42)


X_train_sme, y_train_sme = smoteenn.fit_resample(X_train, y_train)

# %pip install catboost
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier


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
