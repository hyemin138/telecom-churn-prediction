# CatBoost 변수 중요도 (참고용, 정리본은 src/02_feature_selection.py)

# %pip install pgeocode catboost
import pandas as pd
import numpy as np
import pgeocode
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN

# 1. 데이터 로드 및 County 파생
telco = pd.read_csv('/content/telco.csv')

nomi = pgeocode.Nominatim('us')
def county(zip_code):
    info = nomi.query_postal_code(str(zip_code))
    return info.county_name

telco['County'] = telco['Zip Code'].apply(county)
telco.loc[
    (telco['Latitude'] == 38.244806) & (telco['Longitude'] == -120.417301),
    'County'
] = 'Calaveras County'

# 2. 불필요 컬럼 제거 및 결측 처리
drop_cols = [
    'Customer ID', 'Under 30', 'Senior Citizen', 'Dependents',
    'Country', 'State', 'Zip Code', 'Latitude', 'City',
    'Longitude', 'Quarter', 'Referred a Friend', 'Churn Score',
    'Churn Category', 'Churn Reason', 'Customer Status', 'Satisfaction Score'
]
telco = telco.drop(columns=drop_cols, errors='ignore')

# 결측치 처리
telco['Offer'] = telco['Offer'].fillna('-')
telco['Internet Type'] = telco['Internet Type'].fillna('-')

print(f" 전처리 완료 — 남은 변수 수: {telco.shape[1]}개")

# 3. X, y 분리 및 학습/검증 분할
y = (telco['Churn Label'] == 'Yes').astype(int)
X = telco.drop(columns=['Churn Label'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42, test_size=0.2
)

# CatBoost용 범주형 변수 리스트
cat_cols = X.select_dtypes(include='object').columns.tolist()
print(f" 범주형 컬럼 개수: {len(cat_cols)}개")

# 4. SMOTEENN (불균형 보정, 문자형은 코드로 임시 변환)
X_train_num = X_train.copy()
for c in cat_cols:
    X_train_num[c] = X_train_num[c].astype('category').cat.codes

smote = SMOTE(sampling_strategy=0.7, k_neighbors=3, random_state=42)
enn = EditedNearestNeighbours(n_neighbors=3, kind_sel='mode')
smoteenn = SMOTEENN(smote=smote, enn=enn, random_state=42)

X_train_sme, y_train_sme = smoteenn.fit_resample(X_train_num, y_train)
print(f" SMOTEENN 완료 — 리샘플 후 크기: {X_train_sme.shape}")

# 5. CatBoost 학습 및 변수 중요도 분석
cat_idx = [X_train.columns.get_loc(c) for c in cat_cols if c in X_train.columns]
train_pool = Pool(X_train, y_train, cat_features=cat_idx)

cat_model = CatBoostClassifier(
    iterations=800,
    depth=10,
    learning_rate=0.05,
    l2_leaf_reg=3,
    random_state=42,
    verbose=100
)
cat_model.fit(train_pool)

# 변수 중요도 계산
importances = cat_model.get_feature_importance(train_pool)
feature_ranking = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values('Importance', ascending=False)

# 6. 결과 출력 및 저장
print("\n CatBoost 기반 변수 중요도 상위 20개:")
print(feature_ranking.head(20))

top_features = feature_ranking.head(20)['Feature'].tolist()
print("\n 상위 20개 주요 변수:")
for i, f in enumerate(top_features, 1):
    print(f"{i:>2}위 : {f}")


from google.colab import drive
drive.mount('/content/drive')

feature_ranking = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values('Importance', ascending=False).reset_index(drop=True)

# 인덱스를 1부터 시작하도록 설정
feature_ranking.index = feature_ranking.index + 1
feature_ranking.index.name = 'Rank'

feature_ranking

# 중요도 0 변수 확인

# 하위 4개 변수 제외
