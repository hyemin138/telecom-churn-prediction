# archive

대회 당시 팀 노트북 (참고용, 정리본은 `src/`)

| 파일 | 내용 |
|---|---|
| `original_notebooks/competition_pipeline.py` | 대회 제출 버전, 33개 변수(하위 4개 미제거). 대회 제출 수치(F1 0.7271)의 출처 |
| `original_notebooks/pipeline_29_features.py` | 하위 4개 제거한 29개 변수 버전 (F1 0.7182) |
| `original_notebooks/feature_importance.py` | CatBoost 변수 중요도 순위 |

<br>

## 정리하면서 바뀐 점

- 경로: Colab(`/content/...`) → 레포 기준 상대 경로
- County: 행마다 pgeocode 조회 → 고유 Zip Code만 조회 후 `data/zip_county.csv`로 저장 (결과 동일)
- 샘플링 비교: 대회 제출 버전의 RF 3-fold CV 방식 그대로, 29개 변수 기준으로 실행
- Grid Search: SMOTE-ENN을 Pipeline 안으로 옮겨 검증 폴드에 합성 데이터가 섞이지 않도록 수정
- 변수 개수별 F1 곡선: 원본 코드 미보관, 같은 방식으로 재구성
- 기본 파라미터 CatBoost: 보고서 비교표에 있으나 원본 코드 미보관, 기본값으로 추가

<br>

## 제외한 노트북

| 내용 | 제외 이유 |
|---|---|
| 샘플링 기법 Grid Search 실험 | SMOTE·ENN·SMOTE-ENN 파라미터 탐색, 최종 흐름과 별개 |
| 파생변수 실험 | Family_Size 파생변수 검토, 최종 모델 미사용 |
| EDA 그래프 원본 | `src/01_eda.py`로 정리 |
| 초기 EDA | 다른 버전 데이터(Kaggle 기본판 21열) |
