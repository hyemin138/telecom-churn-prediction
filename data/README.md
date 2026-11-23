# data

## `telco.csv`

IBM Telco Customer Churn 샘플 데이터 (7,043행 x 50열)

- 출처: [Kaggle — Telco customer churn (11.1.3+)](https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3), IBM Cognos Analytics 11.1.3+ 샘플 데이터
- 캘리포니아 가상 통신사 고객 정보 (실제 고객 아님)
- 타깃: `Churn Label` (Yes 26.5% / No 73.5%)
- 결측: `Offer` 3,877건(미가입), `Internet Type` 1,526건(인터넷 미이용)

## `zip_county.csv`

`src/data.py` 첫 실행 시 pgeocode로 생성하는 Zip Code -> County 매핑. 이후 실행은 이 파일 사용 (인터넷 불필요)
