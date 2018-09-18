# avito_demand

### kaggle : Avito Demand Prediction Challenge

#### 개요

- 상품설명을 기반으로하는 온라인 광고 수요예측 회귀모델 설립
- RMSE 를 통한 회귀모델 평가
- Light GBM 모델 사용

“최종 결과 : 0.2240 (상위 24% )”

#### 데이터

- Train.csv
트레이닝 데이터 : (1503424, 18)

Target 데이터 : deal_probability (0~1 확률분포)

- Test.csv
테스트 데이터 : (508438, 17)

트레이닝 데이터를 통한 Target 데이터 예측

- periods_train.csv, periods_test.csv
사용자 활동데이터: item_id, activation date, date from, date to

- Train_active.csv, test_active.csv,  (train, test 데이터에 없는 정보로 최종 미사용)

사용자 활동 데이터

- Train_jpg.csv, test_jpg.csv  (데이터셋의 아이템 사진이지만 노이즈가 심해서 최종 미사용)

데이터셋의 아이템 사진

#### 데이터 엔지니어링

- Null 데이터 비율
-   Null data의 경우 상품설명에 관련한 데이터는 Google Translate API를 사용하여 러시아어로 '데이터없음' 이라고 채움
-   param_1 : 4% (데이터 채움)
-   param_2 : 44% (데이터 채움)
-   param_3 : 57% (데이터 채움)
-   description : 8% (데이터 채움)
-   price : 6% (mode 값으로 데이터 채움)
-   image : 7% (이미지는 최종모델에서 사용하지 않아서 채우지 않음)
-   image_top_1 : 7% (이미지는 최종모델에서 사용하지 않아서 채우지 않음)

- active date / account create date
-   년, 월, 일, 주차, 주말 데이터로 변경하여 사용

- Natural Language Process
-   Param_1, 2, 3, description, title 은 전부 상품 상세 설명에 해당
-   Param_1, 2, 3 의 경우 param_1이 대부분의 상품설명, 2, 3은 세부적 상품설명
-   예) param_1, 2, 3 = 여성티셔츠, 황금로고, M사이즈
-   Description은 판매자의 상품에 대한 설명
-   예) 여자친구 선물용 상태 좋음
-  Title은 상품제목
-   예) 여자티셔츠
-   상품상세설명에 대해 자연어 처리를 위해 TF-IDF(Bag of Words) 방법을 사용해 처리
-   분류모델의 경우 word embedding 보다 문서내에서 사용된 빈도수로 Bag of Words에서 처리하는 것이 성능우위
-   Param_1, 2, 3은 같은 상품 설명으로 취급하기 위해 하나의 feature로 합침
-   러시아어를 tokenizing(Part of Speech Tagging) 해주기 위해서 NLTK의 stopwords기능을 사용 (조사등을 제거해 주고, 형태소(명사, 형용사등) 단위로 나눠줌)
-   조사, 기호등 vectorization 해주면서 sparsity문제때문에 임의로 최대 10,000개의 TF-IDF단어만 사용하도록 제한

- 카테고리 데이터
-   카테고리 변수를 더미변수로 변경

- Feature importance
-   Feature engineering이후 24,686개의 feature를 사용하여 첫번째 Light GBM모델을 이용해 학습
-   이후 대부분의 feature가 사용되지 않아서 feature importance가 0 이상인(실제로 분류에 사용된) feature들 680개만 추출하여 새롭게 Light GBM을 사용하여 학습

“최종 680개 Feature 사용”

#### 모델링

- 교차검증
-   RMSE로 분류검증
-   Light GBM을 활용한 성능 평가

“교차검증기준 Light GBM의 feature importance를 통해 중요하게 사용된 feature만 사용하여 다시 Light GBM으로 모델링”

- Light GBM 파라미터 튜닝 (교차검증기준 가장 성능이 우수한 parameter  선택)

-   Boosting_type = gbdt
-   Objective = regression
-   Metric = rmse
-   Max_depth = -1
-   Num_leaves = 37
-   Feature_fraction = 0.7
-   Bagging_fraction = 0.8
-   Learning_rate = 0.015
-   Verbose =  0
-   N_jobs =  -1
-   Reg_rambda = 0.5

- 결론
-   RMSLE결과: 0.2240 (상위 24%)

"
Kaggle의 RMSE의 결과제출의 경우 여러개의 모델을 만든다음 correlation이 높은 결과만 합쳐 평균낼경우(blending) 성능 향상 기대 

Stacking methods를 사용해 모델복잡도를 높일경우 성능향상 기대
"
