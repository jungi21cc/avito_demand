### 1. feature engineering

### user id

- item id가 실제 사용되는 unique id이고 user id는 중복될수 있다
- 77만개의 user id가 있고, 최대 1000번 사용되어 있다.
- label encoding으로 사용할것인가, 다른 방법을 고안해볼 것인가

### region + city?

- 그냥 나둘것인가, mapping해줄것인가, region + city 합쳐서 생각할 것인가
- region : 28, city : 1752
- insight로 지역별 광고노출에 deal probability가 차이가 날수 있다

### parent_category_name, category_name

- region + city와 동일하게 그냥 나둘것인가, mapping해줄것인가, 합쳐서 생각할 것인가
- parent category : 9, category : 47

### param_1, 2, 3

- missing value문제가 발생한다
- param_1의 경우 6%, param_2, 3은 각각 59%, 78%의 missing value를 가지고 있다
- param의 특성상 제품의 상세 내역을 나타낸다
- 의류를 예로들어 아래와 같은 형식을 가진다
- 상위>하위분류    >param1>2  >    3>제목>제품설명
- 잡화>의류,악세사리>여성의복>바지>수량>제목>제품설병
- param1의경우 category_name과 유사한 부분이 많다
- param2,3의 경우 해당 재품의 좀더 상세한 옵션을 제공하다보니 충분히 missing value가 발생할수 있다
- param1+2+3을 합치고 null data의 경우 category_name으로 채운다
- description의 특성을 추출할 경우 더 높은 신뢰도를 가지겠지만, 우선 category_name으로 채운다
- TF-IDF로 vectorization해준다

### title, description

- title은 missing value가 존재하지 않는다
- description은 8%의 missing value가 존재한다
- 앞서 param과 동일하게 제품에 대한 제목과 설명에 해당하는 내용이다
- title의 경우 명사로 대부분 이루어 져있고, description의 경우 구어체와 같은 형식으로 이루어 져있다
- title과 description은 unique한 제품의 설명이기 때문에 TF-IDF로 vectorization해준다

### price

- price의 경우 8%의 missing value를 가진다
- log를 취하였을때 정규분포에 가까운 분포를 보인다
- missing value의 경우 카테고리별 median 값으로 채워넣을수 있다
- 이유는 가격의 편차를 줄이는데 카테고리별 median값이 전체 가격의 mean, median보다 더 잘 설명될수 있다

### item_seq_number

- 3만개의 item_seq_number가 있는데 이는 해당 user_id에 광고노출 횟수를 의미한다고 유추할수 있다
- kaggle data description에는 자세한 설명은 나와있지 않지만, user type별로 company, shop, private별로 구분해 보았을때 shop>company>private순으로 평균 item_seq_number가 나왔다
- 노출횟수라고 가정하면 log를 취해서 조금더 정규분포에 가깝게 조정할수 있다

### activation_date

- 활동날짜에 해당하는 자료인데, periods_train, periods_test에 activation date, date from, date to가 추가 정보가 있다. 하지만 해당 item id에는 train, test데이터에 해당하는 user id가 존재하지 않는다
- periods_train, test에서 item_id를 기준으로 date to 와 from의 차이를 합과 평균으로 groupby 해준다
- 새로 만들어진 dataframe을 가지고 기존의 train, test의 item_id를 기준으로 다시 groupby 해준다
- 그럼 user id를 기준으로 평균 활동시간 차이, 기간차이, 중복된 횟수를 구할수 있다

### user type

- 3가지 user type이 있다 (company, shop, private)
- item_seq_num과 더불어 user type별로 mean, count를 groupby하면 더 풍부한 feature를 만들어 낼수 있다

### image, image_top_1

- 10%의 missing value를 가진다
- 이미지 데이터 자체를 학습해서 사용하기에는 image의 noise가 너무 심해 사용하기 힘들다
- image_top_1의 경우에는 avito model에서 image classification code를 3000가지 종류로 분류해 놓았다
- vgg16과 같은 pre-trained model을 활용해 image feature extraction을 이용해 불필요한 noise를 지우고 이미지 데이터를 활용할수 있다
- feature extraction의 경우 PCA와 같은 방식으로 불필요하게 많은 feature를 선별할수 있다

### target : deal_probability

- log를 취해도 0과 1의 비중이 앞도적으로 높아 정규분포를 띄지 않는다
- 우선은 modeling이후 prediction할 경우 0과 1을 벗어나는 예측값은 clip을 이용해 수치를 조정한다


### 2. modeling

- lightGBM Regressor를 이용해 모델링을 진행한다
- 이후 xgboost, catboost를 활용해 결과값을 비교한다
- 세가지 모델을 stacking하여 최종 결과를 제출한다



