# Deep-FM-Preprocessing-Ablation
## 실험 목적

간단한 추천 시스템에서 **feature preprocessing 방식이 성능에 미치는 영향**을 비교했다.

### 데이터 / 모델

* Dataset: MovieLens-1M
* Task: implicit recommendation
* Positive: 실제 interaction
* Negative: 사용자가 보지 않은 item을 1:1 sampling
* Split: timestamp 기준 chronological split

  * Train 80%
  * Validation 10%
  * Test 10%
* Model: DeepFM
* Metric:

  * AUC
  * LogLoss

## 1. Numerical Feature 정규화

동일한 모델에서 numerical feature 처리 방식만 변경했다.

```text
Raw
Min-Max
Z-score
log1p + Z-score
Log Bucket + Embedding
```

사용한 주요 numerical feature:

```text
age
movie_year
user_activity
item_popularity
time_from_reference_days
```

목적은 **Min-Max normalization이 실제로 추천 성능에 도움이 되는지**, 그리고 long-tail feature에 `log1p`나 bucketization이 더 좋은지를 비교하는 것이다.

## 2. 시간 Feature Encoding

시간 정보를 여러 방식으로 표현해서 비교했다.

```text
Raw
Min-Max
Embedding
Sin/Cos
Embedding + Sin/Cos
```

사용한 시간 feature:

```text
hour
day_of_week
```

특히 시간은 순환 구조가 있기 때문에:

$$
23시 \approx 0시
$$

단순 Min-Max보다 다음과 같은 cyclic encoding도 비교한다.

$$
\sin(2\pi hour/24),\quad
\cos(2\pi hour/24)
$$

요일도 동일하게:

$$
\sin(2\pi day/7),\quad
\cos(2\pi day/7)
$$

## 3. Correlation 기반 Feature Selection

numerical feature 간 correlation이 너무 높은 경우 하나를 제거했다.

```text
All features
|corr| < 0.99
|corr| < 0.95
|corr| < 0.90
```

여기서는 **target과의 correlation으로 feature를 제거하지 않고**, feature-feature 중복만 제거한다.

이유는 DeepFM에서 개별 feature의 target correlation이 낮더라도

```text
user × item
hour × genre
```

같은 interaction을 통해 중요한 feature가 될 수 있기 때문이다.

## 최종 비교

```text
Numerical preprocessing
    ↓
Raw vs MinMax vs Z-score vs Log vs Bucket

Temporal encoding
    ↓
Raw vs MinMax vs Embedding vs Sin/Cos

Feature selection
    ↓
All vs correlation threshold
```

각 실험에서는 다른 조건을 최대한 고정해서 **feature engineering 자체의 효과만 비교**하도록 구성했다.
