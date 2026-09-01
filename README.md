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
결과가 꽤 명확해. 다만 **Raw의 AUC는 높지만 LogLoss 4.12라서 좋은 모델이라고 보기 어렵다.**

### 1. Numerical normalization

| 방식                |        AUC |    LogLoss | 해석                  |
| ----------------- | ---------: | ---------: | ------------------- |
| Raw               |     0.7848 | **4.1158** | calibration 심각하게 나쁨 |
| Min-Max           |     0.7608 |     0.8858 | 안정화되지만 AUC 하락       |
| Z-score           |     0.7731 |     0.8850 | Min-Max보다 나음        |
| **log + Z-score** | **0.7877** | **0.8546** | **가장 좋음**           |
| Log Bucket        |     0.7388 |     2.0139 | 가장 나쁨               |

**결론: `log1p + Z-score`가 가장 적절하다.**

`user_activity`, `item_popularity` 같은 추천 feature는 long-tail이라 단순 Min-Max보다 log transform으로 큰 값을 먼저 압축하는 게 효과적이었던 것으로 보인다.

Raw는 AUC만 보면 2등이지만 LogLoss가 `4.12`로 폭발했다. 즉 **ranking은 어느 정도 맞지만 확률을 지나치게 확신하는 문제가 발생**한 것.

---

### 2. 시간 정보

| 방식                  |        AUC |    LogLoss |
| ------------------- | ---------: | ---------: |
| Raw                 |     0.7895 |     0.7333 |
| **Min-Max**         | **0.7922** | **0.7234** |
| Embedding           |     0.7731 |     0.8850 |
| Sin/Cos             |     0.7907 |     0.7259 |
| Embedding + Sin/Cos |     0.7730 |     0.8610 |

이번 데이터에서는 의외로 **Min-Max가 최고**다.

```text
MinMax     AUC 0.7922
Sin/Cos    AUC 0.7907
Raw        AUC 0.7895
Embedding  AUC 0.7731
```

다만 MinMax와 Sin/Cos 차이는 **0.0016 AUC**밖에 안 되므로 seed 여러 개 없이 "MinMax가 확실히 우월하다"고 하기는 어렵다.

중요한 결과는 오히려:

> **시간을 무조건 embedding하는 것이 좋지 않았다.**

MovieLens에서 hour/day-of-week signal이 강하지 않고 데이터도 제한적이라 24/7개의 별도 embedding을 학습하는 것보다 연속적인 representation이 잘 일반화했을 가능성이 있다.

---

### 3. Correlation filtering

여기는 결과가 완전히 동일하다.

```text
all       0.790664
corr .99  0.790664
corr .95  0.790664
corr .90  0.790664
```

이건 **correlation filtering이 효과가 없었다기보다, threshold를 넘어서 실제 제거된 feature가 없었을 가능성이 매우 높다.**

코드가 출력한:

```text
Removed by correlation: [...]
```

부분을 확인하면 된다. 아마 전부:

```text
Removed by correlation: []
```

일 가능성이 높다.

따라서 현재 결과로는 **"correlation 기반 feature selection이 안 좋다"는 결론을 내릴 수 없다.**

실험 자체가 사실상:

```text
All == corr99 == corr95 == corr90
```

인 동일한 모델을 네 번 학습한 셈이다.

---

### 전체 결론

이번 실험에서 얻을 수 있는 핵심은:

```text
Numerical
    log1p + Z-score  ← Best

Temporal
    Min-Max          ← Best
    Sin/Cos          ← 거의 동일

Correlation
    결론 불가
    → 실제 제거된 feature가 없는지 확인 필요
```

특히 실무적으로는 **모든 numerical feature에 무조건 Min-Max를 적용하는 건 최선이 아니다.** 분포를 보고 처리해야 한다.

```text
user_activity / popularity / count
        ↓
log1p → Z-score

hour / day-of-week
        ↓
SinCos 또는 MinMax

categorical ID
        ↓
Embedding
```

그리고 한 가지 주의할 점이 있다. Numerical 실험의 최고 `log_zscore = 0.7877`과 Temporal 실험의 최고 `minmax = 0.7922`를 보고 **둘을 합치면 더 좋아질 것이라고 바로 결론 내릴 수는 없다.** 마지막으로 `log_zscore + temporal minmax` 조합을 한 번 돌려서 최종 AUC를 확인하는 게 깔끔하다.


각 실험에서는 다른 조건을 최대한 고정해서 **feature engineering 자체의 효과만 비교**하도록 구성했다.
