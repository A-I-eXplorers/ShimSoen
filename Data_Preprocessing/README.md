# 데이터 전처리, 탐색적 데이터 분석(EDA) 및 차원 축소(PCA) 실습

Python의 데이터 분석 및 머신러닝 라이브러리(`pandas`, `scikit-learn`, `seaborn`, `matplotlib`, `missingno`)를 활용하여 대표적인 3가지 데이터셋(**Wine**, **Iris**, **Titanic**)의 특성을 파악하고, 전처리·파생변수 생성·시각화·주성분 분석(PCA)을 수행한 실습 노트북(`데이터전처리.ipynb`)입니다.

---

## 🛠 기술 스택 및 개발 환경

- **환경**: Google Colaboratory (`데이터전처리.ipynb`)
- **언어**: Python 3
- **주요 라이브러리**:
  - **데이터 로드 및 조작**: `scikit-learn` (`datasets`, `model_selection`, `decomposition`), `pandas`, `numpy`
  - **데이터 시각화 및 결측치 탐색**: `matplotlib`, `seaborn`, `missingno`

---

## 데이터셋별 분석 및 전처리 워크플로우

### 1. Wine 데이터셋 (`sklearn.datasets.load_wine`)
와인의 화학적 성분 데이터를 기반으로 3가지 품종(`class_0`, `class_1`, `class_2`, 총 178개 샘플·13개 특성)을 분류하기 위한 탐색 및 차원 축소 과정입니다.

- **데이터 기초 탐색 및 품질 점검**
  - `df.dtypes`, `df.info()`를 통한 자료형 및 결측치 유무 확인 (결측치 없음)
  - `df.describe()`로 변수 간 스케일 차이 확인 (`malic_acid` 등 0~1대 변수와 수백~수천 단위의 `proline` 간 편차 확인)
  - `df['target'].value_counts()`로 클래스 균형 확인 (`class_1`: 71개, `class_0`: 59개, `class_2`: 48개)
- **상관관계 및 분포 시각화**
  - **히트맵 (`sns.heatmap`)**: 화학 성분 간 상관관계 파악 (`flavanoids`와 `total_phenols` 간 강한 양의 상관관계 확인)
  - **박스플롯 (`sns.boxplot`)**: 품종 구분에 핵심적인 성분(`flavanoids`, `od280/od315_of_diluted_wines`, `proline`, `total_phenols`)과 감각적 지표(`alcohol`, `color_intensity`, `hue`, `malic_acid`)의 클래스별 분포 비교
  - **페어플롯 (`sns.pairplot`)**: 주요 변수 간 산점도 및 군집 형성 양상 확인 (`class_0` > `class_1` > `class_2` 순의 수치 경향성 및 이상치 탐색)
- **데이터 분할 및 표준화 (Standardization)**
  - `train_test_split(..., test_size=0.2, stratify=y, random_state=42)`로 층화 추출 분할
  - 데이터 누수(Data Leakage) 방지를 위해 **훈련 세트(`x_train`)의 평균과 표준편차만**을 기준으로 훈련/검증 세트를 모두 표준화(`mean=0`, `std=1`)
- **주성분 분석 (PCA)**
  - 표준화된 훈련 데이터를 2차원(`n_components=2`)으로 축소하여 시각화
  - **설명된 분산 비율**: PC1 `35.8%`, PC2 `19.3%` (총 설명력 **55.1%**)
  - **주성분 적재량(Loadings) 분석**: PC1에는 `flavanoids`(0.430), `total_phenols`(0.394), `od280/od315`(0.383)가 크게 기여하며, PC2에는 `color_intensity`(0.524), `alcohol`(0.487)이 주로 반영됨을 확인

---

### 2. Iris 데이터셋 (`sklearn.datasets.load_iris`)
붓꽃의 꽃받침(Sepal)과 꽃잎(Petal) 측정 데이터(총 150개 샘플·4개 특성, 3개 품종)에 대한 탐색적 시각화, 피처 엔지니어링 및 PCA 실습입니다.

- **데이터 점검 및 분할**
  - `df.info()`, `df.describe()`로 결측치 없음 및 단위(`cm`) 일관성 확인
  - 층화 추출(`test_size=0.2`, `stratify=iris.target`) 후 훈련 세트 기준 표준화 수행
- **시각화 및 이상치 탐색**
  - **페어플롯 (`sns.pairplot`) & 히트맵 (`sns.heatmap`)**: `petal length`와 `petal width` 기준으로 `target 0`(Setosa)이 다른 두 품종과 뚜렷하게 분리됨을 확인
  - **박스플롯 (`sns.boxplot`)**: 4개 수치형 특성의 품종별 사분위수 및 이상치 탐색
- **파생 변수 생성 (Feature Engineering)**
  - **면적 변수**: `petal_area` (꽃잎 길이 × 너비), `sepal_area` (꽃받침 길이 × 너비)
  - **비율 변수**: `petal_ratio` (꽃잎 길이 / 너비), `sepal_ratio` (꽃받침 길이 / 너비)
- **주성분 분석 (PCA)**
  - 2개 주성분으로 축소 시 **설명된 분산 비율**: PC1 `72.7%`, PC2 `23.1%` (총 설명력 **95.7%**)로 원 데이터의 분산을 매우 높은 수준으로 보존함

---

### 3. Titanic 데이터셋 (`seaborn.load_dataset('titanic')`)
타이타닉호 탑승객 데이터(총 891개 샘플·15개 열)의 결측치 패턴 진단 및 기초 정제 과정입니다.

- **결측치 진단 및 시각화**
  - `titanic.info()` 및 `titanic.isnull().sum()`으로 열별 결측치 개수 파악 (`deck` 688개, `age` 177개, `embarked` 2개, `embark_town` 2개)
  - `missingno.bar()`를 활용해 전체 데이터 대비 결측치 비율을 직관적으로 시각화
- **타깃 변수 분포 확인**
  - `titanic['survived'].value_counts()`로 생존 여부 분포 확인 (사망 `0`: 549명, 생존 `1`: 342명)
- **기초 결측치 대체 (Imputation)**
  - `age` 변수의 이상치 영향을 최소화하기 위해 평균(Mean) 대신 **중앙값(`median`)**으로 결측치를 대체하고 `isnull().sum() == 0`으로 정제 완료 검증

---

## 핵심 요약 및 인사이트

1. **스케일링과 데이터 누수 방지**: 변수 간 단위 차이가 큰 데이터(예: Wine의 `proline`)는 PCA 수행 전 표준화가 필수적이며, 반드시 훈련 데이터의 통계량(`mean`, `std`)만을 사용하여 검증 데이터를 변환해야 합니다.
2. **차원 축소 효율 비교**: 4개 변수를 가진 Iris 데이터는 2개 주성분만으로 전체 분산의 **95.7%**를 설명한 반면, 13개 변수를 가진 Wine 데이터는 2개 주성분으로 **55.1%**를 설명하여 데이터 구조에 따른 차원 축소 특성을 비교할 수 있습니다.
3. **결측치 처리 전략**: 수치형 변수에 이상치가 존재할 가능성이 있는 경우(`Titanic`의 `age`), 평균보다 로버스트(Robust)한 중앙값 대체가 유효합니다.

---

## 실행 방법

`missingno` 라이브러리가 설치되어 있지 않은 환경에서는 아래 명령어로 설치합니다.
   ```bash
   pip install missingno
   ```
