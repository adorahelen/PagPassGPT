# PagPassGPT

논문 코드: [PagPassGPT: 생성형 사전 훈련된 트랜스포머를 통한 패턴 기반 암호 추측](https://www.computer.org/csdl/proceedings-article/dsn/2024/410500a429/1ZPxTMt2ao8) (DSN 2024)

-----

## 1 환경 설정

```shell
conda create -n env_name python=3.8.10
conda activate env_name
pip install -r requirements.txt
# pip install numpy==1.24.2 huggingface-hub==0.13.4 fsspec==2022.11.0 torch==2.0.0 transformers==4.29.0 datasets==2.12.0 accelerate==0.17.1
```

- 매번 가상환경을 실행하고 코드를 실행해야함 ($conda activate env_name, 여기서 env_name은 사용자가 지정 가능 ex: CIS)
- requirements.txt 설치시 가상환경 내부에서 해야 버전 오류 발생안함

-----

## 2 사용법

### 2.1 데이터셋 준비

1.  \*\*"RockYou"\*\*와 같은 암호 데이터셋을 준비해야 합니다. 데이터셋에는 **암호만** 포함되어 있는지 확인해야 합니다.

2.  `preprocess.sh` 스크립트를 실행하여 데이터셋을 전처리합니다.

<!-- end list -->

```shell
nohup sh ./scripts/preprocess.sh > preprocess.log 2>&1 &
```
*Note: 전처리 중 발생하는 모든 출력은 **`preprocess.log`** 파일에 기록됩니다.*

*Note: **"RockYou"** 데이터셋 [다운로드 링크](https://www.google.com/url?sa=i&url=https%3A%2F%2Fgithub.com%2Fbrannondorsey%2Fnaive-hashcat%2Freleases%2Fdownload%2Fdata%2Frockyou.txt&psig=AOvVaw3rovncwk_ZO-AVgMK56N5-&ust=1734701481601000&source=images&cd=vfe&opi=89978449&ved=0CAYQrpoMahcKEwiwiazf-LOKAxUAAAAAHQAAAAAQBA)가 여기에 있습니다.*

- 해당 데이터셋은 100MB 넘기에, upload 하지 않음

- 전처리 결과는 아래와 같음 
---

## 전처리 결과, 데이터셋 차이 분석

| 파일명 | 크기 (MB) | 예상 용도 및 차이점 |
| :--- | :--- | :--- |
| **`rockyou.txt`** | **139.9 MB** | **원본 데이터**: 다운로드한 **가장 가공되지 않은** (Raw) 상태의 RockYou 암호 목록 파일입니다. 이 파일이 100MB를 초과하여 Git 푸시 오류를 일으켰던 원인입니다. |
| **`rockyou-cleaned.txt`** | **123.1 MB** | **일반 전처리 완료**: `rockyou.txt`에서 중복되거나 유효하지 않은 항목이 제거되는 등 **일반적인 정제(Cleaning)** 과정을 거친 파일입니다. 원본보다 크기가 줄어든 것을 확인할 수 있습니다. |
| **`rockyou-cleaned-Train.txt`** | **98.5 MB** | **훈련 데이터셋 (Train Set)**: `rockyou-cleaned.txt`에서 모델 훈련을 위해 **분리된 데이터셋**입니다. 이 파일을 기반으로 모델의 학습이 이루어집니다. |
| **`rockyou-cleaned-Test.txt`** | **24.6 MB** | **평가 데이터셋 (Test Set)**: 훈련에 사용되지 않고 모델의 성능을 평가하기 위해 **분리된 데이터셋**입니다. |
| **`rockyou-cleaned-Train-ready.txt`** | **100.6 MB** | **최종 훈련 준비 데이터**: `rockyou-cleaned-Train.txt`를 모델이 즉시 읽고 훈련할 수 있도록 **특정 형식으로 최종 가공**한 파일입니다. (예: 토큰화, 인코딩, 패턴 삽입 등의 후처리). `rockyou-cleaned-Train.txt`보다 크기가 약간 증가한 것은 이 최종 가공 과정에서 추가적인 정보가 포함되었기 때문일 수 있습니다. |

---



### 2.2 PagPassGPT 훈련

훈련을 시작하고, 터미널 세션이 끊어져도 작업이 유지되도록 **백그라운드에서 실행**하며 로그를 기록합니다.
- export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 등으로 그레픽 카드 지정 (default 전부)

<img width="649" height="629" alt="image" src="https://github.com/user-attachments/assets/a2c68f04-c7c3-4831-86ed-6efab67d769d" />

- $watch nvidia-smi (실시간), nvidia-smi (단편)

```shell
nohup sh ./scripts/train.sh > train.log 2>&1 &
```

*Note: 훈련 중 발생하는 모든 출력은 **`train.log`** 파일에 기록됩니다.*

#### [모델 아키텍처 및 설정]
- 모델 종류: GPT2LMHeadModel (GPT-2 기반 언어 모델) 로드
- 파라미터 수: 총 21,358,464개 (약 2,136만 개)의 파라미터를 가진 경량화된 모델
- 주요 구조: 12개 트랜스포머 블록, 은닉 차원 384로 구성

#### [실행 및 진행 상태]
- 총 단계: 전체 훈련은 77,730 스텝으로 계획 (28시간 소요 예상)

### 2.3 암호 생성

암호를 생성하고, 터미널 세션이 끊어져도 작업이 유지되도록 **백그라운드에서 실행**하며 로그를 기록합니다.

```shell
nohup sh ./scripts/generate.sh > generate.log 2>&1 &
```

*Note: 이 셸에서 단 한 줄만 변경하여 **D\&C-GEN** 사용 여부를 선택할 수 있습니다. 생성 과정의 출력은 **`generate.log`** 파일을 확인하세요.*

### 2.4 암호 평가

암호 평가를 실행하고, 터미널 세션이 끊어져도 작업이 유지되도록 **백그라운드에서 실행**하며 로그를 기록합니다.

```shell
nohup sh ./scripts/evaluate.sh > evaluate.log 2>&1 &
```

*Note: 평가는 주로 \*\*적중률(Hit rate)\*\*과 \*\*반복률(Repeat rate)\**에 중점을 둡니다. 
  * 평가 결과는 **`evaluate.log`** 파일에 기록됩니다.*

-----

## 💡 백그라운드 작업 및 로그 확인 팁

  * **실행 중인 작업 확인:** `jobs` 명령어를 사용하거나 `ps -ef | grep train.sh` 등을 통해 현재 백그라운드에서 실행 중인 프로세스를 확인할 수 있습니다.
  * **로그 실시간 확인:** `tail -f [로그 파일명].log` 명령어를 사용하면 파일에 내용이 추가될 때마다 실시간으로 출력되는 로그를 볼 수 있습니다. (예: `tail -f train.log`)

---

## 3 업데이트 기록

* 2024.12.25: 일부 버그 수정.
* 2024.12.20: pip 요구 사항 추가.
* 2024.12.19: 모든 코드 업데이트.
* 일부 버그 수정.
* 보다 정확한 환경 요구 사항 제공.
* 평가를 위한 새 파일 제공.
* 코드를 사용자 친화적으로 개선.
* `README.md` 업데이트.


* 2024.12.12: 논문 링크 업데이트 (arXiv에서 IEEE로).
* 2024.4.15: 최초 코드 업로드.

---

## 4 주요 연구 및 실험 결과

### 4.1 핵심 요약

1. **전이 학습의 효과**: RockYou 데이터셋에서 한국어 데이터셋으로의 전이 학습은 비밀번호 분포 학습에 효과적인 초기화를 제공합니다.
2. **키보드 패턴 의존성**: 한국 사용자 비밀번호는 실제 한글 문자 자체보다 쿼티(QWERTY) 키보드 기반의 영문 입력 패턴에 훨씬 크게 의존합니다.
3. **패턴 표현의 한계**: 단순한 PCFG(Probabilistic Context-Free Grammars) 확장은 한국어 패턴의 복잡성을 충분히 반영하기 어려우며, 결정론적 키보드 변환 규칙만으로는 사용자의 창의적인 변형을 모두 포괄하지 못합니다.

### 4.2 참고 문헌

* [1] Weir, M., et al. (2009). *Password cracking using probabilistic context-free grammars*. IEEE S&P.
* [2] Hitaj, B., et al. (2019). ***PassGAN**: A deep learning approach for password guessing*. NDSS.
* [3] Melicher, W., et al. (2016). *Fast, lean, and accurate: Modeling password guessability using neural networks*. USENIX Security.
* [4] Pasquini, D., et al. (2021). *Improving password guessing via representation learning*. IEEE S&P.
* [5] Nam, J., et al. (2023). *PassGPT: Password modeling and guessing using generative pre-trained transformer*. Computers & Security.
* [6] Yu, S., et al. (2024). *PagPassGPT: Pattern-guided password guessing via generative pretrained transformer*.
* [7] RockYou Leak Dataset. *RockYou password dataset*.

---

## 5 중간 발표 및 데이터 분석

### 5.1 실험 개요

* **대상 Repository**: [PagPassGPT](https://github.com/Suxyuuu/PagPassGPT) 기반 연구 수행.
* **데이터셋 수집**:
* 한국어 기반 데이터셋 수집 및 정제 (Unique 7,463,435개).
* 정제 조건: 중복 제거, 길이 4~12자, Non-ASCII 제거.
* 결과: 약 **461만 개**의 유효 데이터 확보.
* 특이사항: 한글 문자가 직접 포함된 데이터는 극소수(약 10개)이며, 대부분 ASCII 문자로 구성됨.



### 5.2 RockYou 단독 훈련 결과 ( Generation)

| Metric | Normal Gen | DC-Gen |
| --- | --- | --- |
| **Hit Rate** | **0.94%** (0.0093) | **0.99%** (0.0099) |
| **Repeat Rate** | 2.01e-06 | 1.00e-06 |

---

## 6 최종 실험 결과

### 6.1 Fine-tuning (RockYou → 한국어 데이터셋)

GPT-2 모델을 RockYou로 사전 학습(Pre-training)한 후, 한국어 데이터셋으로 미세 조정(Fine-tuning)을 진행했습니다.

| Generation Count | Method | Hit Rate | Repeat Rate |
| --- | --- | --- | --- |
| **** | Normal | 2.39% | 7.10e-06 |
|  | DC-Gen | 2.29% | 4.00e-06 |
| **** | Normal | 14.76% | 2.07e-05 |
|  | DC-Gen | 13.52% | 1.42e-05 |
| **** | Normal | 37.44% | 5.17e-05 |

> **분석:** 기존 한국어 데이터셋 단독 학습 대비 Hit Rate가 개선되었으며, DC-Gen 방식은 Hit Rate를 크게 높이지는 않으나 **Repeat Rate(중복 생성률)를 확실하게 감소**시키는 효과를 보였습니다.

### 6.2 한국어 키보드 패턴 적용 시도

[es-hangul](https://es-hangul.slash.page/) 라이브러리를 활용하여 영문 입력을 한글 타이핑 패턴으로 역산출하여 학습을 시도했습니다.

* **대상 데이터**: 전체 461만 개 중 743,834개에서 한국어 패턴 추출 성공.
* **패턴 예시** (`gksrmf1234a`):
1. **Type 1 (한글 글자수 기준)**: `K2` (한글 2글자) + `N4` + `L1`
2. **Type 2 (알파벳 입력 기준)**: `K6` (알파벳 6글자) + `N4` + `L1`



#### 패턴 학습 결과 ( Gen 기준)

| Pattern Type | Method | Hit Rate | Repeat Rate |
| --- | --- | --- | --- |
| **Pattern Type 1** | Normal | 13.28% | 3.03e-04 |
|  | DC-Gen | 9.33% | 2.85e-05 |
| **Pattern Type 2** | Normal | 13.45% | 3.43e-04 |
|  | DC-Gen | 9.28% | 2.58e-05 |

---

## 7 향후 연구 과제

* 음운 기반 서브워드(Subword) 모델링 도입.
* 다국어 사전학습 모델과의 비교 분석.
* 보다 유연한 한국어 특화 패턴 표현 방식 탐구.

---
