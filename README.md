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

-----

## 3 업데이트 기록

  + 2024.12.25: 일부 버그 수정.
  + 2024.12.20: pip 요구 사항 추가.
  + 2024.12.19: 모든 코드 업데이트.
      + 일부 버그 수정.
      + 보다 정확한 환경 요구 사항 제공.
      + 평가를 위한 새 파일 제공.
      + 코드를 사용자 친화적으로 개선.
      + `README.md` 업데이트.
  + 2024.12.12: 논문 링크 업데이트 (arXiv에서 IEEE로).
  + 2024.4.15: 최초 코드 업로드.
