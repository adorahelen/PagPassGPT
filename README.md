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

-----

## 2 사용법

### 2.1 데이터셋 준비

1.  \*\*"RockYou"\*\*와 같은 암호 데이터셋을 준비해야 합니다. 데이터셋에는 **암호만** 포함되어 있는지 확인해야 합니다.

2.  `preprocess.sh` 스크립트를 실행하여 데이터셋을 전처리합니다.

<!-- end list -->

```shell
sh ./scripts/preprocess.sh
```

*Note: **"RockYou"** 데이터셋 [다운로드 링크](https://www.google.com/url?sa=i&url=https%3A%2F%2Fgithub.com%2Fbrannondorsey%2Fnaive-hashcat%2Freleases%2Fdownload%2Fdata%2Frockyou.txt&psig=AOvVaw3rovncwk_ZO-AVgMK56N5-&ust=1734701481601000&source=images&cd=vfe&opi=89978449&ved=0CAYQrpoMahcKEwiwiazf-LOKAxUAAAAAHQAAAAAQBA)가 여기에 있습니다.*

### 2.2 PagPassGPT 훈련

`train.sh` 스크립트를 실행하여 훈련합니다.

```shell
sh ./scripts/train.sh
```

### 2.3 암호 생성

`generate.sh` 스크립트를 실행하여 암호를 생성합니다.

```shell
sh ./scripts/generate.sh
```

*Note: 이 셸에서 단 한 줄만 변경하여 **D\&C-GEN** 사용 여부를 선택할 수 있습니다.*

### 2.4 암호 평가

```shell
sh ./scripts/evaluate.sh
```

\*Note: 평가는 주로 \*\*적중률(Hit rate)\*\*과 \*\*반복률(Repeat rate)\**에 중점을 둡니다.*

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