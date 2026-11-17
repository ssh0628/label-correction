# Third-party notices

이 폴더는 Co-Correcting 및 SSR+LSL teacher를 학습하고, 교정 라벨을 내보내 A1–A7 Multi-ROI/WCE student를 재학습합니다.

## Co-Correcting

- 저자/저장소: Jiarun Liu 및 Co-Correcting 기여자, `JiarunLiu/Co-Correcting`.
- 연구: Co-Correcting, IEEE Transactions on Medical Imaging, 2021.
- 원본 저장소: <https://github.com/JiarunLiu/Co-Correcting>
- 라이선스: Apache License 2.0. 전문은 이 폴더의 `LICENSE-APACHE-2.0`에 포함되어 있습니다.

Teacher 알고리즘과 curriculum은 위 구현을 기반으로 수정한 로컬 프로젝트에서 가져왔습니다. 저장된 라벨 점수와 체크포인트의 출처를 검증하고 이를 WCE 재학습에 사용합니다.

## 로컬 프로젝트에서 복사·수정한 공통 코드

데이터/NPY 처리, bbox, ROI, 이미지 읽기, augmentation, precision 및 metrics 공통 코드는 이 작업 공간에 있던 기존 구현을 복사·수정했습니다. 그 프로젝트에 포함된 다음 MIT 고지를 보존합니다.

> Copyright (c) 2022 Chen Feng

MIT License 전문과 보증 부인 문구는 `LICENSE`에 포함되어 있습니다. 이 고지는 복사된 코드의 기존 라이선스 고지를 보존하기 위한 것으로, 모든 추가 구현의 원저자가 동일하다는 의미는 아닙니다.

두 Student는 공통 전처리와 분류 지표를 사용합니다. 원본 holdout 평가, 표본 순서·라벨·hash 검증 및 27-view 평가는 로컬 실험 코드에서 가져왔습니다.

## 실행 의존성과 모델 가중치

PyTorch, torchvision, timm, NumPy, Pillow, tqdm 등 설치 패키지는 각각의 라이선스를 따릅니다. `requirements.txt`는 패키지 목록이며 해당 패키지의 소스나 라이선스를 이 고지로 대체하지 않습니다. ConvNeXtV2 사전학습 가중치는 별도 캐시 또는 사용자가 제공한 로컬 파일로 사용하며, 그 가중치의 배포·사용 조건은 원 배포처의 조건을 따릅니다.

배포 시 해당 코드에 적용되는 라이선스와 고지 파일을 함께 보존하세요. 이 폴더의 변경 사항은 원저자의 승인이나 성능 보증을 뜻하지 않습니다.
