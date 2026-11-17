# Label Correction

- 공개 영상 데이터셋의 라벨 교정 전후 분류 성능과 추정 신뢰도 비교.
- 라벨 교정: **SSR+LSL**, **Co-Correcting**.
- Backbone: 사전학습 **ConvNeXt V2 Tiny**.
- 영상 분류: **Multi-ROI + Aspect Letterbox**, 재학습 손실: **Weighted Cross-Entropy(WCE)**.
- 데이터·모델 가중치·실험 결과 파일은 저장소에 포함하지 않음. 아래 표는 로컬 실험 기록과 별도로 제공된 클라우드 실험 정보를 정리한 결과.

## 실험 기준

| 구분 | Train 라벨 | 모델 선택 | Test 라벨 |
|---|---|---|---|
| 공통 Baseline | 원본 | 원본 Val Macro F1 최고 | 원본 |
| SSR+LSL Teacher | SSR+LSL로 학습 | 원본 Val Macro F1 최고 | 원본 |
| Co-Correcting Teacher | Co-Correcting으로 학습 | Fine-tuning 종료 모델 | 원본 |
| SSR+LSL Student | SSR 교정 | SSR 교정 Val Macro F1 최고 | SSR 교정 |
| Co-Correcting Student | Co 교정 | Co 교정 Val Macro F1 최고 | Co 교정 |

- 학습·검증·테스트 **전체 분할의 라벨을 교정**하고, 영상과 데이터 분할은 유지.
- Student는 Teacher의 교정 라벨을 고정하고 사전학습 가중치에서 새롭게 학습.
- Baseline은 두 방법이 공유하는 **동일한 실행 한 개**.
- 학습: 무작위 ROI / 검증: 고정 center ROI / 최종 Test: **27개 ROI의 raw logits 평균**.
- Co Teacher: 각 네트워크의 ROI logits와 두 네트워크의 logits를 평균.
- Co Teacher 전 학습 구간에 클래스 가중치 적용. 라벨 갱신과 Small-Loss 선별 계산은 유지.
- 원본 Test와 교정 Test는 평가 라벨도 다르므로, 결과 차이를 동일한 정답 기준의 성능 향상으로 해석하지 않음.

## 분류 결과

- 지표 단위: **%**. Test 영상 수: **6,994개**.
- Epoch는 1부터 세는 완료 epoch 기준.

| 모델 | Test 라벨 | Accuracy | Balanced Accuracy | Macro F1 | 선택 epoch |
|---|---|---:|---:|---:|---:|
| Baseline · Multi-ROI + WCE | 원본 | 80.2545 | 81.6021 | 81.7331 | 179 |
| SSR+LSL Teacher | 원본 | 75.3646 | 77.8468 | 76.6055 | 264 |
| SSR+LSL Student · Multi-ROI + WCE | SSR 교정 | 86.4598 | 87.5680 | 87.7218 | 205 |
| Co-Correcting Teacher | 원본 | 76.1367 | 78.8594 | 77.5565 | 200 |
| Co-Correcting Student · Multi-ROI + WCE | Co 교정 | 87.9325 | 87.8881 | 88.2354 | 222 |

- 두 Student 모두 각 방법의 교정 Val Macro F1을 기준으로 모델 선택.
- Co Student의 모델 선택 기준과 측정값은 별도로 수행한 클라우드 실험에 대해 제공된 정보를 반영.
- SSR Teacher 27-ROI 수치는 제공된 평가 로그를 반영. 로컬 center 결과·epoch·Val 선택 점수와 대조했으며, 원본 27-ROI 평가 파일은 로컬에 없음.
- 라벨 생성 모델은 분류 결과의 선택 모델과 구분: **SSR은 Train total loss 최저 epoch 267**, **Co는 correction_end epoch 150**.

## 데이터셋

- AI-Hub 반려동물 피부질환 데이터를 기반으로 구성한 **7개 클래스, 46,657개 영상**.
- Train **32,661개** / Validation **7,002개** / Test **6,994개**.

### 원본 클래스 분포

| 코드 | 클래스 | Train | Validation | Test | 전체 |
|---|---|---:|---:|---:|---:|
| A1 | 구진·플라크 | 6,366 | 1,365 | 1,363 | 9,094 |
| A2 | 비듬·각질·상피성 잔고리 | 6,589 | 1,412 | 1,411 | 9,412 |
| A3 | 태선화·과다색소침착 | 6,482 | 1,390 | 1,388 | 9,260 |
| A4 | 농포·여드름 | 2,462 | 528 | 527 | 3,517 |
| A5 | 미란·궤양 | 2,400 | 515 | 514 | 3,429 |
| A6 | 결절·종괴 | 3,706 | 794 | 794 | 5,294 |
| A7 | 정상 | 4,656 | 998 | 997 | 6,651 |
| **전체** | | **32,661** | **7,002** | **6,994** | **46,657** |

### 라벨 변경 수

| 분할 | 영상 수 | SSR+LSL 변경 수 | 변경률 | Co-Correcting 변경 수 | 변경률 |
|---|---:|---:|---:|---:|---:|
| Train | 32,661 | 3,786 | 11.59% | 4,588 | 14.05% |
| Validation | 7,002 | 825 | 11.78% | 1,683 | 24.04% |
| Test | 6,994 | 802 | 11.47% | 1,746 | 24.96% |
| **전체** | **46,657** | **5,413** | **11.60%** | **8,017** | **17.18%** |

### 교정 전후 클래스 분포

- Train·Validation·Test 합산. 괄호는 원본 대비 **클래스별 순증감**.
- 클래스별 순증감과 라벨이 변경된 영상 수는 서로 다른 집계.

| 클래스 | 원본 | SSR+LSL 교정 후 | Co-Correcting 교정 후 |
|---|---:|---:|---:|
| 구진·플라크 | 9,094 | 8,038 (−1,056) | 9,567 (+473) |
| 비듬·각질·상피성 잔고리 | 9,412 | 8,729 (−683) | 7,848 (−1,564) |
| 태선화·과다색소침착 | 9,260 | 10,132 (+872) | 8,635 (−625) |
| 농포·여드름 | 3,517 | 4,026 (+509) | 4,293 (+776) |
| 미란·궤양 | 3,429 | 3,958 (+529) | 3,853 (+424) |
| 결절·종괴 | 5,294 | 5,007 (−287) | 4,880 (−414) |
| 정상 | 6,651 | 6,767 (+116) | 7,581 (+930) |
| **전체** | **46,657** | **46,657** | **46,657** |

## 라벨 신뢰도

| 라벨 | CHI ↑ | CLE ↓ | Data Credibility ↑ |
|---|---:|---:|---:|
| 원본 | 396.2795 | 1.796033 | 0.788103 |
| SSR+LSL 교정 | 466.0164 | 1.762940 | 0.814796 |
| Co-Correcting 교정 | 505.9787 | 1.680174 | 0.814727 |

- 전체 **46,657개 영상**을 통합하여 평가.
- 고정된 사전학습 ConvNeXt V2 Tiny Encoder에서 추출한 **동일한 특징** 사용.
- CLE: Cosine Similarity threshold `0.74`, 최소 Clique 크기 `11`, Entropy 단위 `bits`.
- 특징 공간의 라벨 일관성과 추정 신뢰도 지표이며, 전문가 검수로 확인한 정답 정확도를 의미하지 않음.

## 주요 설정

| 항목 | SSR+LSL | Co-Correcting |
|---|---|---|
| Teacher epochs | 300 | 200: Warmup 10 / Correction 140 / Fine-tuning 50 |
| Teacher seed | 0 | 두 네트워크 0, 1 |
| 교정 설정 | k=200, τ=0.7, Reverse k=20 | Forget rate=0.1, α=0.2, λ=500 |
| Student / Baseline | 300 epochs, batch 256, WCE | 300 epochs, batch 256, WCE |

- 전체 설정: [SSR](configs/ssr.json), [Co-Correcting](configs/co-correcting.json).

## 코드·결과 구조

```text
.
├── run.py                  # 실행 순서와 명령 옵션
├── protocol.py             # 라벨 기준·출처·결과 재사용 검사
├── configs/                # 실험 설정
├── ssr/                    # SSR+LSL 학습·교정·평가
├── cocorrecting/           # Co-Correcting 학습·교정·평가
├── shared/                 # 데이터·ROI·지표·저장 공통 처리
├── dataset_quality/        # CHI·CLE·Data Credibility
└── output/                 # 로컬 전용: Git 제외
    ├── baseline/
    ├── ssr/
    │   ├── teacher/
    │   └── student/
    └── co/
        ├── teacher/
        └── student/
```

- 각 단계: `config.json` 설정 / `metrics.jsonl` 학습 지표 / `evaluation.json` 최종 평가.
- 대표 분류 지표: `evaluation.json`의 `primary_metrics`.
- Teacher의 `corrections/`: Train·Val·Test 교정 NPZ 및 `report.json`.
- 원본·교정 라벨 신뢰도: Baseline 및 Teacher의 `label_quality.json`.
- 기존 라벨 재사용 시 세 분할의 교정 파일과 Teacher 출처가 모두 일치해야 함.
- `tests/`, `test/`, `output/`, 데이터·가중치·로그·캐시·보관본은 `.gitignore`로 제외.
- 결과 표 출처: 로컬 평가 JSON·교정 보고서·신뢰도 지표 기록 및 별도로 제공된 클라우드 실험 정보.

## References

- **SSR** — [BMVC 2022 논문](https://bmvc2022.mpi-inf.mpg.de/0372.pdf) · [저자 코드](https://github.com/MrChenFeng/SSR_BMVC2022)
- **LSL** — [CVPR 2024 논문](https://openaccess.thecvf.com/content/CVPR2024/html/Kim_Learning_with_Structural_Labels_for_Learning_with_Noisy_Labels_CVPR_2024_paper.html)
- **Co-Correcting** — [IEEE TMI 2021 논문](https://doi.org/10.1109/TMI.2021.3091178) · [arXiv](https://arxiv.org/abs/2109.05159) · [저자 코드](https://github.com/JiarunLiu/Co-Correcting)
- 원 저작권·라이선스: [MIT LICENSE](LICENSE) · [Apache 2.0](LICENSE-APACHE-2.0) · [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
