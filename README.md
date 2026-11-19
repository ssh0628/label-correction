# Label Correction

라벨 교정 전후 분류 성능과 추정 신뢰도 비교

- **Label Correction**: SSR+LSL, Co-Correcting
- **Backbone**: 사전학습 ConvNeXt V2 Tiny
- **Training Method**: [Multi-ROI](https://github.com/ssh0628/Multi-ROI)

## Experiment

| Model | Label | Model Selection |
|---|---|---|
| Baseline | Original | Original Val<br>Macro F1 |
| SSR+LSL<br>Teacher | Original | - |
| Co-Correcting<br>Teacher | Original | - |
| SSR+LSL<br>Student | Corrected Label<br>(SSR+LSL) | Corrected Val<br>Macro F1 |
| Co-Correcting<br>Student | Corrected Label<br>(Co-Correcting) | Corrected Val<br>Macro F1 |

- 전체 데이터셋 라벨 교정
- Student는 Teacher의 교정 라벨로 사전학습 가중치에서 새롭게 학습
- Co-Correcting의 검증·테스트 라벨은 Teacher를 고정한 상태에서 두 모델 손실의 라벨 Gradient를 합산하여 최적화

## Classification Results

아래 Co-Correcting 결과는 기존 확률 기반 검증·테스트 교정 실험의 기록이며, 변경된 라벨 최적화 절차의 결과는 포함하지 않는다.

| Model | Test Label | Accuracy | Balanced<br>Accuracy | Macro F1 | Epoch |
|---|---|---:|---:|---:|---:|
| Baseline | Original | 80.2545 | 81.6021 | 81.7331 | 179 |
| SSR+LSL<br>Teacher | Original | 75.3646 | 77.8468 | 76.6055 | 264 |
| Co-Correcting<br>Teacher | Original | 76.1367 | 78.8594 | 77.5565 | 200 |
| SSR+LSL<br>Student | Corrected<br>(SSR+LSL) | 86.4598 | 87.5680 | 87.7218 | 205 |
| Co-Correcting<br>Student | Corrected<br>(Co-Correcting) | 87.9325 | 87.8881 | 88.2354 | 222 |

- Co Student의 모델 선택 기준과 측정값은 별도로 수행한 클라우드 실험에 대해 제공된 정보를 반영.

## Dataset

- 반려동물 피부질환 데이터 ([AI-Hub 원본](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=561)), **7개 클래스, 46,657개 영상**.
- Train **32,661개** / Validation **7,002개** / Test **6,994개**.

### Original Class Distribution

| Code | Class | Train | Validation | Test | Total |
|---|---|---:|---:|---:|---:|
| A1 | 구진·플라크 | 6,366 | 1,365 | 1,363 | 9,094 |
| A2 | 비듬·각질·<br>상피성 잔고리 | 6,589 | 1,412 | 1,411 | 9,412 |
| A3 | 태선화·<br>과다색소침착 | 6,482 | 1,390 | 1,388 | 9,260 |
| A4 | 농포·여드름 | 2,462 | 528 | 527 | 3,517 |
| A5 | 미란·궤양 | 2,400 | 515 | 514 | 3,429 |
| A6 | 결절·종괴 | 3,706 | 794 | 794 | 5,294 |
| A7 | 정상 | 4,656 | 998 | 997 | 6,651 |
| **Total** | | **32,661** | **7,002** | **6,994** | **46,657** |

### Label Changes

| Split | Images | SSR+LSL<br>Changes | SSR+LSL<br>Change Rate | Co-Correcting<br>Changes | Co-Correcting<br>Change Rate |
|---|---:|---:|---:|---:|---:|
| Train | 32,661 | 3,786 | 11.59% | 4,588 | 14.05% |
| Validation | 7,002 | 825 | 11.78% | 1,683 | 24.04% |
| Test | 6,994 | 802 | 11.47% | 1,746 | 24.96% |
| **Total** | **46,657** | **5,413** | **11.60%** | **8,017** | **17.18%** |

### Class Distribution Changes

- Train·Validation·Test 합산. 괄호는 원본 대비 **클래스별 순증감**.
- 클래스별 순증감과 라벨이 변경된 영상 수는 서로 다른 집계.

| Class | Original | SSR+LSL<br>Corrected | Co-Correcting<br>Corrected |
|---|---:|---:|---:|
| 구진·플라크 | 9,094 | 8,038 (−1,056) | 9,567 (+473) |
| 비듬·각질·<br>상피성 잔고리 | 9,412 | 8,729 (−683) | 7,848 (−1,564) |
| 태선화·<br>과다색소침착 | 9,260 | 10,132 (+872) | 8,635 (−625) |
| 농포·여드름 | 3,517 | 4,026 (+509) | 4,293 (+776) |
| 미란·궤양 | 3,429 | 3,958 (+529) | 3,853 (+424) |
| 결절·종괴 | 5,294 | 5,007 (−287) | 4,880 (−414) |
| 정상 | 6,651 | 6,767 (+116) | 7,581 (+930) |
| **Total** | **46,657** | **46,657** | **46,657** |

## Label Reliability

| Method | CHI ↑ | Clique Label<br>Entropy ↓ | Data<br>Credibility ↑ |
|---|---:|---:|---:|
| Baseline | 396.2795 | 1.796033 | 0.788103 |
| SSR+LSL | 466.0164 | 1.762940 | 0.814796 |
| Co-Correcting | 505.9787 | 1.680174 | 0.814727 |

- Clique Label Entropy: Cosine Similarity threshold `0.74`, 최소 Clique 크기 `11`.

## Hyperparameters

| Parameter | SSR+LSL | Co-Correcting |
|---|---|---|
| Teacher Epochs | 300 | 200<br>Warmup 10 / Correction 140 / Fine-tuning 50 |
| Teacher Seed | 0 | 두 네트워크 0, 1 |
| Correction Settings | k=200, τ=0.7<br>Reverse k=20 | Forget rate=0.1<br>α=0.2, λ=500 |
| Validation / Test Label Updates | Confidence-based relabelling | Frozen Teacher<br>30 label-update epochs |
| Student / Baseline | 300 epochs, batch 256, WCE | 300 epochs, batch 256, WCE |

- 전체 설정: [SSR](configs/ssr.json), [Co-Correcting](configs/co-correcting.json).

## Project Structure

```text
.
├── run.py                  # 실행 순서와 명령 옵션
├── protocol.py             # 라벨 기준·출처·결과 재사용 검사
├── configs/                # 실험 설정
├── ssr/                    # SSR+LSL 학습·교정·평가
├── cocorrecting/           # Co-Correcting 학습·교정·평가
├── shared/                 # 데이터·ROI·지표·저장 공통 처리
├── dataset_quality/        # 신뢰도 지표 계산
└── output/                 # 로컬 전용: Git 제외
    ├── baseline/
    ├── ssr/
    │   ├── teacher/
    │   └── student/
    └── co/
        ├── teacher/
        └── student/
```

## References

- **SSR** — [BMVC 2022 논문](https://arxiv.org/abs/2111.11288) · [저자 코드](https://github.com/MrChenFeng/SSR_BMVC2022)
- **LSL** — [CVPR 2024 논문](https://openaccess.thecvf.com/content/CVPR2024/html/Kim_Learning_with_Structural_Labels_for_Learning_with_Noisy_Labels_CVPR_2024_paper.html)
- **Co-Correcting** — [IEEE TMI 2021 논문](https://doi.org/10.1109/TMI.2021.3091178) · [arXiv](https://arxiv.org/abs/2109.05159) · [저자 코드](https://github.com/JiarunLiu/Co-Correcting)
- 원 저작권·라이선스: [LICENSE](LICENSE)
