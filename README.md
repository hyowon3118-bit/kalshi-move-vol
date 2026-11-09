# Kalshi 분포 불확실성 vs MOVE

가설: Kalshi 예측시장 확률분포의 표준편차·엔트로피가 MOVE(미국채 내재변동성)와 양(+)의 상관을 갖는다.
지표: CPI, 근원 CPI, GDP, 고용, 실업률 (Kalshi KXCPI, KXCPICORE, KXGDP, KXPAYROLLS, KXU3)

## 실행 순서
| 단계 | 명령 | 출력 |
|---|---|---|
| 준비 | `pip install -r requirements.txt` | |
| 1 입력 | `python src/01_inputs.py` | `data/input/` (fixers26-1research의 정제된 Kalshi 분포 복사, MOVE 복사 또는 yfinance 다운로드) |
| 2 패널 | `python src/02_daily_panel.py` | `data/daily_panel.parquet` |
| 3 분석 | `python src/03_correlations.py` | `outputs/tables/corr_*.csv`, `outputs/figures/*.png` |

입력 경로는 `config.yaml`의 `source_repo` (기본 `../fixers26-1research`, 그 repo에서 `src/clean/01_kalshi_distributions.py`까지 실행돼 있어야 함)

## 변수 (※ = 레퍼런스 없는 내 선택)
- `sd`: 분포 표준편차 (연준 Diercks–Katz–Wright 방식 적률)
- `H`: 섀넌 엔트로피 −Σ p ln p, `H_norm`: H / ln(구간 수) ※
- 지표·날짜마다 만기가 가장 가까운 이벤트 하나 ※, MOVE 거래일만
- `step`: 기준선 간격 중앙값 (계약 설계 변경 통제) ※

## 분석
- A 수준 상관 (참고용, 추세 때문에 가성 상관 가능), B 남은 일수 통제
- C 일별 변화 상관 (주 결과, 같은 이벤트 안 변화만, Newey–West HAC)
- D 이벤트 단위: 만기 h일 전(7, 1) 분포 vs 그날 MOVE, 기준선 간격·시간 추세 통제 편상관
- 다중검정 Benjamini–Hochberg
