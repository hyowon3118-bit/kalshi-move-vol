# Kalshi 분포 불확실성 vs MOVE

가설: Kalshi 예측시장 확률분포의 표준편차·엔트로피가 MOVE(미국채 내재변동성)와 양(+)의 상관을 갖는다.
지표: CPI, 근원 CPI, GDP, 고용, 실업률 (Kalshi KXCPI, KXCPICORE, KXGDP, KXPAYROLLS, KXU3)

## 실행 순서
| 단계 | 명령 | 출력 |
|---|---|---|
| 준비 | `pip install -r requirements.txt` | |
| 1 입력 | `python src/01_inputs.py` | `data/input/` (fixers26-1research의 정제된 Kalshi 분포 복사, MOVE 복사 또는 yfinance 다운로드) |
| 2 패널 | `python src/02_daily_panel.py` | `data/daily_panel.parquet` |
| 3 분석(MOVE) | `python src/03_correlations.py --target MOVE` | `outputs/tables/MOVE_corr_*.csv`, `outputs/figures/MOVE_*.png` |
| 4 국채선물 수집 | `python src/04_databento_treasury.py --cost-only` → 확인 후 `python src/04_databento_treasury.py` | `data/raw/glbx_treasury_ohlcv1m.dbn.zst` |
| 5 실현변동성 | `python src/05_treasury_rv.py` | `data/treasury_rv_daily.csv` |
| 6 분석(국채선물) | `python src/03_correlations.py --target ZT` (2년물), `--target ZN` (10년물) | `outputs/tables/ZT_corr_*.csv` 등 |

Databento: `.env`에 `DATABENTO_API_KEY=db-...` (`.env.example` 참고). 4단계는 비용 조회(과금 없음) 후 `config.yaml`의 `budget_usd` 이하이고 파일이 아직 없을 때만 다운로드를 1회 호출해

입력 경로는 `config.yaml`의 `source_repo` (기본 `../fixers26-1research`, 그 repo에서 `src/clean/01_kalshi_distributions.py`까지 실행돼 있어야 함)

## 변수 (※ = 레퍼런스 없는 내 선택)
- `sd`: 분포 표준편차 (연준 Diercks–Katz–Wright 방식 적률)
- `H`: 섀넌 엔트로피 −Σ p ln p, `H_norm`: H / ln(구간 수) ※
- 지표·날짜마다 만기가 가장 가까운 이벤트 하나 ※, MOVE 거래일만
- `step`: 기준선 간격 중앙값 (계약 설계 변경 통제) ※

## 국채선물 실현변동성 (05_treasury_rv.py)
- CME Globex(GLBX.MDP3) 1분봉, 거래량 기준 연속 근월물 ZT.v.0(2년), ZN.v.0(10년)
- 5분 로그수익률 제곱합 (Andersen–Bollerslev–Diebold–Labys 2003 정의, 5분 간격은 Liu–Patton–Sheppard 2015)
- 거래일 = 시카고 시간 전날 17:00~당일 16:00 ※, 장 휴식·주말 갭 수익률 제외 ※
- 근월물 교체 순간 수익률 제외 (원가격이라 교체 시 가짜 점프)
- 거래 없는 5분 구간 50% 초과인 날 결측 ※, 연율화 100·√(252·RV) ※

## 분석
- A 수준 상관 (참고용, 추세 때문에 가성 상관 가능), B 남은 일수 통제
- C 일별 변화 상관 (주 결과, 같은 이벤트 안 변화만, Newey–West HAC)
- D 이벤트 단위: 발표 h일 전(7, 1) 분포 vs 그날 변동성, 기준선 간격·시간 추세 통제 편상관
- E (국채선물만) 발표 h일 전 분포 → 발표 당일 실현변동성, 평상시 변동성(직전 5거래일)·간격·추세 통제
- 다중검정 Benjamini–Hochberg

## 가설 검증 (src/hyp, 2026-11-09)
실행: `python src/01_inputs.py` (event_panel·RQ1 예측 복사) → `python src/hyp/00_event_windows.py` → `python src/hyp/run_all.py`
발표 시각 8:30 ET 전후 국채선물 수익률: pre 8:00→8:30, post 8:30→8:50, win 8:20→8:50 ※

| 파일 | 가설 | 근거 |
|---|---|---|
| hA_surprise_size.py | 서프라이즈가 클수록 발표 구간 선물 변동 큼 | Balduzzi, Elton & Green (2001) |
| hB_pre_drift.py | 발표 전 30분 드리프트 (서프라이즈·Kalshi 괴리 방향) | Kurov et al. (2019), Kalshi 확장 ※ |
| hC_priced_in.py | 발표 구간 반응 = Kalshi 예측분(c1) + 예측 못 한 부분(c2) | Campbell & Sharpe 분해의 분 단위 적용 ※ |
| hD_uncertainty_interaction.py | 분포가 넓을수록 서프라이즈 반응 큼 | Pasquariello & Vega (2007) 간접 ※ |
| hE_vol_resolution.py | 분포가 넓을수록 발표 후 변동성 더 하락 | Beber & Brandt (2009)의 실현변동성 버전 ※ |
| hF_skew_direction.py | 분포 왜도 방향으로 발표치 | 선행연구 없음 ※ |

주 검정 16개를 한꺼번에 Benjamini–Hochberg 보정 (`outputs/tables/hyp_summary.csv`)

## 꼬리·비매끄러움 가설 (src/tails, 2026-11-09)
실행: `python src/tails/00_tail_features.py` → `python src/hyp/00_event_windows.py` → `python src/tails/run_all_tails.py`
특징량 정의는 `00_tail_features.py` 머리말. 가설 T1~T11은 파일별 머리말에 근거·모형·주 검정
주 검정 21개(사전 고정) 한 묶음 BH, 보조·추가 종속변수(ES·SR3)는 별도 BH (`tail_summary_main.csv`, `tail_summary_secondary.csv`)

## Goel (2026) 재현 (src/replication/goel2026_iqr.py)
Goel, "Measuring macroeconomic surprise magnitude with prediction markets", IREF 110 (2026) 105577
헤드라인 CPI |발표치−컨센서스| ~ Kalshi 내재 IQR, HC1 + wild-bootstrap-t + 순열 p. 정규화·시점·이산 IQR·통합·기간 변형 비교
