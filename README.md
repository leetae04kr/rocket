# Rocket Atmosphere ML Data Engineering Pipeline

LL2(Launch Library 2) 궤도 발사 이력 데이터와 ECMWF ERA5 재분석 기상 데이터를 연동하여, **발사체 제원 및 대기 환경 조건 기반 카르만 라인 도달/발사 성공 여부(`launch_status: 1 or 0`)** 모델 학습용 대용량 데이터셋을 구축하는 파이썬 ETL 파이프라인입니다.

---

## 1. 아키텍처 및 핵심 엔지니어링 설계

```
 ┌──────────────────────┐          ┌──────────────────────────┐
 │    Launch Library 2  │          │   ECMWF ERA5 Reanalysis  │
 │     REST API (LL2)   │          │      (Single / Pressure) │
 └──────────┬───────────┘          └────────────┬─────────────┘
            │ Pagination + Rate Limit Retry      │ Monthly Batch NetCDF
            ▼                                    ▼
   [ Raw LL2 Parquet ]               [ Cached NetCDF Grids ]
            │                                    │
            └─────────────────┬──────────────────┘
                              │
                              ▼
            ┌────────────────────────────────────┐
            │  Xarray Spatio-Temporal Joiner     │
            │  - Nearest-Neighbor Grid Sel       │
            │  - Vectorized Atmospheric Physics  │
            └─────────────────┬──────────────────┘
                              │
                              ▼
            ┌────────────────────────────────────┐
            │     Streaming CSV Chunk Writer     │
            │     - Yearly partitioned CSVs      │
            │     - Master combined dataset      │
            └────────────────────────────────────┘
```

1. **LL2 대용량 수집 & 지능형 재시도 (Mass Extraction & Rate-Limit Handling)**
   - `requests.adapters.HTTPAdapter` 및 `urllib3.util.retry.Retry`를 적용하여 429(Too Many Requests), 500, 502, 503, 504 에러 시 지수 백오프(Exponential Backoff) 및 `Retry-After` 헤더 기반 지연 처리.
   - `offset`과 `limit`을 활용한 자동 페이지네이션으로 7,000건 이상의 전역 과거 발사 이력 전수 수집.
   - 중간 수집 결과는 `data/raw_ll2/`에 Parquet 형태로 자동 캐싱되어 중단 시 재시작 가능.

2. **ERA5 대용량 배치 요청 & NetCDF 캐싱 (Batch CDS API)**
   - CDS API는 개별 포인트 단위 요청 시 대기 큐 시간이 극도로 길어지므로, 발사 시각(UTC)을 **(연도, 월)** 단위로 그루핑하여 표면(Single-levels) 및 고층(Pressure-levels) 넷CDF 파일로 일괄 요청.
   - 이미 로컬에 캐시된 파일이 있으면 네트워크 호출을 즉시 스킵하는 스마트 캐시 메커니즘.

3. **시공간 최단 정합 & 벡터화 특성 공학 (Spatio-Temporal Nearest Join & Vectorized Physics)**
   - `xarray.sel(time=..., latitude=..., longitude=..., method='nearest')`을 통해 발사 시각 및 발사대 좌표와 가장 가까운 격자 포인트 추출.
   - **표면 특성**:
     - 합성 풍속: $\sqrt{u_{10}^2 + v_{10}^2}$
     - 상대습도(%): August-Roche-Magnus 공식을 활용하여 $T_{2m}$과 $D_{2m}$으로부터 산출.
     - 시정 거리(km): 상대습도와 강수량($tp$) 기반 Koschmieder 대기 소광 모델 근사.
   - **고층 특성**:
     - 지위고도($Z = \Phi / g$) 기반 수직 프로파일 생성.
     - 5km, 10km, 20km 고도별 수평 풍속 선형 보간.
     - 연직 윈드 시어($\max |dV| / dz$, $s^{-1}$): 인접 압력 레벨 간 최대 전단율 산출.
     - 결빙 고도(Freezing Level, km): 지표면 위에서 최초로 0°C에 도달하는 고도 보간.

4. **메모리 누수 방지 스트리밍 (Chunked Streaming & Partitioning)**
   - 연월 배치 단위로 NetCDF 리소스를 컨텍스트 매니저로 즉시 해제하고 `gc.collect()`를 호출하여 대용량 처리 시 OOM(Out of Memory) 방지.
   - 연도별 분할 파일(`launches_weather_{YYYY}.csv`)과 전체 마스터 파일(`combined_rocket_weather_dataset.csv`)에 스트리밍 append.

---

## 2. 데이터셋 스키마 (Dataset Schema)

| 컬럼명 | 데이터 타입 | 단위 / 포맷 | 설명 |
| :--- | :--- | :--- | :--- |
| `launch_id` | String | UUID | LL2 고유 발사 ID |
| `launch_name` | String | Text | 발사 미션명 (예: Falcon 9 Block 5 \| Starlink) |
| `launch_time_utc` | String | ISO 8601 | 발사 시각 (UTC) |
| `pad_name` | String | Text | 발사대 명칭 |
| `latitude` | Float | Decimal Degrees | 발사대 위도 |
| `longitude` | Float | Decimal Degrees | 발사대 경도 |
| `rocket_id` | Integer | ID | LL2 발사체 형상 ID |
| `rocket_name` | String | Text | 발사체 모델명 |
| `rocket_type` | String | Category | Small / Medium / Heavy / Super Heavy |
| `total_mass_kg` | Float | kg | 발사 당시 총중량 (Liftoff Mass) |
| `liftoff_thrust_kn` | Float | kN | 이륙 추력 (Sea Level Thrust) |
| `max_q_limit_kpa` | Float | kPa | 발사체 기체 구조적 최대 동압 허용 한계 |
| `payload_mass_kg` | Float | kg | 적재 페이로드 총 질량 |
| `surface_wind_speed` | Float | m/s | 10m 합성 지상 풍속 ($\sqrt{u_{10}^2 + v_{10}^2}$) |
| `wind_gust` | Float | m/s | 순간 돌풍 속도 (Instantaneous Wind Gust) |
| `surface_temperature` | Float | °C | 지상 2m 기온 |
| `surface_humidity` | Float | % | 지상 2m 상대 습도 (Magnus 식) |
| `precipitation` | Float | mm/hr | 시간당 강수량 |
| `visibility` | Float | km | 시정 거리 |
| `max_wind_shear` | Float | $s^{-1}$ | 대류권~성층권 최대 연직 윈드 시어 ($\Delta V / \Delta z$) |
| `upper_air_wind_speed_5km` | Float | m/s | 고도 5km 상공 풍속 |
| `upper_air_wind_speed_10km` | Float | m/s | 고도 10km 상공 풍속 (Max Q 통과 구간) |
| `upper_air_wind_speed_20km` | Float | m/s | 고도 20km 상공 풍속 |
| `freezing_level_altitude_km` | Float | km | 0°C 결빙 고도 |
| **`launch_status`** | Integer | Binary (0 / 1) | **Target Variable**: 1 (성공/궤도진입), 0 (실패/기상지연/미도달) |

---

## 3. 설치 및 환경 설정

### 1) 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

### 2) ECMWF CDS API 인증키 설정
실제 ERA5 데이터를 ECMWF 서버로부터 직접 다운로드받으려면 CDS API 계정이 필요합니다:
1. [ECMWF CDS 회원가입 및 로그인](https://cds.climate.copernicus.eu/)
2. 프로필 페이지에서 API Key 확인
3. 홈 디렉토리에 `~/.cdsapirc` 파일 생성:
```ini
url: https://cds.climate.copernicus.eu/api
key: <YOUR-CDS-API-KEY>
```
*(참고: 인증키가 없는 오프라인/로컬 테스트 환경에서는 파이프라인이 이를 자동으로 감지하여 표준 대기 물리 모델 기반의 합성 NetCDF를 생성해 무중단 테스트를 지원합니다.)*

---

## 4. 파이프라인 실행 방법

### 기본 실행 (GCAT 전수 데이터: 1957~2024년 7,400+건 전체 수집 및 기상 결합)
Rate Limit 없이 수초 만에 인류 역사상 모든 발사체 기록을 전수 수집합니다:
```bash
python run_pipeline.py
```

### 특정 연도 범위 대용량 수집 (예: 2015~2024년 전수 1,412건)
```bash
python run_pipeline.py --start-year 2015 --end-year 2024
```

### 건수 제한 실행 (테스트용)
```bash
python run_pipeline.py --max-launches 500
```

### 데이터 소스 선택 (`--source`)
- `--source gcat` (기본값): 조나단 맥도웰의 GCAT 전수 아카이브 (1957~2024, 7,431건 전수, Rate Limit 없음, 권장)
- `--source ll2`: Launch Library 2 REST API (호출 제한 준수 필요)

```bash
# LL2 REST API로 수집할 때
python run_pipeline.py --source ll2 --start-year 2015
```

---

## 5. 프로젝트 디렉토리 구조

```
rocket/
├── requirements.txt            # 의존성 패키지 목록
├── run_pipeline.py             # CLI 실행 스크립트
├── pipeline/
│   ├── __init__.py
│   ├── config.py               # 설정값, 상수 및 스키마 정의
│   ├── ll2_client.py           # LL2 API 대용량 수집, Retry, 페이지네이션
│   ├── era5_client.py          # ECMWF CDS API 배치 다운로드 및 캐싱
│   ├── physics.py              # 벡터화된 기상 물리 및 연직 프로파일 계산식
│   ├── joiner.py               # Xarray 시공간 최단 정합 모듈
│   └── pipeline.py             # 전체 파이프라인 오케스트레이터 & CSV 스트리밍
├── tests/
│   └── test_pipeline.py        # 물리 계산식 및 Xarray 정합 단위 테스트
└── data/                       # 실행 시 자동 생성
    ├── raw_ll2/                # LL2 수집 중간 Parquet 캐시
    ├── raw_era5/               # 다운로드된 ERA5 NetCDF 캐시
    └── output/                 # 연도별 및 통합 CSV 데이터셋 저장소
```
