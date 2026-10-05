# Customer Lifetime Value & Churn Engine

A production-style decision-support platform that answers one executive question:

> **Which customers are likely to churn, how much future revenue is at risk, why are they likely to churn, and which customers should the company prioritise for retention?**

This is not a Kaggle churn notebook. It is structured as a customer-intelligence
platform: temporal (leakage-safe) feature engineering, calibrated churn
probabilities, customer lifetime value (CLV), a revenue-at-risk engine, a
retention prioritisation engine, explainability for every individual customer,
and a C-suite dashboard. The success metric is **revenue protected**, not model
accuracy.

---

## 1. Business problem

Customer churn silently destroys recurring and repeat revenue. Management
typically asks five questions:

1. How much revenue is at risk right now?
2. Which customers are most likely to churn?
3. Which churners actually matter financially?
4. Why are they churning?
5. With a limited retention budget, who should we target first?

Most companies answer these with a blunt rule — *"anyone inactive for 90 days is
at risk"* — and a generic discount. That rule:

- targets customers who are already gone (spend on the wrong people);
- ignores customers who are quietly reducing spend but still active;
- ignores customer value, so a £50 customer and a £5,000 customer are treated
  identically;
- cannot explain *why* a customer is at risk.

The platform replaces that rule with a value-aware, explainable, temporally
valid risk engine.

### The core insight

The highest-probability churner is rarely the best retention investment.

| Customer | Churn probability | CLV  | Revenue at risk |
| -------- | ----------------- | ---- | --------------- |
| A        | 90%               | £50  | £45             |
| B        | 55%               | £5,000 | £2,750        |

Prioritisation must combine **probability of churn** with **future value at
risk**, i.e. `Revenue at Risk = P(churn) × Expected future CLV`.

---

## 2. Dataset

Primary source: **UCI Online Retail II** (transaction-level, Dec 2009 – Dec 2011,
~1.07M rows, two yearly sheets).

| Field | Meaning |
| ----- | ------- |
| `Invoice` | Invoice number (`C...` prefix = cancellation) |
| `StockCode` / `Description` | Product identifiers |
| `Quantity` / `Price` | Line quantity and unit price |
| `InvoiceDate` | Timestamp of the transaction |
| `Customer ID` | Customer identifier (missing for some rows) |
| `Country` | Customer country |
| Returns | Negative quantities / `C` invoices |

The raw download is cached under `data/01_raw/` (gitignored) and is reproducible
via the ingestion pipeline.

### Why construct churn labels ourselves?

We deliberately do **not** use a static, pre-labelled churn dataset:

- a static label hides *when* the decision is made, which permits temporal
  leakage and makes the model undeployable;
- it fixes a single churn definition rather than a business-configurable one;
- it removes the ability to simulate "what if we had acted at time *T*".

Instead we build a **temporal prediction problem**:

```
                 observation window            performance window
        |---------------------------------|  |----------------------|
     cutoff T-90..T                    T    T+90
     features computed here           prediction point
                                              churn = no purchase in (T, T+90]
```

- **Observation window**: all features are computed strictly before the cutoff.
- **Calibration / churn label**: no purchase in the 90 days after the cutoff.
- **Leakage control**: features never see information from the performance window.

This mirrors how a real system operates: on 1 January you score customers using
only what happened before 1 January, then act.

---

## 3. Platform architecture

```
              UCI Online Retail II (zip / xlsx)
                          │
                 data ingestion (Kedro)
                          │
             validation + cleaning + medallion layers
       data/01_raw → 02_intermediate → 03_primary → 08_reporting
                          │
              customer feature engineering (RFM + behaviour)
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
        Churn models            CLV model
   (LogReg / RF / LightGBM)   (BTYD-style / gamma-gamma / RF)
              └───────────┬───────────┘
                          ▼
                 revenue-at-risk engine
                 Revenue at Risk = P(churn) × CLV
                          │
              segmentation (K-Means / GMM / hierarchical)
                          │
              retention prioritisation engine
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
   FastAPI inference API            Streamlit C-suite dashboard
```

Orchestration: **Kedro** · Packaging/deps: **uv** · Experiments: **MLflow** ·
Data versioning: **DVC** · Serving: **FastAPI** · Dashboard: **Streamlit** ·
Containers: **Docker** · CI: **GitHub Actions**.

---

## 4. Project structure

```
.
├── conf/
│   ├── base/               # catalog, parameters (versioned)
│   └── local/              # credentials + local overrides (gitignored)
├── data/                   # medallion layers (gitignored, DVC-tracked)
│   ├── 01_raw/             # immutable downloads
│   ├── 02_intermediate/    # parsed raw
│   ├── 03_primary/         # cleaned, analysis-ready
│   ├── 04_feature/         # customer feature store
│   ├── 05_model_input/     # train/val/test splits
│   ├── 06_models/          # trained model artefacts
│   ├── 07_model_output/    # predictions, segments, risk scores
│   └── 08_reporting/       # reports and summaries
├── src/customer_clv_churn/
│   ├── pipelines/
│   │   ├── data_ingestion/      # download → parse → clean → validate → summarise
│   │   ├── data_quality/        # feature-quality curation (flags, winsorise, grouping)
│   │   ├── feature_engineering/ # customer feature store + churn labels
│   │   └── baseline/            # naive 90-day recency rule + evaluation
│   ├── pipeline_registry.py
│   └── settings.py
├── tests/                  # mirror of src/ structure
├── pyproject.toml          # deps, tooling (uv)
├── uv.lock                 # reproducible environment
└── README.md
```

---

## 5. Getting started

Requires [uv](https://docs.astral.sh/uv/) (Python 3.12 is pinned via
`.python-version`).

```bash
# 1. Install the environment (creates .venv, resolves uv.lock)
uv sync

# 2. Run the ingestion pipeline (downloads ~44 MB, cleans, validates)
uv run kedro run --pipeline data_ingestion

# 3. Run the test suite
uv run pytest

# 4. Lint / format
uv run ruff check src tests
uv run ruff format src tests
```

Convenience wrappers are available in the `Makefile` (`make install`, `make
ingest`, `make test`, `make lint`).

### Ingestion outputs

| Dataset | Layer | Description |
| ------- | ----- | ----------- |
| `online_retail_II.xlsx` | 01_raw | Cached raw download |
| `transactions_raw` | 02_intermediate | Both sheets parsed to one frame |
| `transactions` | 03_primary | Cleaned, analysis-ready transactions |
| `data_quality_report` | 08_reporting | Row/customer/revenue counts + issues |
| `transactions_summary` | 08_reporting | Monthly business summary |

Reference run (full dataset):

```
raw rows            1,067,371
clean rows            808,723   (258,648 dropped: no customer, admin codes, dupes)
customers                5,897
invoices                43,959
net revenue         £16,663,260.75
date range       2009-12-01 → 2011-12-09
quality checks            passed
```

### Cleaning rules

- drop rows without a `Customer ID` (not attributable to a customer);
- drop administrative `StockCode`s (POST, D, M, BANK CHARGES, …);
- coerce dtypes; parse `InvoiceDate`; de-duplicate;
- derive `is_cancellation`, `is_return`, `line_revenue`, `invoice_month`;
- validate schema, nulls and expected date range, raising on hard failures.

### Feature-quality curation (`data_quality`)

An EDA-driven layer sits between ingestion and feature engineering. It does
**not** impute `customer_id` (its missingness is MNAR — anonymous vs identified
customers — so exclusion, not imputation, is correct). Instead it:

- removes the last residual administrative stock codes (`ADJUST`, `SP1002`,
  `TEST001/002`) that survived the ingestion filter;
- flags zero-price lines (`is_zero_price`) and defines `is_monetary_eligible`
  so monetary aggregates are not distorted by freebies/service lines;
- splits **net vs gross** revenue and quantifies returns
  (`line_revenue_gross`, `return_value`);
- winsorises the heavy tails (`quantity_winsorised`,
  `line_revenue_winsorised`, `is_outlier`) rather than deleting genuine
  high-value customers;
- collapses long-tail countries into `country_grouped` (41 → 32);
- validates `stock_code` format, non-negative prices and nulls via a gate.

Reference run:

```
rows 808,641  (82 residual admin rows removed)
zero-price lines        61
outliers capped      2,277
returns             17,878
countries            41 → 32 grouped
gross revenue   £17,376,870.09
net revenue     £16,660,273.33   (gross − returns reconciles exactly)
```

---

## 6. Methodology

### 6.1 Feature engineering

Customer-level features across four families:

- **Recency** — days since last/previous purchase.
- **Frequency** — orders total and in 30/90/180-day windows, purchase frequency,
  average inter-purchase interval.
- **Monetary** — total/recent revenue, average and median order value.
- **Behavioural** — revenue and frequency growth/decline, AOV change, purchase
  volatility, product/category diversity, return rate, tenure, spending trend,
  recent-vs-historical activity, country, interval changes, rolling stats.

Target: ~30–50 meaningful, business-interpretable features (no feature bloat).

### 6.1b Feature store & temporal split (implemented)

The `feature_engineering` pipeline turns transactions into a customer feature
store using a strict observation → performance split:

```
observation window (≤ T)            performance window (T, T+90]
features computed here              churn = no purchase here
        |----------------------------------|-----|------------------|
     Dec 2009                        T = 2011-06-09              2011-09-07
```

- `customer_features` — **50 features** per eligible customer, observation-only
  (RFM, rolling 30/90/180/365-day activity, intervals and their volatility,
  product diversity, return/cancellation rates, growth and momentum ratios,
  recency-to-interval ratio, trend slope, grouped country).
- `churn_labels` — `churn` (no purchase in the 90-day performance window) plus
  observed `future_revenue` / `future_orders`, used for evaluation.

Reference run: **4,941 customers, 67.7% 90-day churn**. A test explicitly
asserts that performance-window purchases never leak into the features.

### 6.1c Baseline (current company strategy) — implemented

The `baseline` pipeline encodes the rule *"inactive > 90 days ⇒ at risk"* and
evaluates it against the held-out labels. Value is proxied as 12 × observed
monthly net revenue (annualised run-rate) until the CLV model lands.

Reference result at cutoff `2011-06-09`:

| Metric | Baseline (recency > 90d) |
| ------ | ------------------------ |
| Customers targeted | 2,959 / 4,941 (59.9%) |
| Precision / recall / F1 | 0.833 / 0.738 / 0.783 |
| False positives / false negatives | 493 / 877 |
| Historical revenue targeted | £3,153,608 |
| Potential revenue protected (proxy) | £2,144,325 |

This is the bar the ML strategy must beat — not just on F1, but on the
**value** of the customers it prioritises.

### 6.2 Baseline (current company strategy)

A recency rule: *"no purchase for > 90 days ⇒ at risk ⇒ generic retention
offer."* We measure customers targeted, revenue/CLV targeted, true churners
captured, and potential revenue protected — then compare against the ML strategy.

### 6.3 Churn models

Logistic Regression, Random Forest and LightGBM (plus survival analysis:
Kaplan–Meier, Cox PH, survival forests). Time-based train/validation/test splits
(not random) because the system predicts the future from the past. Calibration
matters because probabilities are multiplied by CLV; we evaluate ROC-AUC,
PR-AUC, precision, recall, F1, calibration curves, and confusion matrices, and
apply isotonic/Platt calibration where needed.

### 6.4 CLV

We distinguish **historical value**, **predicted future value** and **customer
lifetime value**, and estimate expected future CLV per customer (probabilistic
BTYD-style and/or supervised approaches, validated against held-out revenue).
Modelled estimates are always labelled as such.

### 6.5 Revenue at risk & prioritisation

```
Revenue at Risk_i = P(churn_i) × ExpectedFutureCLV_i
Total Revenue at Risk = Σ_i Revenue at Risk_i
```

Customers are ranked by expected revenue at risk, so the platform can answer
*"if we can only target 100/250/500 customers, who?"* and quantify the uplift
over the recency rule.

### 6.6 Segmentation & explainability

Behavioural segments via K-Means / GMM / hierarchical clustering, interpreted in
business terms (Champions, High-value at risk, New, Occasional, Price-sensitive,
High-return, Lost…). Per segment: size, revenue, average CLV, churn rate,
average churn probability, revenue at risk, key risk factors, recommended action.
SHAP explains individual predictions (e.g. ↓ frequency, ↑ days since last
purchase, ↓ recent spend, ↑ interval, ↑ return rate) and aggregate importance is
reported per segment.

### 6.7 Retention recommendations

Segment × risk-factor → intervention mapping (loyalty outreach, second-purchase
incentives, targeted discounts, product-experience review). The dataset is
**observational**: these are hypotheses, and **A/B tests are required to
estimate real treatment effects**. We therefore report *potential* revenue at
risk and *estimated* CLV at risk, never "recovered" revenue.

---

## 7. Evaluation

**ML**: ROC-AUC, PR-AUC, F1, recall, calibration, SHAP/explainability.
**Business**: revenue at risk identified, CLV at risk identified, high-value
churners captured, customers targeted per budget, uplift vs. recency baseline,
value of prioritisation.

---

## 8. Roadmap / status

| Stage | Deliverable | Status |
| ----- | ----------- | ------ |
| 1 | Project scaffold (Kedro + uv + tooling) | ✅ done |
| 2 | Data ingestion, cleaning, validation | ✅ done |
| 2b | Feature-quality curation (flags, winsorise, grouping) | ✅ done |
| 3 | Feature engineering + customer feature store | ✅ done |
| 4 | Recency baseline strategy | ✅ done |
| 5 | Churn models (LogReg/RF/LightGBM) + calibration | ⏳ next |
| 6 | CLV model | ⏳ |
| 7 | Segmentation | ⏳ |
| 8 | Revenue-at-risk + prioritisation engine | ⏳ |
| 9 | SHAP explainability layer | ⏳ |
| 10 | FastAPI inference service | ⏳ |
| 11 | Streamlit C-suite dashboard (5 pages) | ⏳ |
| 12 | MLflow tracking, DVC datasets, Docker, CI | ⏳ |
| 13 | Extension: IBM Telco (subscription churn) | ⏳ |

---

## 9. Limitations

- Observational data: **causal** effects of interventions are unknown until A/B
  tested; all figures are *potential / estimated*.
- The UCI dataset ends Dec 2011 and covers a single UK-heavy retailer; behaviour
  may not transfer to other markets or subscription models.
- Features depend on complete transaction history being available at the cutoff.
- CLV is a modelled estimate, sensitive to its horizon and margin assumptions.

## 10. Future work

- Causal uplift modelling / heterogenous treatment effects for retention offers.
- Survival-based dynamic churn timing and "next purchase" forecasting.
- Automated retraining and drift monitoring in production.
- Generalise the framework to a subscription dataset (IBM Telco Customer Churn),
  carefully reconciling differing churn and value definitions.

---

## License

MIT.
