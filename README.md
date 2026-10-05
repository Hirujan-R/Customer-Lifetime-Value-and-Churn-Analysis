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
│   │   ├── baseline/            # naive 90-day recency rule + evaluation
│   │   ├── segmentation/        # clustering + rule-based segment comparison
│   │   ├── churn_modeling/      # multi-cohort models, calibration, survival
│   │   ├── clv/                 # CLV regression + revenue-at-risk
│   │   ├── prioritisation/      # value-weighted retention ranking + strategy comparison
│   │   ├── explainability/      # SHAP global / customer / segment risk factors
│   │   └── recommendations/     # segment + risk-factor -> retention actions
│   ├── pipeline_registry.py
│   ├── settings.py
│   └── dashboard/               # C-suite Streamlit app (5 pages)
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

### 6.6b Segmentation — implemented

The `segmentation` pipeline clusters customers on **observation features only**
(the churn label is never used to form segments), selecting the best model by
silhouette across K-Means, Gaussian Mixture and Agglomerative models, then
interpreting and profiling the clusters and comparing them to an interpretable
RFM rule-based segmentation.

Reference run: best model **K-Means, k=3 (silhouette 0.30)**.

| Segment | Customers | Net revenue | Avg value proxy | Observed churn | Revenue at risk (proxy) |
| ------- | --------- | ----------- | --------------- | -------------- | ----------------------- |
| Champions | 1,870 (37.9%) | £10,068,594 | £4,303 | 43.2% | £1,985,223 |
| Occasional buyers | 2,628 (53.2%) | £1,344,950 | £588 | 86.8% | £1,287,110 |
| New customers | 443 (9.0%) | £240,590 | £3,538 | 57.6% | £780,834 |

The rule-based comparison surfaces the small high-value cohorts that pure
clustering merges away — e.g. **High-value at risk** (117 customers, £221k at
risk) and **High-return customers** (34), which get their own strategies.
Outputs: `customer_segments`, `segment_profiles.csv`, `rule_segment_profiles.csv`,
`segmentation_report.json`. As models land, `observed_churn_rate` is replaced by
the calibrated churn probability.

### 6.3b Churn models — implemented

The `churn_modeling` pipeline builds **multiple temporal cohorts** (monthly
cutoffs from 2011-03 to 2011-09), each with its own observation features and
90-day label, and splits them by **time** — never randomly:

```
train (5 earlier cohorts)  ->  validation (2011-08)  ->  test (2011-09)
   23,924 rows                    5,117 rows              5,224 rows
```

Models: **logistic regression, random forest and LightGBM**. The winner is
chosen on the **validation cohort only** (never test) by PR-AUC, then calibrated
on the validation cohort. Everything is tracked in **MLflow** (`mlruns/`).

**Model selection — validation PR-AUC decides the winner, test is reported only:**

| Model | Validation PR-AUC | Test PR-AUC |
| ----- | ----------------- | ----------- |
| Logistic Regression | 0.8624 | 0.8170 |
| **Random Forest (selected)** | **0.9079** | 0.8243 |
| LightGBM | 0.9030 | 0.8248 |

Random Forest wins on validation and is therefore selected. LightGBM's **test**
PR-AUC is marginally higher (0.8248 vs 0.8243 — a 0.0005 difference, i.e. noise),
but **test is never used for selection** (that would be leakage / overfitting to
the holdout). The model is picked on validation, then reported on test.

**Selected + calibrated model — test-cohort metrics:**

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Brier |
| ----- | ------- | ------ | --------- | ------ | -- | ----- |
| Selected + calibrated (Random Forest) | 0.809 | 0.816 | 0.730 | 0.871 | 0.794 | 0.177 |

Note the calibrated PR-AUC (0.816) is slightly **below** the raw RF PR-AUC
(0.824). Isotonic calibration is monotonic but maps probabilities into flat
segments, creating ties that slightly reduce rank-based metrics (ROC-AUC /
PR-AUC). Calibration optimises *probability accuracy*, not *ranking* — it is
applied here because the probabilities are multiplied by CLV downstream. It
improved validation Brier (0.1383 → 0.1341), the criterion used to apply it;
on the later test cohort it was neutral/slightly worse (0.1762 → 0.1773) due to
cohort shift, which the pipeline reports honestly.

Business: targeting the top 10% by probability captures **15.7% of all
churners at 90.0% precision (1.57× lift)**. The pipeline emits
`calibration_applied`, Brier and expected calibration error alongside the ranking
metrics.

Survival analysis (exploratory): Kaplan–Meier curves by segment plus a Cox
proportional-hazards model — median time-to-churn 239 days, concordance 0.85,
with recency increasing and order frequency decreasing churn hazard.

Outputs: `churn_test_predictions`, `churn_feature_importance.csv`,
`churn_model_metrics.json`, `survival_report.json`, `survival_curves.csv`, and an
MLflow run.

### 6.4b CLV and revenue-at-risk — implemented

The `clv` pipeline trains a supervised regression on the **same temporal panel**
to predict net revenue in the next 90 days, then combines it with the calibrated
churn probability to estimate CLV and revenue at risk.

Models: Ridge, Random Forest and LightGBM (Tweedie objective for zero-inflated
revenue). Selected by validation MAE. Reference result (test cohort):

| Model | MAE | RMSE | R² | Spearman | Top-decile value capture |
| ----- | --- | ---- | -- | -------- | ------------------------ |
| Ridge | 394.84 | 1,767.72 | 0.674 | 0.438 | 0.58 |
| Random Forest | 351.29 | 1,671.38 | 0.709 | 0.570 | 0.60 |
| **LightGBM (selected)** | **340.94** | 1,693.06 | 0.701 | 0.597 | 0.60 |

It clearly separates **observed** from **modelled** value:

| Quantity | Provenance |
| -------- | ---------- |
| `historical_value` | observed (observation window) |
| `predicted_90d_value` | modelled |
| `expected_future_clv` | modelled (horizon formula) |
| `revenue_at_risk` | modelled = `P(churn) × expected_future_clv` |
| `actual_future_revenue` | observed (held-out; not available at scoring time) |

**CLV assumptions** (explicit): 90-day-ahead revenue is modelled from
observation features only; value per active 90-day period is stationary
(`v = predicted_90d / retention_probability`); per-period retention equals
`1 − calibrated churn probability`; horizon = 365 days; no margin/inflation.
Revenue at risk uses the spec's definition `P(churn) × expected_future_clv`.

Reference portfolio (test cohort, 5,224 customers, 365-day horizon):

```
mean churn probability       0.624
total historical value       £13,327,588  (observed)
total predicted 90-day value  £1,509,250  (modelled)
total expected future CLV     £6,100,763  (modelled)
total revenue at risk           £960,359  (modelled)
top 10% of customers hold 36.0% of all revenue at risk
```

Caveat: like most revenue models, the extreme tail is under-predicted (mean
predicted / mean actual ≈ 0.56), so totals are conservative lower bounds.
Outputs: `customer_value_risk`, `clv_model_metrics.json`,
`clv_feature_importance.csv`, `revenue_at_risk_summary.json`, and an MLflow run.

### 6.5b Prioritisation engine — implemented

The `prioritisation` pipeline ranks customers by **expected revenue at risk**
and compares that ranking against naive alternatives at realistic budgets. This
operationalises the core insight: the highest-probability churner is not the best
retention investment.

Reference result (test cohort, revenue at risk captured at each budget):

| Strategy | Top 100 | Top 250 | Top 500 | Top 1,000 |
| -------- | ------- | ------- | ------- | --------- |
| Recency rule (naive) | £5,341 | £12,034 | £21,768 | £55,004 |
| Churn probability only | £1,233 | £3,206 | £14,983 | £54,295 |
| Expected CLV only | £73,331 | £123,511 | £215,822 | £382,750 |
| **Revenue at risk (recommended)** | **£138,473** | **£228,431** | **£336,884** | **£498,528** |

At a budget of 100 customers, value-weighted prioritisation captures **24.9×
more** revenue at risk than the recency rule and **112× more** than ranking by
churn probability alone. The full recency rule targets 63% of customers to
capture £468k, whereas the ML ranking captures more with 500. The report also
exposes `target_top_100/250/500/1000` flags so the engine answers *"if we can
only target N customers, who?"*.

### 6.6b Explainability and risk factors — implemented

The `explainability` pipeline explains the churn model with **SHAP** for all
5,224 scored customers (81 features):

- **Global** top drivers: `n_orders_365`, `revenue_365`, `active_month_ratio`,
  `purchase_frequency`, `recency_days`.
- **Per customer**: top-5 risk factors with direction (↑ increases risk / ↓
  decreases risk) and SHAP value.
- **Per segment** aggregated risk factors (e.g. Champions and New customers are
  both driven by `n_orders_365` and `active_months`).

SHAP attributions are **predictive/correlational**, not causal.

### 6.7b Retention recommendations — implemented

The `recommendations` pipeline maps each customer's segment and dominant SHAP
risk factor to a recommended action, then aggregates an executive action list
ranked by potential CLV at risk:

| Action (top entries) | Customers | Potential CLV at risk |
| -------------------- | --------- | --------------------- |
| Re-engagement campaign to rebuild purchase habit | 2,313 | £479,273 |
| Targeted offer to reactivate declining spend | 1,192 | £165,024 |
| Value-oriented offers and bundles | 684 | £106,923 |
| Reward loyalty; protect experience | 346 | £77,280 |
| Win-back outreach | 225 | £41,061 |

**Caveat:** the data is observational — these are prioritised hypotheses. A/B
testing or causal experimentation is required to estimate the actual treatment
effect of each intervention; no recovered revenue is claimed.

### 6.8 C-suite dashboard — implemented

A five-page Streamlit app designed for senior management:

1. **Executive Overview** — KPIs (total/active customers, customers at risk,
   revenue at risk, future CLV at risk, high-value customers at risk), risk
   distribution, revenue at risk by segment, top customers, and management focus.
2. **Customer Segmentation** — segment size/revenue/CLV/churn/revenue at risk,
   with per-segment characteristics, SHAP risk factors and intervention.
3. **Churn Risk** — searchable/filterable customer table; select a customer for
   profile, churn probability, CLV, revenue at risk, SHAP explanation and action.
4. **Revenue Impact** — baseline vs ML at a selectable retention budget, with the
   prioritised customer list and segment mix.
5. **Recommended Actions** — the executive action list with counts and potential
   CLV at risk.

Run it with:

```bash
uv run streamlit run src/customer_clv_churn/dashboard/app.py
# or: make dashboard
```

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
| 5 | Churn models (LogReg/RF/LightGBM) + calibration | ✅ done |
| 6 | CLV model | ✅ done |
| 7 | Segmentation | ✅ done |
| 8 | Revenue-at-risk + prioritisation engine | ✅ done |
| 9 | SHAP explainability layer | ✅ done |
| 9b | Retention recommendations | ✅ done |
| 10 | FastAPI inference service | ⏳ next |
| 11 | Streamlit C-suite dashboard (5 pages) | ✅ done |
| 12 | MLflow tracking, DVC datasets, Docker, CI | ⏳ (MLflow done) |
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
