# FraudGuard

Real-time credit card fraud detection: calibrated scoring, explainable
decisions, an analyst case queue, and drift monitoring — with the security
controls built into the design rather than bolted on.

Everything here runs. `make all` trains, tests, and demonstrates the whole
system in about a minute.

---

## Quick start

```bash
pip install -r requirements-dev.txt
python scripts/generate_secrets.py >> .env

make train      # train and calibrate; prints the artifact SHA-256
make test       # 78 tests
make demo       # end-to-end: attack, replay, audit, cases, monitoring
```

Then, optionally:

```bash
make serve                              # API on 127.0.0.1:8000
make dashboard                          # analyst UI
make compare                            # benchmark logreg / rf / gbdt
python scripts/train_model.py --source csv --path your_transactions.csv
```

Training prints a SHA-256. Put it in `.env` as `FG_MODEL_SHA256` — the service
refuses to load a model whose hash does not match.

---

## What it does

```
transaction ──► validate ──► tokenize PAN ──► features ──► calibrated model
                                                  │              │
                             card history ◄───────┤              ▼
                          (velocity + baseline)   │   approve / review / decline
                                                  │         + reason codes
                                                  ▼              │
                                          drift monitor          ▼
                                                            audit log
                                                                 │
                                                                 ▼
                                                    analyst queue ──► labels
                                                                       │
                                                                       ▼
                                                              next model version
```

The loop closes: analyst verdicts become the training labels for the next
model. That is the part most tutorials leave out, and it is the part that
determines whether the system still works in six months.

---

## The model

**Algorithm.** Gradient-boosted trees, then isotonic calibration. Tabular fraud
data is where boosted trees still beat deep learning, and they train in
seconds without a GPU.

**Four decisions that matter more than the algorithm:**

1. **Chronological split, not random.** Fraud drifts. A random split lets the
   model learn from the future and produces a number you cannot ship on.

2. **Calibration on a held-out slice.** A raw boosted-tree score is a ranking,
   not a probability. The cost threshold below is only meaningful if `p=0.6`
   really means a 60% chance of fraud. The base model is fitted on one slice
   and the calibration curve on a later one it never saw — and if that slice
   holds too few fraud cases, calibration is skipped and the report says so,
   because a curve fitted on a handful of points is worse than none.

3. **PR-AUC and expected calibration error, not accuracy.** At a 1% base rate,
   approving everything scores 99% accurate and catches nothing.

4. **Threshold tuned on cost.** A missed fraud costs ~180 units; a wrongly
   blocked customer ~12. Those two numbers in `models/train.py` are the most
   consequential setting in the system — tune them with your risk team.

### Features

Twenty features, built by the **same code path** at training and serving time,
which is how train/serve skew is avoided.

| Group | Examples | Why |
|---|---|---|
| Stateless | amount, hour (sin/cos), channel, cross-border, high-risk MCC, 3-D Secure | Cheap, no state, identical everywhere |
| Stateful | count in 1h/24h, amount vs card's baseline, distance and implied speed from last location, new device | Highest signal, and the easiest to get wrong |

Implied travel speed is usually the strongest single feature: a card used in
Bangalore and then New York twenty minutes later implies 40,000 km/h.

### An honest benchmark result

```
algorithm    PR-AUC  ROC-AUC     ECE   recall   saving/1k
logreg       0.9868   0.9992  0.0118   0.9861     1552.44
rf           0.9859   0.9994  0.0030   0.9722     1534.90
gbdt         0.9866   0.9983  0.0095   0.9861     1555.37
```

Logistic regression is within 0.0002 PR-AUC of the boosted trees here. On this
data the complex model is not buying much, and a linear model is far easier to
defend to a regulator. Run `make compare` on your own data before assuming
otherwise — that is the point of keeping the baseline in the codebase.

---

## Security

Threat model: the attacker has the card number, can reach the API, and can
observe traffic.

### Cardholder data
A PAN enters at one function and does not survive it. What persists is a keyed
HMAC token plus the last four digits. The token is *deterministic*, so velocity
features still link transactions on the same card, and *keyed*, so a stolen
token database cannot be brute-forced back to card numbers. PANs are
Luhn-checked on entry, masked by a log-handler filter, and kept out of
`repr()`. Redaction sits at the handler, not the call site, because the call
site you forget is the one that leaks.

### Request integrity
Scoring requests are HMAC-signed over
`timestamp.nonce.method.path.sha256(body)`. That gives authenticity, integrity,
and replay protection at once. The nonce is consumed **only after** the
signature verifies, so an attacker cannot burn a legitimate nonce by replaying
it with a bad signature. In Redis the check-and-store is a single atomic
`SET NX`, so two concurrent replays cannot both win.

### Authentication
API keys are issued as `fgk_<key_id>.<secret>`. Only the secret half is hashed
(Argon2id); the id is indexed. Without that split, every request would have to
Argon2-verify against every stored key — deliberately expensive hashing turned
into O(n) work per call, which is a denial-of-service handed to the attacker.
Unknown key ids are still verified against a dummy hash so failed lookups cost
the same time as real ones. Keys expire, can be revoked instantly by status,
and are never deleted, so the audit trail keeps referring to something real.

Roles are least-privilege: a gateway can score but cannot overturn a decision;
an auditor can read the trail but cannot write to it.

### Tamper-evident audit trail
Every consequential action is appended as a record whose hash covers the
previous record's hash. Editing or deleting any past entry breaks the chain and
`verify_chain()` names the record where it broke. The demo proves this by
editing a record and watching detection fire. This is what makes a decision
defensible in a chargeback dispute.

### Model artifacts are code
A pickled model is executable code; loading one from an untrusted path is
remote code execution. The loader verifies the artifact against a pinned hash
and refuses to start on mismatch.

### Failing closed, deliberately
Two backends make opposite choices on purpose. If **rate limiting** cannot
reach Redis it fails *open* — taking payments offline because Redis blipped is
worse than briefly unmetered traffic. If **replay protection** cannot reach
Redis it fails *closed* — an unverifiable request is not one to trust with a
payment. In production the app refuses to boot without Redis at all, because
in-process fallbacks are per-instance and would silently stop working behind a
load balancer.

The config layer also refuses to boot in production if secrets are missing,
TLS is disabled, CORS is wildcarded, the model hash is unpinned, or the
database is SQLite.

### Verified, not asserted
`bandit -r src/ -ll` reports no issues across 2,490 lines. `ruff` passes clean
with security lints (`S`) enabled. CI additionally runs `pip-audit` for
dependency CVEs, gitleaks for committed secrets, and Trivy against the built
image — on every push and weekly, since new CVEs appear without any code
changing.

---

## Two bugs worth describing

**Fraud concentrated in the holdout.** The first training run scored ROC-AUC
0.5000 — constant predictions. The generator was placing every fraud episode at
the end of each card's timeline, so the chronological split put *all* fraud in
the test set and none in training. Fixed by distributing onset across each
card's active window. `test_fraud_present_in_every_slice` now asserts it.

**History poisoning.** The first working version declined the *first*
transaction of a card-testing burst and approved the next five. The attacker's
own declined transaction was being written into the card's history, resetting
"last known location" to the attacker's city and marking their device as
familiar — so by attempt two the strongest features had collapsed to zero. The
attack was establishing its own baseline.

The fix splits history in two. Velocity counts include every attempt, because a
burst of declines is itself signal. But the baseline — last known-good
location, familiar devices, normal spend — only updates on approved
transactions. An attacker now has to beat the real cardholder's history on
every attempt. `TestHistoryPoisoning` locks it in.

---

## Tools, and why each one

| Layer | Choice | Reason |
|---|---|---|
| Model | scikit-learn `HistGradientBoosting` | Best-in-class on tabular fraud; seconds to train; no GPU |
| Calibration | Isotonic / Platt via `FrozenEstimator` | Turns a ranking into a probability the cost threshold can use |
| Baseline | Logistic regression, Random Forest | Know what simple scores before shipping complex |
| Features | pandas, NumPy | Vectorised chronological replay |
| API | FastAPI + Pydantic v2 | Validation *is* a security control here |
| Hashing | Argon2id | Memory-hard; current standard for credentials |
| Tokens | PyJWT | Explicit algorithm allow-list blocks `alg=none` |
| Encryption | `cryptography` (Fernet) | Authenticated — tampering is detectable, not silent |
| Hot state | Redis | Velocity windows, rate limits, nonces; atomic, TTL'd, shared |
| Durable state | SQLAlchemy → PostgreSQL | Keys, scored transactions, case decisions |
| Explainability | Leave-one-out inline, SHAP offline | Fast enough for the request path; exact values for case files |
| Monitoring | PSI drift + Prometheus `/metrics` | Early warning while labels are still 30–90 days out |
| Dashboard | Streamlit | Analyst queue over the API, with no database access of its own |
| Scanning | bandit, pip-audit, gitleaks, Trivy | Our code, our dependencies, our secrets, our image |

---

## What is genuinely not solved

These are not gaps in the implementation. They are properties of the problem.

**Label latency.** Chargebacks confirm fraud 30–90 days later. Recent
performance is always provisional. The system handles this as well as it can
be handled — `/v1/monitoring/performance` reports only on analyst-labelled
transactions and excludes unlabelled traffic rather than assuming it was clean,
and drift monitoring gives an early signal in the meantime. But no amount of
engineering makes the labels arrive sooner.

**Adversarial adaptation.** Fraud is one of the few ML problems where the data
distribution actively fights back. Drift monitoring tells you *that* something
moved, not what the attacker will try next.

**Cold start.** A new card has no history, so every stateful feature is neutral
and the first transactions are scored on weaker signal.

**PCI-DSS compliance.** The cardholder-data handling here aligns with PCI
principles, but certification requires network segmentation, key management in
an HSM or KMS, scope reduction, and a QSA assessment. That is organisational
work no codebase can do for you.

**Benchmark numbers are inflated.** PR-AUC near 0.97 reflects synthetic data
with cleanly separable patterns. Real portfolios land in 0.4–0.8. Quote these
figures as evidence the pipeline works, not that the problem is solved. See
`docs/MODEL_CARD.md` for the full limitations and the fairness monitoring that
should happen before deployment.

---

## Layout

```
src/fraudguard/
  config.py              settings; fails closed in prod
  schemas.py             request/response contracts
  observability.py       JSON logging with redaction
  security/
    crypto.py            PAN tokenization, field encryption, masking
    auth.py              API keys, JWT, scopes, RBAC
    middleware.py        signing, replay, rate limit, headers
    audit.py             hash-chained audit log
  features/engineering.py   feature construction, trusted-vs-attempt history
  models/
    train.py             calibration, threshold tuning, benchmarking
    scorer.py            integrity-checked loading, reason codes
    drift.py             PSI against the training distribution
  data/
    generate.py          synthetic transactions with realistic fraud patterns
    loaders.py           Kaggle and generic CSV ingestion
  db/
    redis_stores.py      velocity, rate limits, nonces (+ fallbacks)
    sql_stores.py        keys, scored transactions, case decisions
  api/app.py             FastAPI wiring
dashboard/app.py         analyst case review
scripts/                 train_model, demo, explain, generate_secrets
tests/                   78 tests
deploy/                  hardened Dockerfile, compose with dropped capabilities
docs/MODEL_CARD.md       intended use, limitations, fairness, retraining
.github/workflows/ci.yml tests, lint, security scans, image scan
```

---

## Next steps that would genuinely add value

Champion/challenger shadow scoring so a new model sees production traffic
before promotion. Kafka ingestion when you outgrow request/response. A feature
store if other teams need the same velocity features. Per-segment fairness
dashboards as described in the model card.
