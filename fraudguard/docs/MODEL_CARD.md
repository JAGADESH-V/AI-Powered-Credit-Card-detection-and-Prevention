# Model card: FraudGuard transaction scorer

## What it does
Assigns a fraud probability to a card transaction and returns one of three
decisions: approve, review, or decline. Intended to sit inline in an
authorisation flow, with declined and reviewed transactions going to a human
analyst queue.

## What it is not
Not an automatic account-closure system, not a chargeback adjudicator, and not
a substitute for an analyst. Every non-approve decision is designed to be
reviewable by a person, with reason codes attached.

## Inputs
Twenty features across two groups:

- **Stateless** — amount, hour of day, channel, cross-border flag, merchant
  category risk, 3-D Secure status.
- **Stateful** — transaction counts over 1h and 24h, amount relative to the
  card's own baseline, distance and implied travel speed from the last
  known-good location, distinct merchants in 24h, device familiarity.

No PAN, name, address, or demographic attribute is used as a feature.

## Training
- Gradient-boosted trees, calibrated with isotonic regression (or Platt scaling
  when the calibration slice holds fewer than 100 fraud cases).
- Strictly chronological three-way split: train → calibrate → test. The
  calibration layer never refits the base model.
- Decision threshold selected by minimising expected cost, not by maximising
  F1 or defaulting to 0.5.

## Metrics that matter
PR-AUC, recall at the operating threshold, false-positive rate, and expected
calibration error. Accuracy is deliberately not reported: at a 1% base rate it
is uninformative.

## Known limitations

**Label latency.** Chargebacks confirm fraud 30–90 days after the fact. Recent
performance figures are always provisional. `/v1/monitoring/performance`
reports only on analyst-labelled transactions and excludes unlabelled traffic
rather than assuming it was legitimate.

**Adversarial drift.** Attackers adapt to the boundary. A sudden fall in the
alert rate is a possible evasion, not a success — treat it as an alarm.

**Cold start.** A card with no history gets neutral values for every stateful
feature, so the first transactions on a new card are scored on stateless
signal alone and are weaker.

**Geographic features need coordinates.** If the gateway does not supply
latitude and longitude, implied speed and distance are zero and the strongest
feature in the model is unavailable.

**Synthetic benchmark numbers are inflated.** The bundled generator produces
cleanly separable fraud patterns. PR-AUC near 0.97 reflects that, not
production performance. On real portfolios, expect 0.4–0.8.

## Fairness
The model uses no protected attribute as a feature. Two features nonetheless
warrant monitoring for disparate impact:

- `cross_border` will flag legitimate travellers and diaspora remittance
  corridors more often than domestic-only customers.
- `high_risk_mcc` encodes merchant categories, which correlate with spending
  patterns that vary by demographic.

Before deployment, measure decline rates by cardholder country and by issuer
segment, and review any segment with a materially higher false-positive rate.
This is an organisational commitment, not something the code enforces.

## Retraining
Retrain when drift is flagged as major (PSI ≥ 0.25) on a top-five feature,
when measured precision on labelled cases falls materially, or quarterly —
whichever comes first. Always compare against the incumbent on an identical
split before promotion.

## Ownership
Record the accountable team, review cadence, and escalation path here before
this model goes anywhere near production traffic.
