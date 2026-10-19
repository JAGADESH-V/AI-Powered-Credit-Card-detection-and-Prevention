"""Analyst case-review dashboard.

    streamlit run dashboard/app.py

Talks to the API over HTTP with an analyst credential. It deliberately holds
no database connection of its own: the dashboard is the most exposed surface
in the system, and giving it direct database access would make a session
hijack far more damaging than it needs to be. Everything it can do is bounded
by the scopes on the credential it presents.
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("FG_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 10

st.set_page_config(page_title="FraudGuard cases", layout="wide")


def api_get(path: str, key: str, **params):
    response = requests.get(
        f"{API_URL}{path}", headers={"X-API-Key": key},
        params=params, timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def api_post(path: str, key: str, payload: dict):
    response = requests.post(
        f"{API_URL}{path}", headers={"X-API-Key": key},
        json=payload, timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# credential
# ---------------------------------------------------------------------------
st.sidebar.title("FraudGuard")
api_key = st.sidebar.text_input("Analyst API key", type="password")

if not api_key:
    st.info("Enter an analyst API key in the sidebar to load open cases.")
    st.stop()

try:
    status = api_get("/ready", api_key)
except requests.HTTPError as exc:
    st.error(f"Could not reach the API: {exc.response.status_code}")
    st.stop()
except requests.RequestException:
    st.error(f"No response from {API_URL}. Is the service running?")
    st.stop()

st.sidebar.caption(f"Model {status['model_version']} · {status['algorithm']}")
if not status.get("distributed_stores"):
    st.sidebar.warning("Running on in-process stores — single instance only.")

# ---------------------------------------------------------------------------
# queue
# ---------------------------------------------------------------------------
st.title("Open cases")

try:
    payload = api_get("/v1/cases", api_key, limit=100)
except requests.HTTPError as exc:
    if exc.response.status_code == 403:
        st.error("This key does not carry the case:read scope.")
    else:
        st.error(f"Could not load cases: {exc.response.status_code}")
    st.stop()

cases = payload["open_cases"]

if not cases:
    st.success("No open cases. Nothing is waiting for review.")
    st.stop()

flagged_value = sum(c["amount"] for c in cases)
left, middle, right = st.columns(3)
left.metric("Awaiting review", len(cases))
middle.metric("Value at risk", f"{flagged_value:,.2f}")
right.metric("Highest risk", f"{cases[0]['fraud_probability']:.1%}")

st.dataframe(
    pd.DataFrame(
        [
            {
                "Transaction": c["transaction_id"],
                "Card": f"****{c['card_last_four']}",
                "Amount": c["amount"],
                "Merchant": c["merchant_id"],
                "Country": c["merchant_country"],
                "Risk": c["fraud_probability"],
                "Decision": c["decision"],
            }
            for c in cases
        ]
    ),
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------------------------
# review one case
# ---------------------------------------------------------------------------
st.header("Review")

selected_id = st.selectbox(
    "Transaction", [c["transaction_id"] for c in cases],
    format_func=lambda tid: f"{tid} — {next(c['fraud_probability'] for c in cases if c['transaction_id'] == tid):.1%}",
)
case = next(c for c in cases if c["transaction_id"] == selected_id)

details, reasons = st.columns([1, 1])

with details:
    st.subheader("Transaction")
    st.write(
        {
            "Card": f"****{case['card_last_four']}",
            "Amount": f"{case['amount']:,.2f} {case['currency']}",
            "Merchant": case["merchant_id"],
            "Merchant country": case["merchant_country"],
            "Occurred": case["occurred_at"],
            "Model decision": case["decision"],
        }
    )

with reasons:
    st.subheader("Why it was flagged")
    if case["reason_codes"]:
        for reason in case["reason_codes"]:
            st.write(f"**{reason['contribution']:+.3f}** — {reason['description']}")
    else:
        st.caption("No reason codes recorded.")

st.progress(min(case["fraud_probability"], 1.0), text=f"Risk {case['fraud_probability']:.1%}")

note = st.text_area("Note", placeholder="What did you check, and what did you find?")

confirm, dismiss = st.columns(2)

with confirm:
    if st.button("Confirm fraud", type="primary", use_container_width=True):
        try:
            api_post("/v1/cases/decision", api_key, {
                "transaction_id": selected_id, "confirmed_fraud": True, "note": note,
            })
            st.success("Recorded as fraud. This becomes a training label.")
            st.rerun()
        except requests.HTTPError as exc:
            st.error(f"Could not record the decision: {exc.response.status_code}")

with dismiss:
    if st.button("Not fraud", use_container_width=True):
        try:
            api_post("/v1/cases/decision", api_key, {
                "transaction_id": selected_id, "confirmed_fraud": False, "note": note,
            })
            st.success("Recorded as legitimate.")
            st.rerun()
        except requests.HTTPError as exc:
            st.error(f"Could not record the decision: {exc.response.status_code}")

# ---------------------------------------------------------------------------
# monitoring
# ---------------------------------------------------------------------------
st.header("Model health")

try:
    drift = api_get("/v1/monitoring/drift", api_key)
    performance = api_get("/v1/monitoring/performance", api_key, days=30)
except requests.HTTPError:
    st.caption("Monitoring needs the case:read scope.")
else:
    scores = drift.get("scores", {})
    a, b, c = st.columns(3)
    a.metric("Alert rate", f"{scores.get('alert_rate', 0):.2%}")
    b.metric("Mean score", f"{scores.get('mean_score', 0):.4f}")
    c.metric("Worst feature PSI", f"{drift.get('worst_psi', 0):.3f}")

    if not drift.get("sufficient_data"):
        st.caption(
            f"Drift needs more traffic: {drift.get('samples_observed', 0)} scored so far."
        )
    elif drift.get("features_drifting"):
        st.warning("Features have moved away from the training distribution.")
        st.dataframe(pd.DataFrame(drift["features_drifting"]), hide_index=True)
    else:
        st.success("No feature drift detected against the training distribution.")

    if performance.get("labelled"):
        st.caption(
            f"On {performance['labelled']} analyst-labelled transactions in the last "
            f"30 days: precision {performance.get('precision')}, "
            f"recall {performance.get('recall')}. Unlabelled traffic is excluded."
        )
    else:
        st.caption(
            "No analyst-labelled transactions yet. Measured performance needs "
            "verdicts — chargeback labels take 30-90 days to arrive."
        )
