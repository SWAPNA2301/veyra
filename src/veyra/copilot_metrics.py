"""Small, explicit metric router over checksum-verified aggregate reports only."""

import json
import math
import re

from .copilot_corpus import CorpusUnavailable, corpus, fingerprint


def source(document, number):
    metadata = {key: value for key, value in document.items() if key != "text"}
    return {"id": number, "passage": document["text"], "similarity": None,
            "citation": {**metadata, "chunk_id": fingerprint(document)}}


def metric_answer(root, question):
    """Return None for ordinary RAG questions; fail closed for known metric intents.

    TransactionID counts transactions, never people. These are historical dataset
    aggregates, not live business/customer totals. No raw data or embedding loads.
    """
    text = question.lower()
    count = bool(re.search(r"\b(how many|number|count|total|unique|distinct|size)\b", text))
    customer = bool(re.search(r"\b(customers?|clients?|users?|people)\b", text))
    fraud_rate = bool(re.search(r"\bfraud\b", text) and re.search(r"\b(rate|prevalence|percent(?:age)?|proportion)\b", text))
    transactions = count and bool(re.search(r"\b(transactions?|rows?|dataset)\b", text))
    fraud_count = count and bool(re.search(r"\b(fraud|fraudulent)\b", text))
    threshold = bool(re.search(r"\bthreshold\b", text) and re.search(r"\b(what|value|exact|show)\b", text)
                     and not re.search(r"\b(why|how|selected|selection|choose|mean|means)\b", text))
    metrics = [key for key, pattern in {"precision": r"\bprecision\b", "recall": r"\brecall\b", "f1": r"\bf1\b",
               "pr_auc": r"\bpr[- _]?auc\b|average precision", "roc_auc": r"\broc[- _]?auc\b"}.items() if re.search(pattern, text)]
    if "average precision" in text:
        metrics = [key for key in metrics if key != "precision"]
    if metrics and re.search(r"\b(why|how|mean|means|definition|explain|improve|floor|minimum|target|constraint|logistic|baseline)\b|\b0\.\d+", text):
        return None
    if not (customer and (count or fraud_rate) or transactions or fraud_count or fraud_rate or threshold or metrics):
        return None
    reply = {"status": "insufficient_evidence", "answer": "The approved reports do not provide a verified value for that metric and scope. I cannot infer it from other totals.",
             "sources": [], "model": None}
    try:
        documents, _ = corpus(root)  # Existing allowlist, checksums and privacy policy.

        def document(filename, section):
            return next(item for item in documents if item["source"] == filename and item["section"] == section)

        if customer:
            rows = document("reports/data_quality.md", "Rows, schema and duplicates")
            reply.update(answer="A verified unique-customer count is not available in the approved reports. They count transaction rows and transaction IDs, not distinct people; identity-table rows are not a customer count either. I cannot infer how many customers we have from those totals. [1]",
                         sources=[source(rows, 1)])
            return reply
        # Do not answer a subgroup, recent/live period or other metric with a
        # whole-dataset total. Unknown scopes require explicit evidence.
        if re.search(r"\b(today|yesterday|current|currently|live|now|latest|week|month|year|batch|upload|product|device|country|merchant|identity|amount|loss|per|by|since|before|after|between|above|below|with|without|where|card4|card6)\b", text):
            return reply
        if metrics:
            if fraud_rate or fraud_count or transactions:
                reply["answer"] = "Please ask about dataset counts/rates or model performance separately so the population and metric stay unambiguous."
                return reply
            if re.search(r"\b(train|training)\b", text):
                return reply
            scopes = re.findall(r"\b(validation|test|holdout)\b", text)
            scopes = list(dict.fromkeys("test" if scope == "holdout" else scope for scope in scopes)) or ["validation", "test"]
            answers, evidence = [], []
            for index, split in enumerate(scopes, 1):
                doc = document("reports/holdout_evaluation.json", split)
                values = json.loads(doc["text"].split(": ", 1)[1])
                parts = []
                for key in metrics:
                    value = values[key]
                    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                        return reply
                    label = {"pr_auc": "PR-AUC (average precision)", "roc_auc": "ROC-AUC", "f1": "F1"}.get(key, key)
                    display = f"{value * 100:.2f}%" if key in {"precision", "recall"} else f"{value:.6f}"
                    parts.append(f"{label} {display}")
                answers.append(f"{split.capitalize()}: {', '.join(parts)}. [{index}]")
                evidence.append(source(doc, index))
            reply.update(status="answered", answer="Saved LightGBM results at the frozen operating threshold: " + " ".join(answers), sources=evidence)
            return reply
        if threshold and not (fraud_rate or fraud_count or transactions):
            doc = document("reports/holdout_evaluation.json", "threshold")
            value = json.loads(doc["text"].split(": ", 1)[1])
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                return reply
            reply.update(status="answered", answer=f"The saved LightGBM frozen threshold is {value}. Scores at or above it are flagged predictions, not confirmed fraud. [1]", sources=[source(doc, 1)])
            return reply
        scopes = set(re.findall(r"\b(train|training|validation|test|holdout)\b", text))
        scopes = {"train" if scope == "training" else "test" if scope == "holdout" else scope for scope in scopes}
        if len(scopes) > 1:
            reply["answer"] = "Please name one split (train, validation or test) or ask for the full-dataset total. I won't mix populations in a single rate."
            return reply
        if scopes:
            split = scopes.pop()
            candidates = [item for item in documents if item["source"] == "reports/chronological_split_summary.csv"]
            doc = next(item for item in candidates if item["text"].startswith(f"split: {split};"))
            fields = dict(field.split(": ", 1) for field in doc["text"].split("; "))
            rows, fraud = int(fields["rows"]), int(fields["fraud_rows"])
            rate = float(fields["fraud_rate_pct"])
            population = f"the frozen {split} split"
        else:
            doc = document("reports/data_quality.md", "Join and labels")
            match = re.search(r"Fraud: ([\d,]+) / ([\d,]+) = \*\*([\d.]+)%", doc["text"])
            if not match:
                return reply
            fraud, rows = (int(value.replace(",", "")) for value in match.group(1, 2))
            rate = float(match.group(3))
            population = "the full historical IEEE-CIS labelled dataset (all splits)"
        if not (rows > 0 and 0 <= fraud <= rows and math.isfinite(rate) and abs(100 * fraud / rows - rate) < 0.0001):
            return reply
        if fraud_rate:
            answer = f"The observed fraud-label rate in {population} is {rate:.4f}%: {fraud:,} fraud-labelled transactions out of {rows:,}. [1] This is dataset prevalence, not model precision, a live fraud rate or a customer-level rate."
        elif fraud_count:
            answer = f"There are {fraud:,} fraud-labelled transactions out of {rows:,} transactions in {population}. [1] These are transaction labels, not unique customers or model flags."
        else:
            answer = f"There are {rows:,} transactions in {population}. [1] This is a transaction count, not a unique-customer count or a live business total."
        reply.update(status="answered", answer=answer, sources=[source(doc, 1)])
    except (CorpusUnavailable, OSError, ValueError, KeyError, StopIteration):
        reply.update(answer="The required aggregate report is missing, changed or could not be verified. I cannot provide a reliable count or rate until it is reviewed.", sources=[])
    return reply
