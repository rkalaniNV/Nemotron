"""A6 — is the oracle itself falsifiable?

A1 removed 230 lines and declared the remaining 877 — backend 465, assertions 182,
tools 162, fixtures 68 — to be ground truth that cannot be cut. That was never measured.
A1 simply did not touch those files, and the study has quoted the number ever since.

This arm measures it for the largest of them. Corrupt `backend.py` one edit at a time and
ask every check the pack has whether it notices: the validation cases, the twelve
oracle-validation checks, A0's replayed traces, the pack's own assertions, and finally a
full pipeline run to `published` and tier. A mutant that survives all of them is a line
the pack does not pin — the backend says it, and nothing in the benchmark depends on it
being true.

This is A4's question asked one level down. A4 corrupted an *episode* and asked whether
the assertions noticed; A6 corrupts the *oracle* and asks whether anything at all notices.
No model is involved in either.
"""
