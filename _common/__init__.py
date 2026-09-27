"""Shared infrastructure for the IDEA smoke tests.

Every experiment folder (NN_slug/) uses these modules so that:
  * every Jev / local-LLM call is logged verbatim (request + raw response + timestamps) to JSONL;
  * analysis scripts read ONLY those logs (plus frozen stimuli), so every number is reproducible;
  * pre-registration is frozen in an append-only ledger before any data is collected.
"""
