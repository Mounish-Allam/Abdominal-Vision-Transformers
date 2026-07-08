### Report grounding evaluation (before/after RAG, n=30 slices, seed 42)

| Metric | No-RAG (legacy) | RAG-grounded | Delta |
|---|---|---|---|
| Structure adherence (Findings:/Impression: present) | 100.0% | 100.0% | +0.0pp |
| Uncertainty-flagging rate (when any organ flagged) | 28.0% | 100.0% | +72.0pp |
| Reference usage (>= 1 passage cited) | N/A | 100.0% | -- |
| Unsupported-claim rate (manual review) | 1.3% (2/153) | 0.6% (1/175) | -0.7pp |

*Unsupported-claim rate is the fraction of sentences marked `unsupported` in `outputs/claim_scoring_sheet.csv`, out of 153 no-RAG / 175 RAG sentences. See the scoring methodology note for how this sheet was filled in. Partially-supported sentences (12 no-RAG / 14 RAG) are tracked separately, not counted as unsupported.*
