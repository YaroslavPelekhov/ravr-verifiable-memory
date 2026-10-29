# Pre-publication checklist

This directory is approved for publication as a standalone repository.

- Public release authorized by the owner on 2026-10-25.
- Repository name: `ravr-verifiable-memory`; author: Yaroslav Pelekhov.
- Apache-2.0 license added.
- Create a new random `RAVR_MEMORY_API_KEY`; never reuse an LLM provider key.
- Deploy with a persistent `/data` volume and HTTPS.
- Run the local HTTP contract tests and a hosted load smoke.
- Pin the public commit used by the deployment.
- Submit the AML access request before 2026-10-31 23:59 UTC+8.
- Access request `request_2b365dd770fd4edc8223` was approved for v1.0.1 on
  2026-10-25. Do not represent v1.0.2 as that approved version without an amendment.
- Candidate v1.0.2 passed 11 unit tests, Ruff, a 25-check local HTTP contract probe,
  and public evidence-retrieval comparisons on LoCoMo and LongMemEval-S.
- Delete evaluation data within 30 days after the run.
