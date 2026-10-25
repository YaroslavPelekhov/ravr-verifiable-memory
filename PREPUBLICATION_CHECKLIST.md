# Pre-publication checklist

This directory is a release candidate, not yet an authorized public repository.

- Confirm that public release does not conflict with double-blind submissions.
- Confirm the repository name and author identity to display.
- Choose and add an explicit software license.
- Create a new random `RAVR_MEMORY_API_KEY`; never reuse an LLM provider key.
- Deploy with a persistent `/data` volume and HTTPS.
- Run the local HTTP contract tests and a hosted load smoke.
- Pin the public commit used by the deployment.
- Submit the AML access request before 2026-10-31 23:59 UTC+8.
- Delete evaluation data within 30 days after the run.
