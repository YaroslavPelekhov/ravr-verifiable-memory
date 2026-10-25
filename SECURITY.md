# Security policy

## Supported version

Only the latest tagged release is supported.

## Reporting a vulnerability

Please report suspected vulnerabilities privately to
`yaroslavpelekhov@gmail.com`. Do not include production credentials or evaluation
data. Expect an acknowledgement within 72 hours.

## Deployment boundary

`/add` and `/search` fail closed unless `RAVR_MEMORY_API_KEY` is configured. `/health`
is intentionally unauthenticated and returns only a status and version. Deployments
must terminate TLS at the hosting provider, use a dedicated random key, mount
persistent storage at `/data`, and avoid request logging that captures payloads or
authorization headers.
