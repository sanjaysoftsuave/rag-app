# API rate limits and keys — Northwind Cloud

> Synthetic sample document for rag-app's quick start. Not a real company's policy.

The Free plan allows 60 requests per minute per API key. The Pro plan allows
600 requests per minute. Requests over the limit receive HTTP 429 with a
Retry-After header giving the number of seconds to wait.

Rotating an API key revokes the old key immediately. A service that keeps
sending the old key will receive HTTP 401 until its configuration is updated
with the new one.
