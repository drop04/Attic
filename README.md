# Attic

A private household-document memory for Alexa+, built as a self-hosted **MCP server**
(Streamable HTTP, protocol 2025-11-25). Photograph a receipt or warranty card once;
later ask "when does my fridge warranty expire?" or "anything expiring this month?".

Documents, OCR and the database stay on the machine running Attic (designed for a
retired Android phone / small Linux box). Only the answer to each query travels
through the tunnel. An optional cloud tier (Amazon Bedrock) improves field extraction
and sees **redacted text only**.

```
Phone camera -> /upload page -> OCR (Tesseract) -> rules (+ optional Bedrock) -> you confirm -> SQLite
Alexa+ --HTTPS--> tunnel --> /mcp (bearer token) --> tools --> short, speakable answers
```

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt           # + apt install tesseract-ocr for photo OCR
cp .env.example .env                      # then: set ATTIC_TOKEN
export ATTIC_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')
python -m attic                           # http://127.0.0.1:8787
```

Open `http://<device-ip>:8787/upload` on your phone, enter the token, photograph a document,
check the extracted fields, save.

**Expose only `/mcp` to the internet.** With Cloudflare Tunnel or Tailscale Funnel, route just
that path and add the public hostname to `ATTIC_ALLOWED_HOSTS` (Host-header allow-list).
Keep `/upload` and `/api/*` on your LAN.

## MCP surface

| Tool | Purpose |
|---|---|
| `search_documents(query, doc_type?)` | Find documents by product, store, model, scanned words |
| `get_warranty_status(item)` | Days left, expiry date, model/serial, confidence, evidence, claim steps |
| `list_expiring(days=30)` | Warranties/policies expiring soon (and expired in the last 30 days) |
| `get_document(id)` | Structured fields for one document (no scanned text) |
| `explain_document(id, question?)` | Untrusted excerpt + optional Bedrock summary |
| `add_note(id, text)` | Attach a searchable note |

Resource `attic://index`; prompt `monthly_household_review`. Read tools carry
`readOnlyHint`. Every result has a short `spoken` field for voice output.

## Security model (what is and isn't done)

Done: bearer-token auth with constant-time compare; Host/Origin allow-list; scanned text only
returned in `untrusted_*` fields and never used to choose tools; LLM output is JSON-parsed and
schema-validated (a hostile document can at worst yield a wrong field, which the human-confirm
step catches); redaction (phone, GSTIN, PAN, Aadhaar-style, long digit runs, labelled
name/address lines) before any cloud call; identifiers (model/serial) are local-only;
audit log of every tool call.

**Not done yet:** encryption at rest (SQLite and stored images are plaintext; use full-disk
encryption for now); OAuth 2.1 (static bearer token only; check what Alexa+ registration
requires); PDF ingestion (images and .txt only); free-text names/addresses that are not on a
labelled line are not redacted.

## Tests

```bash
pytest -q                 # unit, HTTP and OCR tests
python -m tests.e2e_mcp   # real MCP client against a live server
```

OCR tests use synthetic rendered receipts, not crumpled thermal paper: measure accuracy on your own
documents before quoting numbers.

<!-- ## License

MIT -->
