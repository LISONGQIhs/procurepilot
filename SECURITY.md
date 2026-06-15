# Security

ProcurePilot is an MVP demonstration project and should not be connected to production procurement, finance, vendor, or HR systems without additional security work.

## Sensitive Data

Do not commit:

- `.env` files.
- API keys or service credentials.
- Real employee, supplier, budget, contract, invoice, or purchase-order data.
- Internal policies that are not approved for public release.

## Current Limitations

The MVP uses in-memory storage, local fixture data, no authentication, and simulated business tools. Production deployment would require authentication, authorization, audit retention, secret management, data classification, rate limits, and persistent storage.
