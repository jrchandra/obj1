# Hotel Takia CloudPOS — Development Release 0.1

**Status:** Working starter application / pilot foundation, **NOT production-ready or FRCS VMS-accredited**.

## Delivered
- Browser POS for Bar, Restaurant and Reception
- Administrator, manager and cashier roles with password authentication
- Products, VAT-inclusive prices, stock adjustments, stock deduction per sale
- Cash, EFTPOS, card and **unverified pending room-charge** sale recording
- Sales history, daily dashboard, immutable sale lines, stock/audit events
- Reconciliation CSV for accounting review (NOT an MYOB API connector or direct MYOB import template)
- Docker Compose + PostgreSQL, API documentation at `/docs`

## Not implemented — critical before live hotel use
- **Preno**: no API access, room verification or actual folio posting. Room charges remain `pending_preno` and MUST be reconciled manually; do not collect guest payment twice.
- **MYOB Business**: no OAuth/API integration or automated posting; reconciliation CSV only.
- **FRCS VMS**: no fiscal receipt, SDC connection or accreditation. All sales are marked `not_fiscalised`; DO NOT use this application as a live fiscal POS.
- No returns/voids, shift close, printer integration, table service, kitchen display, recipe costing, multi-location stock, offline sales, tax configuration UI, purchase orders, automated backups, migrations or external penetration test.
- VAT rate in code is a **demonstration assumption of 12.5%** and is not verified as current for Fiji or applicable to every item. Do not use its tax figures for filing without correction and professional review.

## Installation (Linux VPS or local Docker host)
1. Install Docker Engine and Docker Compose plugin.
2. Copy `.env.example` to `.env` and replace ALL secrets. Generate a random `SECRET_KEY` with `openssl rand -hex 32`.
3. Run `docker compose up -d --build`.
4. Open `http://localhost:8000` **from the host**. The web port is bound to loopback; for remote access configure a TLS reverse proxy (Caddy/Nginx) and domain.
5. Sign in using `ADMIN_EMAIL` and `ADMIN_PASSWORD` from `.env`.
6. Add products with initial stock under Products & stock; select an outlet to record transactions.
7. Review `/docs` for API routes. Export reconciliation data from Sales & room charges.
8. Backup database with `docker compose exec -T db pg_dump -U takia takia > backup.sql` and verify restore regularly.

**Production prerequisites:** HTTPS, firewall, dedicated non-default secrets, correct VAT setup, external integrations tested with provider credentials, fiscal approval, secure operational procedures, disaster recovery and acceptance testing.

## Suggested next milestones
1. Configurable Fiji VAT/tax rules, refunds/void approvals, cashier sessions, inventory by outlet.
2. Preno approved API integration with guest lookup, checked-in validation, folio charge posting, idempotency, retry queue and reconciliation.
3. MYOB Business OAuth, chart-of-accounts/tax mapping, balanced journals and duplicate-post prevention.
4. FRCS-approved fiscal path and accreditation/testing for the exact deployed POS + SDC combination.
5. Production hardening: database migrations, automated backups, secure cookies/token handling, monitoring, TLS, rate limiting, concurrency-safe stock reservation and audit review.

## Data & design notes
All amounts are in FJD. Database stores money as decimal, not floating-point. The backend recalculates sale prices and totals; browser totals are indicative. Sales use an idempotency key; room charges are **not** treated as Preno-confirmed. Stock and sale write in a single database transaction; however, concurrent inventory race handling needs strengthening before live use. Do not enter real guest details into a public development instance.

## Railway deployment
See `RAILWAY_DEPLOYMENT.md`. Railway uses managed PostgreSQL and the Dockerfile; do not deploy `docker-compose.yml` as a Railway service.
