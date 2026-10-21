# Hotel Takia CloudPOS — Railway deployment guide (development/test only)

## Important limitations
This release is **not production-ready**, **not FRCS VMS accredited**, and does not actually post room charges to Preno or accounting entries to MYOB. Use fictional transactions only. Do not expose actual guest information. The /health endpoint checks application responsiveness, not database connectivity. No automated backups, database migrations, rate limiting, or full operational hardening are included.

## 1. Prepare source repository
1. Extract this ZIP. Use the `takia_cloudpos` folder as the repository root (the folder containing `Dockerfile`, `railway.json` and `requirements.txt`).
2. Create a **private** GitHub repository, e.g. `hotel-takia-cloudpos`.
3. Upload/push the contents of the folder. **Never commit `.env`, passwords, database dumps, or API keys.** The `.gitignore` excludes `.env`.

## 2. Create Railway project
1. Log in to https://railway.com/ and create a project.
2. Deploy from your private GitHub repository; select the `takia_cloudpos` root if required.
3. Add a PostgreSQL database service from Railway's database templates (name it `Postgres` to match the example reference below).
4. In the **application** service Variables, set:
   - `DATABASE_URL` = `${{Postgres.DATABASE_URL}}` (Railway reference variable; use the actual database service name). Use the private/internal connection URL, not a public database URL.
   - `SECRET_KEY` = long cryptographically random string, generated locally, e.g. `python -c "import secrets; print(secrets.token_urlsafe(48))"`
   - `ADMIN_EMAIL` = administrator email
   - `ADMIN_PASSWORD` = unique strong password (at least 12 characters; recommend 20+)
   - `CORS_ORIGINS` = `https://pos.hoteltakia.com`
5. Do **not** set a public database domain. Keep Postgres private. No `POSTGRES_PASSWORD` is needed in the application service when using Railway Postgres.
6. Deploy. The `Dockerfile` starts Uvicorn on port 8000. In Railway's service networking settings set the service port to **8000** if prompted. Railway supplies the HTTPS ingress and routes to this port.
7. Check deployment logs. Test the Railway-provided HTTPS domain: `/health` should return `{"status":"ok"}`. Verify the login page and admin login with **test data only**.

## 3. Connect the domain without changing the website
1. In Railway application service > Settings > Networking > Custom Domain, add `pos.hoteltakia.com`.
2. Railway will display the **exact** DNS record (usually CNAME target) and any verification records. **Copy exactly what Railway shows**; do not invent a target.
3. In the DNS provider authoritative for `hoteltakia.com` (Hostinger, Cloudflare or other), add/edit **only** the `pos` record to Railway's target. If a `pos` A/AAAA/CNAME record exists, check before replacing it.
4. **Do not modify** root `@`, `www`, MX, NS, SPF/DKIM/DMARC or other existing records.
5. Wait for DNS and Railway-managed TLS certificate issuance; test `https://pos.hoteltakia.com/health` and browser login. Do not send live sales until regulatory and integration work is finished.

## 4. Verify and secure
- Confirm `/health`, login, 3 outlets and a test sale; check that the sale remains after a redeploy.
- Confirm Postgres is not publicly exposed and app secrets are set only in Railway Variables.
- Enable PostgreSQL backups with an off-platform restore-tested copy and set up uptime/log monitoring before production.
- Do not connect real Preno/MYOB credentials yet; connectors are not implemented.
- Current code creates database tables at startup (`create_all`); implement Alembic migrations before evolving a live schema.
- Do not enable real payments or issue purported FRCS fiscal receipts: fiscalisation/accreditation remains outstanding.

## Troubleshooting
- `RuntimeError: Set SECRET_KEY`: configure `SECRET_KEY` in application Variables.
- Database connection refused: confirm PostgreSQL service is healthy and `DATABASE_URL` references its **private** URL; redeploy application.
- 502/healthcheck failure: inspect Railway deploy logs and ensure exposed service port is 8000.
- Domain pending: verify DNS at the authoritative DNS provider and follow Railway's exact CNAME/verification instructions.
- Admin login fails: the initial admin account is created only when absent; changing `ADMIN_PASSWORD` later does not reset an existing account.

## Local test (optional)
Use `docker compose up -d --build` with `.env` as described in `INSTALLATION.md`; the Railway deployment uses its own managed PostgreSQL service, not the local compose database.
