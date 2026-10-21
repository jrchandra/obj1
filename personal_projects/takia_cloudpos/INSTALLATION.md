# Deployment and Installation Guide

## Requirements
Linux machine with Docker Engine and Docker Compose plugin; 2 CPU cores, 4 GB RAM suggested for a pilot. Domain + HTTPS reverse proxy for any external access.

## Commands
```bash
cp .env.example .env
# edit .env and use strong secrets; NEVER commit it
openssl rand -hex 32
docker compose up -d --build
docker compose ps
docker compose logs -f web
```
The container listens on `127.0.0.1:8000` on the Docker host. Configure a TLS reverse proxy to expose the app externally. Set `CORS_ORIGINS` to the actual HTTPS origin. The reverse proxy must pass requests to `127.0.0.1:8000`.

## Initial setup
Sign in with `ADMIN_EMAIL` and `ADMIN_PASSWORD`, add products and opening stock. Three outlets are created automatically. Check `http://localhost:8000/health` for health and `/docs` for API routes.

## Backups
```bash
docker compose exec -T db pg_dump -U takia takia > takia-backup.sql
```
Back up `.env` securely and keep encrypted offsite backups. Test restoring to a **separate** test database before relying on backups.

## Upgrade
Take a verified backup first. This prototype uses `create_all`, not versioned migrations; schema changes require an Alembic migration plan before upgrades.

## Security warning
This starter has no authentication rate limiting, CSRF-safe browser session mechanism, password reset, multi-factor authentication, transaction cancellation, concurrent stock locking or central log monitoring. Protect the instance from public access until hardened. Rotate initial passwords before testing with staff. No fiscal receipts can be issued.
