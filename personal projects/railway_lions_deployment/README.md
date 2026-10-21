# Lions Club of Labasa — Railway deployment files

These files must be placed at the root of a **complete Laravel 12 project**. The previous `Lions_Club_Labasa_Digital_Portal_MVP.zip` is only an overlay and **cannot be deployed directly**.

## 1. Assemble the full project on your computer

Requirements: PHP 8.2+, Composer, Node.js 20+, Git. Laravel 12 needs a compatible PHP environment.

```bash
composer create-project laravel/laravel lions-portal "^12.0"
cd lions-portal
composer require laravel/breeze --dev
php artisan breeze:install blade
npm install
```

Copy the MVP ZIP's `app/Models`, `app/Http/Controllers`, `database/migrations`, `resources/views` and `routes/web.php` into this project, retaining the folder structure. **Do not copy the ZIP's composer.json.** Copy Dockerfile, .dockerignore and railway.json from this deployment pack to the project root.

**Before public launch:** Remove public registration routes from `routes/auth.php`; provision users privately. Review authorization policies and privacy controls. The current starter is NOT approved for storing actual medical or beneficiary records.

```bash
composer install
npm run build
php artisan route:list
```

Commit the complete assembled Laravel project to a **private GitHub repository**, including `composer.lock` and `package-lock.json`, but NEVER `.env`.

## 2. Railway setup

1. Sign in at https://railway.com and select New Project > Deploy from GitHub repo; authorize your private repository.
2. Add a PostgreSQL service to the project (New > Database > PostgreSQL).
3. In your web service > Variables set:

```
APP_NAME=Lions Club of Labasa
APP_ENV=production
APP_DEBUG=false
APP_KEY=base64:<generate locally with php artisan key:generate --show>
APP_URL=https://<your-railway-domain>
DB_CONNECTION=pgsql
DB_HOST=${{Postgres.PGHOST}}
DB_PORT=${{Postgres.PGPORT}}
DB_DATABASE=${{Postgres.PGDATABASE}}
DB_USERNAME=${{Postgres.PGUSER}}
DB_PASSWORD=${{Postgres.PGPASSWORD}}
SESSION_DRIVER=database
CACHE_STORE=database
QUEUE_CONNECTION=database
LOG_CHANNEL=stderr
```

Variable references above assume the database service is named **Postgres**. Railway variable-reference UI is safer than typing these by hand. Keep APP_KEY unchanged after setup.

4. In web service Settings > Networking, generate a public domain. Update APP_URL with the generated HTTPS URL.
5. Wait for the Docker build. If build fails, check Railway build logs and fix the first error; the Laravel/Breeze overlay may require code adjustments.
6. In Railway service shell or a one-off command (if supported), run `php artisan migrate --force` **after** database variables are configured. Do not automatically run migrations on every app start.
7. Create the first administrator only through a trusted provisioning process. Do not allow public registration or assign roles from public requests.
8. Test `/`, `/projects`, `/login`, `/portal` and database-backed pages with dummy data.

## 3. Important limitations

- This Dockerfile uses Laravel's `artisan serve` as a **test/development deployment only**; for production, replace it with Nginx + PHP-FPM or an appropriate production PHP application server.
- Railway app filesystem is ephemeral: **never store signed documents, photos or receipts locally**. Use private S3-compatible object storage with authorization and retention policies.
- Set up reliable off-platform PostgreSQL backups, restore testing, email delivery, and monitoring before production.
- The MVP needs stronger club-level authorization, invitation-only onboarding, consent/privacy workflows, anti-CSRF validation checks, and stock transaction integrity testing before using real data.
- Railway charges and included credits may change; check the current plan in your dashboard.
