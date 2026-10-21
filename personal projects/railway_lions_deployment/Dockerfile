FROM node:22-alpine AS assets
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM php:8.3-cli-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends git unzip libpq-dev libzip-dev libicu-dev && docker-php-ext-install pdo_pgsql pgsql zip intl bcmath && rm -rf /var/lib/apt/lists/*
COPY --from=composer:2 /usr/bin/composer /usr/local/bin/composer
WORKDIR /var/www/html
COPY composer.json composer.lock ./
RUN composer install --no-dev --prefer-dist --no-interaction --no-progress --optimize-autoloader --no-scripts
COPY . .
COPY --from=assets /app/public/build /var/www/html/public/build
RUN composer dump-autoload --optimize && mkdir -p storage/framework/{sessions,views,cache} storage/logs bootstrap/cache && chmod -R ug+rwX storage bootstrap/cache
ENV APP_ENV=production APP_DEBUG=false LOG_CHANNEL=stderr SESSION_DRIVER=database CACHE_STORE=database QUEUE_CONNECTION=database
EXPOSE 8080
CMD ["sh", "-c", "php artisan optimize:clear && php artisan serve --host=0.0.0.0 --port=${PORT:-8080}"]
