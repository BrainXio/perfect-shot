# Makefile

.PHONY: build up down logs clean test

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

run:
	docker compose up

logs:
	docker compose logs -f

clean:
	docker compose down -v --remove-orphans

test:
	docker compose run --rm app python test.py