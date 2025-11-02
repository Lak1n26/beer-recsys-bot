PROJECT_ROOT := /Users/vkirpichov/arcadia/junk/ai_productivity_week/dlyapin

POSTGRES_HOST ?= 127.0.0.1
POSTGRES_PORT ?= 5432
POSTGRES_DB ?= $(or $${POSTGRES_DB},postgres)
POSTGRES_USER ?= $(or $${POSTGRES_USER},postgres)
POSTGRES_PASSWORD ?= $(or $${POSTGRES_PASSWORD},postgres)

.PHONY: run pgcli db-reset

pgcli:
	PGGSSENCMODE=disable \
	PGPASSWORD=$(POSTGRES_PASSWORD) \
	pgcli postgresql://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@$(POSTGRES_HOST):$(POSTGRES_PORT)/$(POSTGRES_DB)

db-reset:
	docker compose down -v
	docker compose up -d
