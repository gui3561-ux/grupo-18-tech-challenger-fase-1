# Makefile — grupo-18-tech-challenger-fase-1
# Requer: https://docs.astral.sh/uv/
#
# Uso: make help

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

UV ?= uv
APP := src.main:app
HOST ?= 0.0.0.0
PORT ?= 8000
PY_SOURCES := src tests

.DEFAULT_GOAL := help

.PHONY: help
help: ## Mostra os alvos disponíveis
	@printf '\nAlvos principais:\n'
	@grep -E '^[a-zA-Z0-9_.-]+:.*?##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@printf '\nVariáveis: UV=%s HOST=%s PORT=%s\n\n' "$(UV)" "$(HOST)" "$(PORT)"

# ---------------------------------------------------------------------------
# Ambiente
# ---------------------------------------------------------------------------

.PHONY: venv
venv: ## Cria .venv com o Python gerenciado pelo uv
	$(UV) venv

.PHONY: install
install: ## Instala só dependências de runtime (uv sync)
	$(UV) sync

.PHONY: install-dev
install-dev: ## Instala runtime + ferramentas de dev (pytest, ruff, mypy)
	$(UV) sync --extra dev

.PHONY: dev
dev: install-dev ## Atalho idêntico a install-dev
	@true

.PHONY: lock
lock: ## Atualiza uv.lock a partir do pyproject.toml
	$(UV) lock

.PHONY: lock-upgrade
lock-upgrade: ## Atualiza uv.lock permitindo bump de versões dentro dos ranges
	$(UV) lock --upgrade

# ---------------------------------------------------------------------------
# Qualidade (mesmo escopo do CI: src + tests)
# ---------------------------------------------------------------------------

.PHONY: lint
lint: ## Ruff: verifica estilo e erros
	$(UV) run ruff check $(PY_SOURCES)

.PHONY: lint-fix
lint-fix: ## Ruff: aplica correções automáticas
	$(UV) run ruff check $(PY_SOURCES) --fix

.PHONY: format
format: ## Ruff: formata código
	$(UV) run ruff format $(PY_SOURCES)

.PHONY: format-check
format-check: ## Ruff: verifica formatação sem alterar arquivos
	$(UV) run ruff format --check $(PY_SOURCES)

.PHONY: typecheck
typecheck: ## Mypy no pacote src
	$(UV) run mypy src

.PHONY: check
check: lint format-check typecheck ## Lint + format check + mypy (sem testes)

.PHONY: ci
ci: lint format-check typecheck test ## Pipeline completo igual ao GitHub Actions (requer install-dev)

# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------

.PHONY: test
test: ## Pytest (config em pyproject.toml)
	$(UV) run pytest

.PHONY: test-verbose
test-verbose: ## Pytest com -v
	$(UV) run pytest -v

.PHONY: test-unit
test-unit: ## Apenas testes marcados @pytest.mark.unit
	$(UV) run pytest -m unit -v

.PHONY: test-integration
test-integration: ## Apenas testes @pytest.mark.integration
	$(UV) run pytest -m integration -v

.PHONY: test-smoke
test-smoke: ## Apenas testes @pytest.mark.smoke
	$(UV) run pytest -m smoke -v

.PHONY: test-schema
test-schema: ## Apenas testes @pytest.mark.schema
	$(UV) run pytest -m schema -v

# ---------------------------------------------------------------------------
# Aplicação
# ---------------------------------------------------------------------------

.PHONY: run
run: ## Sobe a API com reload (http://$(HOST):$(PORT))
	$(UV) run uvicorn $(APP) --reload --host $(HOST) --port $(PORT)

.PHONY: run-prod
run-prod: ## Sobe sem reload (porta $(PORT))
	$(UV) run uvicorn $(APP) --host $(HOST) --port $(PORT)

# ---------------------------------------------------------------------------
# Docker (imagem na raiz do repositório)
# ---------------------------------------------------------------------------

IMAGE_NAME ?= churn-api:local

.PHONY: docker-build
docker-build: ## docker build -t $(IMAGE_NAME) .
	docker build -t $(IMAGE_NAME) .

# ---------------------------------------------------------------------------
# Limpeza
# ---------------------------------------------------------------------------

.PHONY: clean
clean: ## Remove caches locais (__pycache__, .pytest_cache, ruff, mypy)
	@rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage 2>/dev/null || true
	@find . -type d -name '__pycache__' -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
	@printf 'Caches removos ( .venv mantido ).\n'

.PHONY: clean-venv
clean-venv: ## Remove o diretório .venv (recrie com: make venv install-dev)
	@rm -rf .venv
	@printf '.venv removido.\n'
