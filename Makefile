UV_RUN=uv run --group dev
PYTEST=$(UV_RUN) pytest

test:
	$(PYTEST)

test-trace:
	$(PYTEST) pytest --full-trace
