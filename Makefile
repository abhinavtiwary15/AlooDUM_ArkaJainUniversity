.PHONY: demo api web test lint clean

demo:
	python scripts/run_demo.py

api:
	python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

web:
	cd web && npm run dev

test:
	python -m pytest tests/
	cd web && npm test

lint:
	python -m ruff check .
	cd web && npm run lint
