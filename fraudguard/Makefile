.PHONY: install train compare test demo explain lint security serve dashboard all

install:
	pip install -r requirements-dev.txt

train:
	python scripts/train_model.py --cards 2000 --days 90

compare:
	python scripts/train_model.py --compare --cards 1500 --days 90

test:
	PYTHONPATH=src pytest tests/ -q

demo:
	PYTHONPATH=src python scripts/demo.py

explain:
	python scripts/explain.py --transactions 300

lint:
	ruff check src/ tests/ scripts/
	mypy src/ --ignore-missing-imports

security:
	bandit -r src/ -ll
	pip-audit -r requirements.txt

serve:
	PYTHONPATH=src uvicorn fraudguard.api.app:app --host 127.0.0.1 --port 8000

dashboard:
	streamlit run dashboard/app.py

all: train test demo
