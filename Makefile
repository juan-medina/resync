.PHONY: setup auth sync dump label train

setup:
	python -m venv .venv
	.venv/Scripts/pip install -r requirements.txt

auth:
	.venv/Scripts/python scripts/setup_auth.py

sync:
	.venv/Scripts/python -m src.sync.main

dump:
	.venv/Scripts/python -m src.train.dump_dataset

label:
	.venv/Scripts/python -m src.train.ai_labeler

train:
	.venv/Scripts/python -m src.train.train_model
