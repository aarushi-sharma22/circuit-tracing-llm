.PHONY: venv run

venv:
	bash scripts/env.sh

run:
	. .venv/bin/activate && python run_direct_analysis.py --prompt "55+56=" --layer 0 --head 0
