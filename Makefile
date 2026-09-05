TESTS = data/input/tests
RUN = uv run python -m src

install:
	uv sync

run:
	@uv run python -m src

debug:
	uv run python -m pdb -m src

test:
	$(RUN) --functions_definition $(TESTS)/functions_all.json \
		--input $(TESTS)/prompts_all.json
	@echo "===== invalid input tests ====="
	! $(RUN) --functions_definition $(TESTS)/broken_functions_invalid_json.json
	! $(RUN) --functions_definition $(TESTS)/broken_functions_not_array.json
	! $(RUN) --functions_definition $(TESTS)/broken_functions_empty_array.json
	! $(RUN) --functions_definition \
		$(TESTS)/broken_functions_missing_description.json
	! $(RUN) --input $(TESTS)/broken_prompts_invalid_json.json
	! $(RUN) --input $(TESTS)/broken_prompts_not_array.json
	! $(RUN) --functions_definition $(TESTS)/does_not_exist.json
	! $(RUN) --input $(TESTS)/does_not_exist.json

clean:
	rm -rf __pycache__ src/__pycache__
	rm -rf .mypy_cache .pytest_cache

lint:
	uv run flake8 .
	uv run mypy . 	--warn-return-any --warn-unused-ignores \
					--ignore-missing-imports --disallow-untyped-defs \
					--check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

.PHONY: install run debug test clean lint lint-strict
