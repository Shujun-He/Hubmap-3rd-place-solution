all: ## Build wheel
	zip -r --exclude="venv/*" $(shell basename $$PWD).qz .

setup: ## Install dependencies
	python3 -m venv venv
	./venv/bin/pip3 install -U pip
	./venv/bin/pip3 install wheel
	cat requirements.txt | xargs -n 1 ./venv/bin/pip3 install
	mkdir tmp

clean: ## Clean
	rm -rf tmp dist venv *.whl *.qz

help: ## List all make tasks
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
