.PHONY:

VENV_DIR = venv

ACTIVATE = source $(VENV_DIR)/bin/activate &&

# REQUIREMENTS
recompile-requirements:
	@pip-compile requirements.in

recompile-dev-requirements:
	@pip-compile dev-requirements.in

install-requirements:
	@$(ACTIVATE) pip install -r requirements.txt

update-main-dep:
	@$(ACTIVATE)
	@echo "Updating main packages..."
	@pip3 install -r requirements.in
	@echo "Main packages updated."
	@pip-compile requirements.in
	@echo "requirements.txt recompiled"

outdated:
	@$(ACTIVATE) pip3 list --outdated 

# CODE SHORTCUTS
write-imports:
	@echo "Adding imports to new Python files..."
	@python3 automate_imports.py

lint:
	@echo "Checking Black changes..."
	@black --diff .

	@echo "Running Black..."
	@black .

	@echo "Running Isort..."
	@isort .

	@echo "Running Flake8..."
	@flake8 . --show-source --statistics


# PACKAGE INSTALL FOR RELATIVE IMPROTS
install-package:
	@pip install -e .

uninstall-package:
	@pip uninstall modelska2
