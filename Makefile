.PHONY: 

VENV_DIR = venv

ACTIVATE = source $(VENV_DIR)/bin/activate &&

recompile-requirements:
	@pip-compile requirements.in

install-requirements:
	@$(ACTIVATE)
	@pip install -r requirements.txt

update-main-dep:
	@$(ACTIVATE)
	@echo "Updating main packages..."
	@pip3 install -r requirements.in
	@echo "Main packages updated."
	@pip-compile requirements.in
	@echo "requirements.txt recompiled"

outdated:
	@$(ACTIVATE) pip3 list --outdated 
