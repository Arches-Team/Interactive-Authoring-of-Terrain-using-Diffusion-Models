
SRC_DIR  := src
VENV_DIR := venv

ifeq ($(OS),Windows_NT)
	PYTHON := python
	PIP    := pip
	VENV   := . $(VENV_DIR)/Scripts/activate;
else
	PYTHON := python3
	PIP    := pip3
	VENV   := . $(VENV_DIR)/bin/activate;
endif

RUN := $(VENV) $(PYTHON) -m

install: venv
	$(VENV) $(PIP) install -Ur requirements.txt

install-dev: install
	$(VENV) $(PIP) install -Ur requirements_dev.txt

venv:
	test -d $(VENV_DIR) || $(PYTHON) -m venv $(VENV_DIR)

mask:
	$(RUN) terrain_diffusion.collection.mask

download:
	$(RUN) terrain_diffusion.collection.downloader

dataset:
	$(RUN) terrain_diffusion.core.terrain_dataset --generate_sketch --generate_elevation --ignore_satellite --output_sizes 32 256

archives:
	$(RUN) terrain_diffusion.core.terrain_dataset --archive_patterns elevation/32x32.png --archive_name data/datasets/elevation-32.tar.gz
	$(RUN) terrain_diffusion.core.terrain_dataset --archive_patterns elevation/256x256.png --archive_name data/datasets/elevation-256.tar.gz
	
	$(RUN) terrain_diffusion.core.terrain_dataset --archive_patterns derivative/32x32.png --archive_name data/datasets/derivative-32.tar.gz
	$(RUN) terrain_diffusion.core.terrain_dataset --archive_patterns derivative/256x256.png --archive_name data/datasets/derivative-256.tar.gz

	$(RUN) terrain_diffusion.core.terrain_dataset --archive_patterns sketch/256x256.png --archive_name data/datasets/sketch-256.tar.gz
	
	$(RUN) terrain_diffusion.core.terrain_dataset --archive_patterns satellite/1024x1024.jpg --archive_name data/datasets/satellite-1024.tar.gz

blender-addon:
	$(RUN) app.create_addon

clean-pycache:
	find $(SRC_DIR) -type d -name __pycache__ -exec rm -r {} +

clean: clean-pycache
	rm -rf $(VENV_DIR)
