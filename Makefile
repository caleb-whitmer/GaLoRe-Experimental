# Copyright 2026 Caleb Whitmer

dir_env			= env
dir_src			= src
pip					= $(dir_env)/bin/pip
interpreter = $(dir_env)/bin/python3

.PHONY: clean run setup

run: 
# Run the main script using the environment interpreter
	@ $(interpreter) $(dir_src)/main.py

setup:
# Setup the environment directory
	python -m venv $(dir_env)
# Install all dependencies:
	$(pip) install opencv-python # (image handling)
	$(pip) install matplotlib 	 # (data visualization)
	$(pip) install PyQt6 				 # (display of data visualization)

clean:
# Remove the environment directory
	rm -rf $(dir_env)