install:
	pip install -r requirements.txt

install-mac:
	brew install cmake openmpi

try:
	python try.py

run:
	python main.py