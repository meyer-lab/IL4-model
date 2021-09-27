SHELL := /bin/bash

flist = 1

.PHONY: clean test all testprofile testcover spell

all: pylint.log $(patsubst %, output/figure%.svg, $(flist))

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install -Uqr requirements.txt
	touch venv/bin/activate

output/figure%.svg: venv genFigures.py src/figures/figure%.py
	mkdir -p ./output
	. venv/bin/activate && ./genFigures.py $*

lean:
	mv output/requests-cache.sqlite requests-cache.sqlite || true
	rm -rf prof output coverage.xml .coverage .coverage* junit.xml coverage.xml profile profile.svg pylint.log
	mkdir output
	mv requests-cache.sqlite output/requests-cache.sqlite || true
	rm -f profile.p* stats.dat .coverage nosetests.xml coverage.xml testResults.xml
	find -iname "*.pyc" -delete

test: venv
	. venv/bin/activate && pytest

testcover: venv
	. venv/bin/activate && pytest --junitxml=junit.xml --cov-branch --cov=src --cov-report xml:coverage.xml

pylint.log: venv common/pylintrc
	. venv/bin/activate && (pylint --rcfile=./common/pylintrc src > pylint.log || echo "pylint3 exited with $?")
