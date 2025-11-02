.PHONY: sample explore modify model assess all
PY?=python

sample:
	$(PY) src/stages/sample.py
explore:
	$(PY) src/stages/explore.py
modify:
	$(PY) src/stages/modify.py
model:
	$(PY) src/models/train.py
assess:
	$(PY) src/assess/assess.py
all: sample explore modify model assess
