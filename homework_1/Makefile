PY ?= python
.PHONY: all p1 p4 p5 p5c
all: p1 p4 p5
p1:  ; cd src && $(PY) p1_kelly_check.py
p4:  ; cd src && $(PY) p4_residual_normality.py
p5:  ; cd src && $(PY) p5_classification.py && $(PY) p5_regression.py && $(PY) make_tables.py
p5c: ; cd src && $(PY) p5_classification.py && $(PY) make_tables.py
