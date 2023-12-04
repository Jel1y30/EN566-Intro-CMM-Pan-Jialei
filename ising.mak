# Default PART value
PART ?= 1

.PHONY: ising
ising:
	python ising.py --part=$(PART)
