# Default value for PART
PART 1 = default_part

# Default target
all: rnumbers rwalk diffusion gases

# Targets for each command
rnumbers:
	make rnumbers PART=$(PART 1)

rwalk:
	make rwalk PART=$(PART 2)

diffusion:
	make diffusion PART=$(PART 3)

gases:
	make gases PART=$(PART 4)

# Phony targets
.PHONY: all rnumbers rwalk diffusion gases
