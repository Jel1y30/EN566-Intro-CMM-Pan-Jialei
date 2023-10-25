FXN ?= sin
TXT ?= data.txt
FMT ?= jpeg

.PHONY: plot write read

plot:
	python trigonometry.py --function=$(FXN)

write:
	python trigonometry.py --function=$(FXN) --write=$(TXT)

read:
	python trigonometry.py --read_from_file=$(TXT) --print=$(FMT)
