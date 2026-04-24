.PHONY: all report build clean help

all: report

report:
	python3 find_missing_rax_avx.py $(if $(RAX),--rax $(RAX),)

build:
	python3 find_missing_rax_avx.py $(if $(RAX),--rax $(RAX),) --no-idump

clean:
	rm -rf build

help:
	@echo "RAX AVX corpus coverage"
	@echo ""
	@echo "Targets:"
	@echo "  make report  - Build corpus, run idump, and report remaining __asm"
	@echo "  make build   - Build corpus object only"
	@echo "  make clean   - Remove generated corpus artifacts"
	@echo ""
	@echo "Variables:"
	@echo "  RAX=/path/to/rax  Source RAX checkout (auto-detects ../rax or sibling rax if omitted)"
