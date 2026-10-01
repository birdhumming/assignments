TEXBIN := $(dir $(shell command -v latexmk || echo /Library/TeX/texbin/latexmk))
BUILD_DIR := ../../tmp/pdfs/a2/build
OUTPUT_DIR := .
SOURCES := $(wildcard *.tex units/*.tex) style/worksheet.sty
SOURCES += $(wildcard figures/*.png)
all: $(OUTPUT_DIR)/assignment2.pdf

$(BUILD_DIR)/%.pdf: %.tex $(SOURCES) Makefile
	@mkdir -p $(BUILD_DIR)
	@if PATH="$(TEXBIN):$$PATH" command -v latexmk >/dev/null 2>&1; then \
	 PATH="$(TEXBIN):$$PATH" latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=$(BUILD_DIR) $<; \
	elif command -v tectonic >/dev/null 2>&1; then \
	 tectonic --keep-logs --outdir $(BUILD_DIR) $<; \
	else echo "Install latexmk or Tectonic."; exit 1; fi

$(OUTPUT_DIR)/assignment2.pdf: $(BUILD_DIR)/main.pdf
	@mkdir -p $(OUTPUT_DIR)
	cp $< $@

.SECONDARY:
.PHONY: all
