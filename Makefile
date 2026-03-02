# Setting default
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs/source
BUILDDIR      = docs/build
VENV          := $(CURDIR)/venv

current_branch := $(shell git rev-parse --abbrev-ref HEAD)

message ?= Default-commit-message
level ?= patch

# Default help command to list available Sphinx options
help:
	@echo "Available commands:"
	@echo "  help       - Show this help message"
	@echo "  html       - Generate HTML documentation with sphinx-build (output at docs/build/html/)"
	@echo "  clean      - Clean the documentation build directory docs/build/"
	@echo "  bump       - Update the version of the package (default: patch, use level=major/minor/patch)"
	@echo "  git        - Commit and push changes to master (use message='Your commit message')"
	@echo "  test       - Run the tests of the package with pytest"

.PHONY: help Makefile

# Check the Tests
test:
	@. $(VENV)/bin/activate && pytest tests

# Update the version of the package
bump:
	bumpver update --$(level) --no-fetch

# Clean the documentation
clean:
	@echo "Cleaning up generated files at docs/source/_autosummary/"
	@rm -rf docs/source/_autosummary
	@echo "Cleaning up generated files at docs/source/_gallery/"
	@rm -rf docs/source/_gallery
	@echo "Cleaning up generated files at docs/source/_gallery_backreferences/"
	@rm -rf docs/source/_gallery_backreferences
	@echo "Cleaning up generated file name docs/source/sg_execution_times.rst"
	@rm -f docs/source/sg_execution_times.rst
	@echo "Cleaning up generated build documentation at docs/build/"
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O);
	@echo "Recreating necessary directories..."
	cd $(BUILDDIR); mkdir -p html; mkdir -p latex
	@echo "Clean complete."

# Generate HTML documentation
html:
	@echo "Generating HTML documentation at $(BUILDDIR)/html/"
	$(SPHINXBUILD) -v -b html $(SOURCEDIR) $(BUILDDIR)/html

# Git Push origin Master
git:
	git checkout $(current_branch)
	git add -A .
	git commit -m "$(message)"
	git push origin $(current_branch)



