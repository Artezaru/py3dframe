# 0. Setting default
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs/source
BUILDDIR      = docs/build

current_branch := $(shell git rev-parse --abbrev-ref HEAD)

message ?= Default-commit-message
level ?= patch

# 0. Default help command to list available Sphinx options
help:
	@echo "Available commands:"
	@echo "  help       - Show this help message"
	@echo "  html       - Generate HTML documentation with sphinx-build (output at docs/build/html/)"
	@echo "  latexpdf   - Generate LaTeX PDF documentation with Sphinx sphinx-build (output at docs/build/latex/)"
	@echo "  clean      - Clean the documentation build directory docs/build/"
	@echo "  bump       - Update the version of the package (default: patch, use level=major/minor/patch)"
	@echo "  git        - Commit and push changes to master (use message='Your commit message')"
	@echo "  app        - Build the application with PyInstaller (output at dist/)"
	@echo "  test       - Run the tests of the package with pytest"

.PHONY: help Makefile

# 1. Check the Tests
test:
	@. $(VENV)/bin/activate && pytest tests

# 2. Update the version of the package
bump:
	bumpver update --$(level) --no-fetch

# 3. Clean the documentation
clean:
	@echo "Cleaning up generated files at docs/source/generated/"
	@rm -rf docs/source/generated
	@echo "Removing build directory: $(BUILDDIR)"
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O);
	@echo "Recreating necessary directories..."
	cd $(BUILDDIR); mkdir -p html; mkdir -p latex
	@echo "Clean complete."

# 4. Generate autosummary documentation
autosummary:
	@echo "Generating autosummary files in docs/source/api_doc/*.rst"
	@sphinx-autogen -o docs/source/generated/ docs/source/api_doc/*.rst
	@echo "Autosummary generation complete."
	@echo "Run python autosummary_change_titles.py to update the names in the generated files."
	@python3 docs/source/autosummary_change_titles.py

# 5. Generate HTML documentation
html:
	@echo "Generating HTML documentation at $(BUILDDIR)/html/"
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html

# 6. Git Push origin Master
git:
	git checkout $(current_branch)
	git add -A .
	git commit -m "$(message)"
	git push origin $(current_branch)

# 7. Create the application
app:
	echo "from py3dframe.__main__ import __main_gui__" > run_gui.py
	echo "__main_gui__()" >> run_gui.py
	pyinstaller --name py3dframe --onefile --windowed run_gui.py
	rm run_gui.py
	rm -rf build
	rm run_gui.spec
