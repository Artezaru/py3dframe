# 0. Setting directories for the documentation
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs/source
BUILDDIR      = docs/build

# 1. Default help command to list available Sphinx options
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# 2. Generate HTML documentation
html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html

# 3. Generate LaTeX PDF documentation
latexpdf:
	$(SPHINXBUILD) -b latex $(SOURCEDIR) $(BUILDDIR)/latex
	cd $(BUILDDIR)/latex && pdflatex py3dframe.tex && pdflatex py3dframe.tex

# 4. Clean the documentation
clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O);
	cd $(BUILDDIR); git worktree add -f html gh-pages