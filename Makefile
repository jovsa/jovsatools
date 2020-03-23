#####################################################################
#						AUTO GENERATED COMMANDS
#####################################################################
SRC = $(wildcard nbs/*.ipynb)

all: jsmltools docs

jsmltools: $(SRC)
	nbdev_build_lib
	touch jsmltools

docs_serve: docs
	cd docs && bundle exec jekyll serve

docs: $(SRC)
	nbdev_build_docs
	touch docs

test:
	nbdev_test_nbs

release: pypi
	nbdev_bump_version

pypi: dist
	twine upload --repository pypi dist/*

dist: clean
	python setup.py sdist bdist_wheel

clean:
	rm -rf dist

#####################################################################
#							CUSTOM COMMANDS
#####################################################################
VENV = venv-jsmltools

# create local env
local_env:
	# delete old venv (if it exists)
	if [ -d $(VENV) ]; then rm -r $(VENV); fi
	# create venv
	python3 -m virtualenv $(VENV)
	# script that:
	# 	1 - sources venv
	# 	2 - install pacages from settings.ini (via setup.py)
	#	3 - creates jupyter kernel used to run nbdev scripts.
	# 		Note: kernel is names `python3` since this is the
	# 		default kernel name used in nbdev.
	#		nbdev code link: https://github.com/fastai/nbdev/blob/master/nbdev/test.py#L77
	( \
		source $(VENV)/bin/activate; \
		pip install .; \
		ipython kernel install --name "python3" --user; \
	)
	echo "source $(VENV) and you are all set!"

trust:
	nbdev_install_git_hooks

format:
	black . -v

update:
	nbdev_build_lib && nbdev_build_docs && nbdev_clean_nbs && nbdev_test_nbs