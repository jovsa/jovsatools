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
	ipython kernel install --name "python3" --user
	nbdev_test_nbs

release: pypi
	nbdev_bump_version

pypi: dist
	twine upload --repository pypi dist/*

dist: clean
	python setup.py sdist bdist_wheel

clean:
	rm -rf dist