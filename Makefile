SOURCES = milk/*/*.cpp

debug: $(SOURCES)
	DEBUG=2 python setup.py build --build-lib=.

fast: $(SOURCES)
	python setup.py build --build-lib=.

clean:
	rm -rf build milk/*/*.so

tests: debug
	nosetests -vx

docs:
	rm -rf build/docs
	cd docs && make html && cp -r build/html ../build/docs
	@echo python setup.py upload_docs

.PHONY: clean docs tests fast debug

