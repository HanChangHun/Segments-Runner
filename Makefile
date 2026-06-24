.PHONY: test build publish clean
test:
	uv run pytest
build: clean
	uv build
publish: build
	uv run twine upload -r devpi-stable dist/*
clean:
	rm -rf dist build *.egg-info
