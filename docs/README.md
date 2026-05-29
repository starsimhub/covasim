# Covasim docs

Published site: https://docs.covasim.org

## Build locally

1. Install Covasim and doc dependencies:

```sh
pip install -e .[docs]
```

2. Install [Quarto](https://quarto.org/docs/get-started/) and the interlinks extension:

```sh
cd docs
quarto add machow/quartodoc --no-prompt
```

3. Render:

```sh
./render          # full site (API + notebooks with freeze/cache)
./preview         # live preview
./clean_all       # remove build artifacts
```

Output is written to `docs/_site/`.

## Publish

GitHub Pages publishing (maintainers):

```sh
./publish
```

Or use the **Publish Covasim docs** GitHub Actions workflow.
