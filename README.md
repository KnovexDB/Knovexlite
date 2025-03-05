# A Neural Graph Database Engine

This repo is based on pyg. Supports various fundamental infra to faciliate
the research in neural graph databases.

## Dependency

We use `poerty` to manage the package

To install poetry

```
 curl -sSL https://install.python-poetry.org | python3 -
```

To install packages
```
poetry sync
```

If your `poetry.lock` file is too old, use this.
```
poetry update --lock
```

Currently, we don't track the `poetry.lock` file.
