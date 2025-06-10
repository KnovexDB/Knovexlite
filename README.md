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
poetry install
```

If your `poetry.lock` file is too old, use this.
```
poetry update --lock
```

Currently, we don't track the `poetry.lock` file in the early development stage

## Soft Query Dataset

The project can load uncertain query datasets from
the [Soft Queries on Uncertain KG](https://github.com/HKUST-KnowComp/Soft-Queries-on-Uncertain-KG)
repository. Use `SoftPyGAADataset` and `SoftPyGAACollator` from
`knovex.utils.dataloader` to work with these JSON files.
