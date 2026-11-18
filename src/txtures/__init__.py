"""Word- and region-level features of text for psycholinguistic research.

The Python API::

    import txtures as tx

    table = tx.compute("surprisal", ["The horse raced past the barn fell."])
    table = tx.run("bk21")
    table = tx.read_table("derivatives/bk21/main-0123456789ab")

See `txtures.api` for the details.
"""

from importlib.metadata import version

__version__ = version("txtures")
__all__ = ["compute", "read_table", "run"]


def __getattr__(name: str):
    # Import the API on first use, so the command line starts without pandas.
    if name in __all__:
        from txtures import api

        return getattr(api, name)
    raise AttributeError(f"module 'txtures' has no attribute {name!r}")
