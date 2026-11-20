"""A reannotation worker for `gcg_treebank` builds: `python -m txtures.gcg.worker LISTING K N`.

It reannotates every `N`-th file of the listing, starting at `K`.
"""

import sys
from pathlib import Path

from txtures.gcg.treebank import worker

if __name__ == "__main__":
    worker(Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
