"""A corpus worker for `ngram_corpus` builds: `python -m txtures.ngram_worker LISTING K N`.

It prepares every `N`-th job of the listing, starting at `K`. It is its own
module, outside `txtures.modules`, so that running it does not import it
twice.
"""

import sys
from pathlib import Path

from txtures.modules.ngram import _worker

if __name__ == "__main__":
    _worker(Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
