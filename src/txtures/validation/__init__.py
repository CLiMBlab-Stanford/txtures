"""Benchmarks of txtures against reference implementations: ModelBlocks and the Berkeley parser.

These reproduce, outside the test suite, the checks recorded in txtures'
validation evidence: the gcg15 reannotation against ModelBlocks' Perl on
all of WSJ, the latent-variable PCFG trainer against Berkeley's
`GrammarTrainer`, the left-corner compilation against ModelBlocks'
`ccmodel2synproccptmodel`, and the synproc parser against ModelBlocks'
`synproc`. Run them with `python -m txtures.validation COMMAND`; `--help`
lists the commands, which run in the order of the pipeline. Long ones are
meant to run as Slurm jobs, for example
`sbatch --wrap "python -m txtures.validation berkeley-train 2"`.

They need:

- `TXTURES_VALIDATION_WORK`: a directory for everything they write. Much
  of it is derived from the licensed Penn Treebank, so keep it out of
  repositories and shared places.
- `TXTURES_MODELBLOCKS`: a ModelBlocks checkout (its scripts, sources, and
  headers are used as they are; the C++ is built in the work directory).
- the `corpora.ptb3` setting: a copy of the Penn Treebank 3.
- `perl`, `python3`, `java` and `javac` (1.8 or later), `g++` (C++17), and
  `git` on the path; `berkeley-build` clones the Berkeley parser.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from txtures.errors import TxturesError

DATA = Path(__file__).with_name("data")
BERKELEY_REPOSITORY = "https://github.com/slavpetrov/berkeleyparser.git"
BERKELEY_COMMIT = "0df1ae756616f1f05498d9c5747f65505eaf1f93"
BERKELEY_JAR = "BerkeleyParser-1.7.jar"


@dataclass(frozen=True)
class Paths:
    """Where the validation reads and writes."""

    work: Path
    modelblocks: Path
    ptb3: Path

    @property
    def gcg15(self) -> Path:
        return self.work / "gcg15"

    @property
    def berkeley(self) -> Path:
        return self.work / "berkeley"

    @property
    def mb(self) -> Path:
        return self.work / "modelblocks"

    @property
    def wsj(self) -> Path:
        for candidate in (
            self.ptb3 / "parsed" / "mrg" / "wsj",
            self.ptb3 / "treebank_3" / "parsed" / "mrg" / "wsj",
        ):
            if candidate.is_dir():
                return candidate
        raise TxturesError(f"{self.ptb3} has no parsed/mrg/wsj directory")

    @property
    def jar(self) -> Path:
        return self.berkeley / "src" / BERKELEY_JAR

    @property
    def classpath(self) -> str:
        return f"{self.jar}:{self.berkeley / 'classes'}"


def paths() -> Paths:
    """The paths from `TXTURES_VALIDATION_WORK`, `TXTURES_MODELBLOCKS`, and `corpora.ptb3`."""
    from txtures import api

    missing = [
        name
        for name in ("TXTURES_VALIDATION_WORK", "TXTURES_MODELBLOCKS")
        if not os.environ.get(name)
    ]
    if missing:
        raise TxturesError(
            f"set {' and '.join(missing)} (see `python -m txtures.validation --help`)"
        )
    ptb3 = api.settings()["corpora.ptb3"]
    if ptb3 is None:
        raise TxturesError("set corpora.ptb3 to a copy of the Penn Treebank 3")
    work = Path(os.environ["TXTURES_VALIDATION_WORK"]).expanduser().absolute()
    for sub in ("gcg15", "berkeley", "modelblocks"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    return Paths(work, Path(os.environ["TXTURES_MODELBLOCKS"]).expanduser(), Path(ptb3))


def run(command: str, cwd: Path | None = None) -> None:
    """Run a shell pipeline, failing on any error in it."""
    subprocess.run(["bash", "-o", "pipefail", "-ec", command], cwd=cwd, check=True)
