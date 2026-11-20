"""The `gcg_treebank` resource: Penn Treebank sections reannotated into gcg15.

The Penn Treebank is licensed by the LDC, so txtures neither ships it nor
downloads it. A lab points the `corpora.ptb3` site setting at its copy of
Treebank-3 (LDC99T42), or a request names a directory with `treebank`. The
resource is identified by the contents of the `.mrg` files it reads, never by
their path, so moving the copy rebuilds nothing; and its trees, being derived
from the treebank, stay in the resource store.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from txtures.errors import TxturesError
from txtures.gcg import ARTICLE, GCG15_VERSION, ptb_lines, reannotate
from txtures.modules.base import Resource, RunContext, Workload
from txtures.store import read_json, write_json

TREES_FILE = "trees.txt.gz"
PROPERTIES_FILE = "properties.json"
WSJ = Path("parsed") / "mrg" / "wsj"
DEFAULT_SECTIONS = tuple(f"{n:02d}" for n in range(2, 22))
"""WSJ sections 02-21, the conventional training set."""

_MISSING = (
    "gcg_treebank needs the Penn Treebank 3 (LDC99T42), which txtures cannot provide; "
    "point the corpora.ptb3 site setting at a licensed copy "
    "(txtures site set corpora.ptb3=DIRECTORY), or set treebank to its directory"
)


def wsj_directory(spec: Mapping[str, Any], context: RunContext) -> Path:
    """The `parsed/mrg/wsj` directory of the requested treebank.

    `treebank` is `ptb3` (the site's copy) or the absolute path of a copy.
    Either Treebank-3's root or its `treebank_3` directory may be given.
    """
    name = spec.get("treebank", "ptb3")
    if Path(name).is_absolute():
        root = Path(name)
    elif name in context.corpora:
        root = Path(context.corpora[name])
    else:
        raise TxturesError(_MISSING)
    for candidate in (root / WSJ, root / "treebank_3" / WSJ):
        if candidate.is_dir():
            return candidate
    raise TxturesError(f"{root} has no {WSJ} directory; {_MISSING}")


def section_files(wsj: Path, sections: Sequence[str]) -> list[str]:
    """The `.mrg` files of the sections, relative to `wsj`, in treebank order."""
    files = []
    for section in sections:
        found = sorted((wsj / section).glob("*.mrg"))
        if not found:
            raise TxturesError(f"WSJ section {section} has no .mrg files in {wsj}")
        files += [path.relative_to(wsj).as_posix() for path in found]
    return files


def files_digest(wsj: Path, files: Sequence[str]) -> str:
    hasher = hashlib.sha256()
    for name in files:
        hasher.update(name.encode() + b"\0" + (wsj / name).read_bytes() + b"\0")
    return hasher.hexdigest()


class GcgTreebank(Resource):
    """WSJ sections of the Penn Treebank in gcg15, one tree per line.

    Spec keys: `treebank` (`ptb3`, the `corpora.ptb3` site setting, or the
    absolute path of a copy) and `sections` (two-digit WSJ sections; default
    02-21). Each file's trees follow a `!ARTICLE` line, as in ModelBlocks.
    Trees whose reannotation fails are `FAIL` trees over their words, as
    ModelBlocks writes them.
    """

    kind = "gcg_treebank"
    version = 1

    def resolve(self, context: RunContext) -> dict[str, Any]:
        sections = [f"{int(s):02d}" for s in self.spec.get("sections", DEFAULT_SECTIONS)]
        wsj = wsj_directory(self.spec, context)
        files = section_files(wsj, sections)
        return {
            "grammar": "gcg15",
            "gcg_version": GCG15_VERSION,
            "sections": sections,
            "files": files,
            "revision": "sha256:" + files_digest(wsj, files),
        }

    def workload(self, resolved: Mapping[str, Any]) -> Workload:
        return Workload(slurm=True, ncores=16, memory_gb=16, time_hours=4, ngpus=0)

    def build(self, resolved: Mapping[str, Any], directory: Path, context: RunContext) -> None:
        wsj = wsj_directory(self.spec, context)
        files = list(resolved["files"])
        if "sha256:" + files_digest(wsj, files) != resolved["revision"]:
            raise TxturesError(f"the treebank in {wsj} changed after planning; run again")
        parts = directory / "parts"
        parts.mkdir()
        _run_workers(wsj, files, parts, max(1, min(_cores(), len(files))))
        trees = fails = 0
        with gzip.open(directory / TREES_FILE, "wt", encoding="utf-8") as out:
            for index in range(len(files)):
                part = parts / f"{index:05d}.txt"
                for line in part.read_text(encoding="utf-8").splitlines():
                    out.write(line + "\n")
                    if line != ARTICLE:
                        trees += 1
                        fails += line.startswith("(FAIL ")
                part.unlink()
        parts.rmdir()
        write_json(directory / PROPERTIES_FILE, {"trees": trees, "fail_trees": fails})

    def load(self, resolved: Mapping[str, Any], directory: Path, context: RunContext) -> Path:
        return directory / TREES_FILE


def read_trees(path: Path) -> Iterator[str]:
    """The trees of a built `gcg_treebank`, without the `!ARTICLE` lines."""
    with gzip.open(path, "rt", encoding="utf-8") as lines:
        for line in lines:
            line = line.rstrip("\n")
            if line != ARTICLE:
                yield line


def properties(directory: Path) -> dict[str, Any]:
    return read_json(directory / PROPERTIES_FILE)


def _cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def reannotate_file(source: Path, output: Path) -> None:
    """Write `!ARTICLE` and the gcg15 trees of one `.mrg` file."""
    lines = [ARTICLE, *ptb_lines(source.read_text(encoding="utf-8"))]
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text("".join(reannotate(line) + "\n" for line in lines), encoding="utf-8")
    temporary.rename(output)


def _run_workers(wsj: Path, files: list[str], parts: Path, workers: int) -> None:
    """Reannotate the files in `workers` processes (`python -m txtures.gcg.worker`)."""
    jobs = [[str(wsj / name), str(parts / f"{i:05d}.txt")] for i, name in enumerate(files)]
    if workers == 1:
        for source, output in jobs:
            reannotate_file(Path(source), Path(output))
        return
    listing = parts / "jobs.json"
    listing.write_text(json.dumps(jobs), encoding="utf-8")
    processes = [
        subprocess.Popen(
            [sys.executable, "-m", "txtures.gcg.worker", str(listing), str(k), str(workers)]
        )
        for k in range(workers)
    ]
    failed = sum(process.wait() != 0 for process in processes)
    listing.unlink()
    if failed:
        raise TxturesError(f"{failed} of {workers} reannotation workers failed; see their output")


def worker(listing: Path, index: int, workers: int) -> None:
    """Reannotate every `workers`-th file of `listing`, starting at `index`."""
    for position, (source, output) in enumerate(json.loads(listing.read_text(encoding="utf-8"))):
        if position % workers == index:
            reannotate_file(Path(source), Path(output))
