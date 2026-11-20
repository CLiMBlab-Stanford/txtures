"""Non-Python tools, such as KenLM, run from an Apptainer image.

Every program txtures needs that is not a Python package lives in one image,
built from `tools.def` in this package. The image is the `tools_image`
resource: its identity is the SHA-256 of the definition, so editing the
definition makes a new image, and resources built with the image depend on
it. It is built as a sandbox directory in the resource store and never
opened writable.

Apptainer itself is txtures' own pinned copy, a relocatable conda-forge
build (the one nro uses), installed on first use in
`RESOURCES/apptainer/`. Setting `execution.runtime` to an Apptainer
executable uses that instead. The host's Apptainer is never used implicitly.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from txtures.errors import TxturesError
from txtures.modules.base import Resource, RunContext, Workload
from txtures.store import lock, make_shared_dir, now, read_json, write_json

DEFINITION = Path(__file__).with_name("tools.def")
"""The image definition packaged with txtures."""

MANAGED_RUNTIME = {
    "name": "apptainer",
    "version": "1.4.5",
    "distribution": "conda-forge",
    "architecture": "x86_64",
    "minimum_glibc": "2.17",
    "artifact": "apptainer-1.4.5-conda-glibc217-x86_64.tar.gz",
    "sha256": "6659c298aacfaf5ada67f2358ee97468aa685cc4c1d76bd75da08bb6e2dca4f6",
    "source": (
        "https://github.com/CLiMBlab-Stanford/nro/releases/download/"
        "resources-apptainer-1.4.5-conda-glibc217-x86_64-v1/"
        "apptainer-1.4.5-conda-glibc217-x86_64.tar.gz"
    ),
}
"""The pinned Apptainer bundle, a `conda-pack` archive of conda-forge's `apptainer=1.4.5`."""

RUNTIME_MANIFEST = ".txtures-runtime.json"
IMAGE_MANIFEST = "tools.json"
SANDBOX = "sandbox"
BUILD_FLAGS = ("--fakeroot", "--ignore-fakeroot-command")
"""Unprivileged build in a root-mapped namespace. Ignoring the fakeroot command
keeps the host's `libfakeroot`, which needs the host's glibc, out of the build."""


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def definition_digest(definition: Path | None = None) -> str:
    """The SHA-256 of an image definition (by default `DEFINITION`), which identifies the image."""
    return sha256(DEFINITION if definition is None else definition)


# The Apptainer runtime


def runtime_directory(resources: Path) -> Path:
    """Where the managed runtime is installed in a resources directory."""
    spec = MANAGED_RUNTIME
    name = f"apptainer-{spec['version']}-{spec['architecture']}-{spec['sha256'][:16]}"
    return resources / "apptainer" / name


def require_platform() -> None:
    """Refuse hosts the managed runtime cannot run on.

    Raises:
        TxturesError: The host is not Linux x86_64 with glibc 2.17 or newer.
    """
    message = "the managed Apptainer runs only on Linux x86_64 with glibc 2.17 or newer"
    if platform.system() != "Linux" or platform.machine() != MANAGED_RUNTIME["architecture"]:
        raise TxturesError(f"{message}; set execution.runtime to an Apptainer executable")
    library, version = platform.libc_ver()
    try:
        actual = tuple(int(part) for part in version.split("."))
    except ValueError:
        actual = ()
    required = tuple(int(part) for part in MANAGED_RUNTIME["minimum_glibc"].split("."))
    if library != "glibc" or actual < required:
        raise TxturesError(f"{message}; set execution.runtime to an Apptainer executable")


def probe(executable: Path) -> str:
    """Return the output of `apptainer --version`.

    Raises:
        TxturesError: The executable does not run.
    """
    try:
        result = subprocess.run(
            [str(executable), "--version"], capture_output=True, text=True, timeout=60
        )
    except OSError as error:
        raise TxturesError(f"cannot run Apptainer at {executable}: {error}") from error
    if result.returncode != 0:
        raise TxturesError(f"{executable} --version failed: {result.stderr.strip()}")
    return result.stdout.strip()


def managed_runtime(resources: Path) -> Path | None:
    """The managed runtime's executable if it is installed and intact, else `None`."""
    root = runtime_directory(resources)
    try:
        manifest = read_json(root / RUNTIME_MANIFEST)
    except (OSError, ValueError):
        return None
    executable = root / "bin" / "apptainer"
    if manifest.get("specification") != MANAGED_RUNTIME or not executable.is_file():
        return None
    if manifest.get("executable_sha256") != sha256(executable):
        return None
    return executable


def _download(url: str, target: Path, checksum: str) -> None:
    if not url.startswith("https://"):
        raise TxturesError(f"refusing to download {url} without HTTPS")
    make_shared_dir(target.parent)
    staged = target.with_name(f".{target.name}.{os.getpid()}.part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, staged.open("wb") as out:
            shutil.copyfileobj(response, out, 1 << 20)
        if sha256(staged) != checksum:
            raise TxturesError(f"the download of {url} does not match its pinned checksum")
        os.replace(staged, target)
    except OSError as error:
        raise TxturesError(f"cannot download {url}: {error}") from error
    finally:
        staged.unlink(missing_ok=True)


def install_runtime(resources: Path, progress=print) -> Path:
    """Install the managed runtime in `resources` unless it is there; return its executable.

    The archive is checked against its pinned SHA-256, extracted into a
    staging directory beside the final one, probed, and renamed into place,
    with its manifest written before the rename, so a reader sees either no
    runtime or a complete one. Concurrent installs wait on a lock.

    Raises:
        TxturesError: The host cannot run it, the download fails or does not
            match, or the extracted runtime does not run.
    """
    found = managed_runtime(resources)
    if found is not None:
        return found
    require_platform()
    root = runtime_directory(resources)
    make_shared_dir(root.parent)
    with lock(root.with_name(root.name + ".lock")):
        found = managed_runtime(resources)
        if found is not None:
            return found
        archive = root.parent / "artifacts" / MANAGED_RUNTIME["artifact"]
        if not archive.is_file() or sha256(archive) != MANAGED_RUNTIME["sha256"]:
            progress(f"downloading Apptainer {MANAGED_RUNTIME['version']}")
            _download(MANAGED_RUNTIME["source"], archive, MANAGED_RUNTIME["sha256"])
        if root.exists():
            shutil.rmtree(root)
        staging = Path(tempfile.mkdtemp(dir=root.parent, prefix=".stage-"))
        try:
            with tarfile.open(archive, "r:gz") as stream:
                stream.extractall(staging, filter="data")
            executable = staging / "bin" / "apptainer"
            output = probe(executable)
            if MANAGED_RUNTIME["version"] not in output:
                raise TxturesError(f"the managed Apptainer reports {output!r}")
            write_json(
                staging / RUNTIME_MANIFEST,
                {
                    "specification": MANAGED_RUNTIME,
                    "executable_sha256": sha256(executable),
                    "probe": output,
                    "installed": now(),
                },
            )
            os.replace(staging, root)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
    progress(f"installed Apptainer {MANAGED_RUNTIME['version']} in {root}")
    return root / "bin" / "apptainer"


def runtime(context: RunContext, progress=print) -> Path:
    """The Apptainer executable to use: `execution.runtime`, or the managed one, installed now.

    Raises:
        TxturesError: The configured executable is missing, or the managed
            runtime cannot be installed.
    """
    if context.runtime is not None:
        if not (context.runtime.is_file() and os.access(context.runtime, os.X_OK)):
            raise TxturesError(f"execution.runtime {context.runtime} is not an executable file")
        return context.runtime
    if context.resources is None:
        raise TxturesError("the managed Apptainer needs a resources directory")
    return install_runtime(context.resources, progress)


def runtime_environment(executable: Path, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment for running `executable`.

    For the managed runtime, its `bin` directory is appended to `PATH`, so
    Apptainer finds the bundled `squashfuse_ll` and `fuse-overlayfs`;
    helpers already on `PATH` keep precedence.
    """
    env = dict(os.environ if base is None else base)
    if (executable.parent.parent / RUNTIME_MANIFEST).is_file():
        directory = str(executable.parent)
        parts = env.get("PATH", "").split(os.pathsep)
        if directory not in parts:
            env["PATH"] = os.pathsep.join([*filter(None, parts), directory])
    return env


# The image


@dataclass
class Tools:
    """A built tools image, ready to run programs in."""

    sandbox: Path
    executable: Path
    manifest: dict[str, Any]

    def command(self, argv: Sequence[str], binds: Sequence[Path] = ()) -> list[str]:
        """The command that runs `argv` in the image with `binds` mounted at the same paths.

        The host environment and home directory are not passed in.
        """
        command = [str(self.executable), "exec", "--cleanenv", "--no-home"]
        for path in dict.fromkeys(str(Path(b).resolve()) for b in binds):
            command += ["--bind", path]
        return [*command, str(self.sandbox), *argv]

    def run(
        self, argv: Sequence[str], binds: Sequence[Path] = (), **options
    ) -> subprocess.CompletedProcess:
        """Run `argv` in the image; `options` are passed to `subprocess.run`."""
        options.setdefault("env", runtime_environment(self.executable))
        return subprocess.run(self.command(argv, binds), **options)


def _scratch() -> Path:
    """A directory on local disk for building, preferring the job's own scratch."""
    for name in ("SLURM_TMPDIR", "TMPDIR"):
        value = os.environ.get(name)
        if value and Path(value).is_dir():
            return Path(value)
    return Path(tempfile.gettempdir())


class ToolsImage(Resource):
    """The tools image built from `tools.def`.

    The resolved spec holds `definition_sha256`. Building runs `apptainer
    build --sandbox` on local disk and copies the sandbox into the resource
    directory with `tools.json`, the tools and package versions it records.
    """

    kind = "tools_image"
    version = 1

    def __init__(self, spec: Mapping[str, Any] | None = None):
        super().__init__(spec or {})

    def resolve(self, context: RunContext) -> dict[str, Any]:
        return {"definition_sha256": definition_digest()}

    def stale(self, manifest: Mapping[str, Any]) -> str | None:
        """An image of another definition, built by an older txtures.

        An image built by this or a newer version may belong to a checkout
        under development, or to a newer installation, that shares the store.
        """
        from txtures import __version__
        from txtures.engine.collection import built_before

        if manifest["spec"].get("definition_sha256") == definition_digest():
            return None
        if not built_before(manifest.get("txtures_version"), __version__):
            return None
        return f"built from another tools definition by txtures {manifest['txtures_version']}"

    def workload(self, resolved: Mapping[str, Any]) -> Workload:
        return Workload(slurm=True, ncores=8, memory_gb=16, time_hours=2, ngpus=0)

    def build(self, resolved: Mapping[str, Any], directory: Path, context: RunContext) -> None:
        if definition_digest() != resolved["definition_sha256"]:
            raise TxturesError(
                f"{DEFINITION} changed after this build was planned; run the request again"
            )
        executable = runtime(context)
        with tempfile.TemporaryDirectory(dir=_scratch(), prefix="txtures-tools-") as work:
            work = Path(work)
            (work / "tmp").mkdir()
            shutil.copyfile(DEFINITION, work / "tools.def")
            env = runtime_environment(executable)
            env.update(APPTAINER_CACHEDIR=str(work / "cache"), APPTAINER_TMPDIR=str(work / "tmp"))
            command = [
                str(executable),
                "build",
                *BUILD_FLAGS,
                "--sandbox",
                str(work / SANDBOX),
                str(work / "tools.def"),
            ]
            with (directory / "build.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                tail = (directory / "build.log").read_text(errors="replace").splitlines()[-20:]
                raise TxturesError(
                    f"building the tools image failed ({result.returncode}):\n" + "\n".join(tail)
                )
            shutil.copytree(work / SANDBOX, directory / SANDBOX, symlinks=True)
        recorded = directory / SANDBOX / "opt" / "txtures" / IMAGE_MANIFEST
        tools = json.loads(recorded.read_text(encoding="utf-8"))
        write_json(
            directory / IMAGE_MANIFEST,
            {
                **tools,
                "definition_sha256": resolved["definition_sha256"],
                "apptainer": probe(executable),
                "built": now(),
                "host": platform.node(),
            },
        )

    def load(self, resolved: Mapping[str, Any], directory: Path, context: RunContext) -> Tools:
        return Tools(
            sandbox=directory / SANDBOX,
            executable=runtime(context),
            manifest=read_json(directory / IMAGE_MANIFEST),
        )
