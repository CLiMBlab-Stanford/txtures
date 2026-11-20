"""Tests of the Apptainer runtime, the tools image, and resource dependencies."""

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from txtures import tools
from txtures.engine.execution import build_resource
from txtures.engine.planning import from_record, plan_resource, resource_identity, resource_record
from txtures.errors import TxturesError
from txtures.identity import digest
from txtures.modules import Resource, RunContext, Workload
from txtures.store import EntryStore

FAKE_APPTAINER = """#!/bin/sh
# A stand-in for apptainer: `--version`, `build ... --sandbox DIR DEF`, and `exec`.
if [ "$1" = "--version" ]; then echo "apptainer version 1.4.5"; exit 0; fi
if [ "$1" = "build" ]; then
    arguments="$*"
    sandbox=""
    while [ $# -gt 0 ]; do
        if [ "$1" = "--sandbox" ]; then sandbox=$2; fi
        shift
    done
    mkdir -p "$sandbox/opt/txtures"
    echo '{"kenlm": {"commit": "abc"}, "packages": {"g++": "12"}}' > "$sandbox/opt/txtures/tools.json"
    echo "built with $arguments"
    exit 0
fi
if [ "$1" = "exec" ]; then shift; echo "exec $*"; exit 0; fi
exit 2
"""


def fake_bundle(tmp_path: Path) -> tuple[Path, str]:
    """A runtime archive holding the fake apptainer, and its SHA-256."""
    archive = tmp_path / "bundle.tar.gz"
    data = FAKE_APPTAINER.encode()
    with tarfile.open(archive, "w:gz") as stream:
        info = tarfile.TarInfo("bin/apptainer")
        info.size, info.mode = len(data), 0o755
        stream.addfile(info, io.BytesIO(data))
    return archive, hashlib.sha256(archive.read_bytes()).hexdigest()


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """Point the managed runtime at a fake bundle already in the artifacts directory."""
    archive, checksum = fake_bundle(tmp_path)
    spec = {**tools.MANAGED_RUNTIME, "sha256": checksum, "artifact": "bundle.tar.gz"}
    monkeypatch.setattr(tools, "MANAGED_RUNTIME", spec)
    resources = tmp_path / "resources"
    artifacts = tools.runtime_directory(resources).parent / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "bundle.tar.gz").write_bytes(archive.read_bytes())
    return resources


def test_install_runtime_checks_probes_and_records_it(bundle):
    executable = tools.install_runtime(bundle, progress=lambda message: None)
    assert executable == tools.runtime_directory(bundle) / "bin" / "apptainer"
    manifest = json.loads((executable.parent.parent / tools.RUNTIME_MANIFEST).read_text())
    assert manifest["probe"] == "apptainer version 1.4.5"
    assert tools.managed_runtime(bundle) == executable
    assert tools.install_runtime(bundle, progress=pytest.fail) == executable


def test_damaged_runtime_is_reinstalled(bundle):
    executable = tools.install_runtime(bundle, progress=lambda message: None)
    executable.write_text("#!/bin/sh\necho changed\n")
    assert tools.managed_runtime(bundle) is None
    assert tools.install_runtime(bundle, progress=lambda message: None) == executable
    assert tools.managed_runtime(bundle) == executable


def test_archive_with_the_wrong_checksum_is_downloaded_again(bundle, monkeypatch):
    artifact = tools.runtime_directory(bundle).parent / "artifacts" / "bundle.tar.gz"
    artifact.write_bytes(b"not the bundle")

    def refuse(url, target, checksum):
        raise TxturesError("offline")

    monkeypatch.setattr(tools, "_download", refuse)
    with pytest.raises(TxturesError, match="offline"):
        tools.install_runtime(bundle, progress=lambda message: None)


def test_configured_runtime_must_be_executable(tmp_path):
    with pytest.raises(TxturesError, match="not an executable file"):
        tools.runtime(RunContext(runtime=tmp_path / "missing"))
    with pytest.raises(TxturesError, match="HTTPS"):
        tools._download("http://example.org/x", tmp_path / "x", "0")


def test_managed_runtime_puts_its_helpers_on_path(bundle):
    executable = tools.install_runtime(bundle, progress=lambda message: None)
    env = tools.runtime_environment(executable, {"PATH": "/usr/bin"})
    assert env["PATH"] == f"/usr/bin:{executable.parent}"
    assert tools.runtime_environment(Path("/opt/apptainer/bin/apptainer"), {"PATH": "/x"}) == {
        "PATH": "/x"
    }


def test_image_identity_follows_the_definition(tmp_path, monkeypatch):
    definition = tmp_path / "tools.def"
    definition.write_text("Bootstrap: docker\nFrom: debian\n")
    monkeypatch.setattr(tools, "DEFINITION", definition)
    first = plan_resource("tools", tools.ToolsImage(), RunContext())
    expected = hashlib.sha256(b"Bootstrap: docker\nFrom: debian\n").hexdigest()
    assert first.resolved == {"definition_sha256": expected}
    definition.write_text("Bootstrap: docker\nFrom: ubuntu\n")
    second = plan_resource("tools", tools.ToolsImage(), RunContext())
    assert first.identity != second.identity


def test_image_builds_a_sandbox_and_records_its_tools(bundle, tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    context = RunContext(resources=bundle)
    image = plan_resource("tools", tools.ToolsImage(), context)
    store = EntryStore(tmp_path / "store")
    assert build_resource(store, image, context, lambda message: None)
    directory = store.get(image.kind, image.identity)
    manifest = json.loads((directory / tools.IMAGE_MANIFEST).read_text())
    assert manifest["kenlm"] == {"commit": "abc"}
    assert manifest["definition_sha256"] == tools.definition_digest()
    log = (directory / "build.log").read_text()
    assert "built with build --fakeroot --ignore-fakeroot-command --sandbox" in log
    loaded = image.resource.load(image.resolved, directory, context)
    command = loaded.command(["lmplz", "--help"], binds=[tmp_path])
    assert command[1:4] == ["exec", "--cleanenv", "--no-home"]
    assert command[4:6] == ["--bind", str(tmp_path.resolve())]
    assert command[-3:] == [str(directory / "sandbox"), "lmplz", "--help"]
    assert loaded.run(["true"], capture_output=True, text=True).stdout.startswith("exec ")


def test_image_refuses_a_definition_changed_after_planning(bundle, tmp_path, monkeypatch):
    image = plan_resource("tools", tools.ToolsImage(), RunContext(resources=bundle))
    definition = tmp_path / "tools.def"
    definition.write_text("changed\n")
    monkeypatch.setattr(tools, "DEFINITION", definition)
    with pytest.raises(TxturesError, match="changed after this build was planned"):
        image.resource.build(image.resolved, tmp_path, RunContext(resources=bundle))


def test_image_runs_on_slurm_without_gpus():
    workload = tools.ToolsImage().workload({})
    assert workload.slurm and workload.ngpus == 0


# Resource dependencies


BUILT: list[str] = []


class Base(Resource):
    kind = "test_base"
    version = 1

    def build(self, resolved, directory, context):
        BUILT.append("base")
        (directory / "value").write_text(str(resolved["n"]))

    def load(self, resolved, directory, context):
        return int((directory / "value").read_text())


class Derived(Resource):
    kind = "test_derived"
    version = 1

    def dependencies(self):
        return {"base": Base({"n": self.spec["n"]})}

    def resolve(self, context):
        # Only the dependency depends on `n`, so identities differ through it alone.
        return {}

    def workload(self, resolved):
        return Workload(slurm=True)

    def build(self, resolved, directory, context, dependencies):
        BUILT.append("derived")
        (directory / "value").write_text(str(dependencies["base"] * 10))

    def load(self, resolved, directory, context):
        return int((directory / "value").read_text())


def test_identity_without_dependencies_is_unchanged():
    resource = Base({"n": 1})
    assert resource_identity(resource, {"n": 1}) == digest(
        {"kind": "test_base", "version": 1, "spec": {"n": 1}}
    )
    assert resource_identity(resource, {"n": 1}, {}) == resource_identity(resource, {"n": 1})


def test_dependencies_enter_identity_and_are_built_first(tmp_path):
    BUILT.clear()
    one = plan_resource("derived", Derived({"n": 1}), RunContext())
    two = plan_resource("derived", Derived({"n": 2}), RunContext())
    assert one.resolved == two.resolved and one.identity != two.identity
    assert (
        one.dependencies["base"].identity
        == plan_resource("b", Base({"n": 1}), RunContext()).identity
    )
    store = EntryStore(tmp_path)
    assert build_resource(store, one, RunContext(), lambda message: None)
    assert BUILT == ["base", "derived"]
    assert (store.get(one.kind, one.identity) / "value").read_text() == "10"
    assert store.manifest(one.kind, one.identity)["dependencies"] == {
        "base": one.dependencies["base"].identity
    }


def test_records_rebuild_planned_resources_and_check_identity():
    planned = plan_resource("derived", Derived({"n": 3}), RunContext())
    record = json.loads(json.dumps(resource_record(planned)))
    rebuilt = from_record(record)
    assert rebuilt.identity == planned.identity
    assert rebuilt.dependencies["base"].identity == planned.dependencies["base"].identity
    record["dependencies"]["base"]["resolved"]["n"] = 4
    with pytest.raises(TxturesError, match="builds differently"):
        from_record(record)
