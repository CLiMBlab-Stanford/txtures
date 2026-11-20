# Tools

Some features need programs that are not Python packages, such as KenLM's
`lmplz`, which trains n-gram models. txtures runs every such program from
one [Apptainer](https://apptainer.org) image, the tools image, so no
installation compiles them and every host runs the same builds.

## The image

The image is built from `tools.def`, which is part of txtures
(`src/txtures/tools/tools.def` in a checkout). The definition pins its base
image by digest and each tool by release or commit. The image is a resource
like a language model: it is stored in the `resources` directory, shared by
everyone who uses that directory, and identified by the SHA-256 of the
definition. Resources built with the image, such as an n-gram model, depend
on it, so a new definition gives them new identities.

The image is an Apptainer sandbox, a directory, at
`RESOURCES/store/tools_image/.../sandbox`. Next to it, `tools.json` lists
the tools and the version of every installed package, and `build.log`
holds the build's output. txtures never opens the sandbox writable.

```console
$ txtures tools build
job 18586713 builds the tools image; logs: .../outputs/.txtures-jobs/.resources/tools_image-59d23ab8b37b/logs
$ txtures tools status
runtime: Apptainer 1.4.5 in .../resources/apptainer/apptainer-1.4.5-x86_64-6659c298aacfaf5a (installed)
definition: .../src/txtures/tools/tools.def (sha256 0b9b796b53b8)
image 59d23ab8b37b: .../resources/store/tools_image/59/59d23ab8b37b2a92.../sandbox
  built 2026-11-19T16:18:36+00:00 on john1.stanford.edu with apptainer version 1.4.5
  kenlm 4cb443e60b7b, 189 packages
```

`txtures tools build` submits a Slurm job where Slurm is set up (8 cores,
16 GB, 2 hours, no GPU; see [Job sizes](slurm.md#job-sizes)), and builds in
the calling process otherwise or with `--local`. A request that needs the
image builds it the same way when it is missing, so the command is needed
only to build it ahead of time.

`txtures tools exec -- COMMAND` runs a command in the image with the
working directory mounted, for example `txtures tools exec -- lmplz
--help`. `--bind PATH` mounts more directories. The command sees neither
the host's environment nor the home directory.

## Changing the image

Edit `tools.def` in a development checkout and run `txtures tools build`.
Because the definition's digest identifies the image, the edit builds a new
image, and the old one stays for the resources built with it. Record each
tool you add in `/opt/txtures/tools.json`, as the definition does for
KenLM.

Images are built without root privileges, through a user namespace that
maps the user to root. Apptainer's `--ignore-fakeroot-command` keeps the
host's `fakeroot` library, which needs the host's C library, out of the
build, so the base image need not match the host. `apt-get` in the
definition must run as root
(`APT::Sandbox::User "root"`), because the build cannot switch to the
`_apt` user.

## Apptainer

txtures uses its own Apptainer: a relocatable conda-forge build of
Apptainer 1.4.5, pinned by checksum. It is installed on first use in
`RESOURCES/apptainer/`, where every host that shares the `resources`
directory finds it, and it includes the FUSE helpers that let Apptainer
mount images without unpacking them. It needs Linux x86_64 with glibc 2.17
or newer.

To use another Apptainer, such as one an administrator installed, set
`execution.runtime` to its executable:

```bash
txtures site set execution.runtime=/usr/bin/apptainer
```

txtures never uses an Apptainer it finds on `PATH` unless this setting
names it.
