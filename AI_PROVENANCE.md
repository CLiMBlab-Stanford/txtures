# AI provenance

## Declaration

This repository is developed with AI coding assistance. AI-generated output is
subject to human selection, review, testing, and approval. The human
contributors remain responsible for the code and documentation they publish.

## Involvement vocabulary

- **Generated:** AI output was accepted substantially as written.
- **Assisted:** AI suggestions were materially revised by a human.
- **Reviewed:** AI evaluated or commented on existing work without being its
  principal generator.

## Systems used

| Provider | Tool or agent | Model identifier | Period | Roles | Identification basis |
| --- | --- | --- | --- | --- | --- |
| Anthropic | Claude Code | claude-opus-5-5 | 2026-11 | design, implementation, testing, documentation | The active Claude Code environment exposes the model ID claude-opus-5-5. |

## Agent registration

An AI agent making a material contribution must read this manifest before
changing the repository. If its provider, tool, and exposed model identifier
are not already represented above, it must add a row. If they are represented,
it must extend the period or roles when necessary. Agents must use only identity
information explicitly exposed by their environment and must not record
prompts, private context, conversation logs, or session identifiers.

Registration describes the system involved; it does not make that system a Git
author or transfer human responsibility. The human contributor should add the
applicable `Assisted-by:` trailer when committing the work. Commits and pull
requests made with agent assistance must also follow the identity and
authentication policy in [CONTRIBUTING.md](CONTRIBUTING.md).

## Historical coverage

`notebooks/surp_v_dist.py` predates this repository. It is a research notebook
kept as reference material and is not part of the package. Its provenance is
not recorded here.

Historical commits that do not contain an `Assisted-by:` trailer have not been
retroactively attributed. Absence of a trailer therefore does not prove that AI
assistance was absent.

## Human accountability

- A human contributor must understand and approve every submitted change.
- Normal review, testing, security, licensing, and contribution requirements
  apply regardless of how a change was produced.
- AI systems are not Git authors, copyright holders, or DCO signatories.
- `Signed-off-by:` is reserved for a human making the applicable certification.
- An agent must not invent or silently substitute a human, bot, or service
  identity when committing or publishing work.

## Recording limitations

Record the exact provider model identifier when the tool exposes it. Otherwise,
record the user-visible model or product label and state `underlying model not
disclosed` in the identification-basis column. Do not infer or invent a model
version.
