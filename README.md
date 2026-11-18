# textures

textures computes word- and region-level features of text for psycholinguistic
research. The first feature under development is surprisal from causal
language models, with word-boundary correction and user-defined regions.

textures is in early development and does not yet compute any features. Its
interfaces will change during the 0.x series.

## Development

```bash
uv sync --group dev
uv run pytest -q
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution, attribution, and
release rules, and [WRITING_POLICY.md](WRITING_POLICY.md) for documentation
style.

## License

MIT. See [LICENSE](LICENSE).
