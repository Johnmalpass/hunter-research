# Contributing

HUNTER is in a pre-registered study window from now through August 31, 2026. Most code-level contributions are not appropriate during that window because the SHA-256 hash on the seven core engine files (`hunter.py`, `prompts.py`, `config.py`, `theory.py`, `thesis_dedup.py`, `portfolio_feedback.py`, `cycle_detector.py`) cannot change without breaking the pre-registration. After the summer study reports in September 2026 the lock will be released and a v2 development cycle opens.

## What's welcome right now

- **Prior-art pointers.** If you've seen the compositional alpha thesis in published literature, send the citation. Especially anything before 2024 that names the cross-silo residual or argues for it formally. Open an issue or email.
- **Replication attempts.** Pull the frozen Zenodo corpus, clone this repo, run `python run.py preregister check` against the locked manifest. Report mismatches.
- **Bug reports on non-locked modules.** Anything outside the seven hashed files is fair game. `database.py`, the analyser modules, dashboards, scripts, tests. Open an issue.
- **Documentation improvements.** Typos, broken links, unclear sections, factual corrections. PRs welcome.
- **Independent analyses on the Zenodo corpus.** The corpus is CC-BY-4.0. Run your own analyses, publish your own results, cite the corpus.

## What's not welcome

- Cold pitches to invest, license, or commercialise. This is a research project at the pre-empirical-test stage.
- Mid-study changes to scoring weights, thresholds, kill rounds, or any pre-registered parameter.
- Asking me to add features to the live pipeline before September. The point of pre-registration is that the test is run against the system as locked.

## Setup

```bash
git clone https://github.com/Johnmalpass/hunter-research.git
cd hunter-research
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # add your Anthropic API key
python test_core.py
```

If `python test_core.py` reports `0 crashed`, the install is good.

## Style

Code style is enforced loosely by `ruff` (config in `pyproject.toml`). Line length 100. Import order standard. New code should have type hints; legacy code in `hunter.py` will get type hints in v2.

For docs, the house style is short sentences, no em-dashes, no triple-parallel structures, plain prose. See `docs/MATH_VERIFICATION.md` as the reference voice.

## Reporting a bug

Open a GitHub issue. Include:
- The command that triggered the bug.
- The full traceback or error message.
- The Python version (`python --version`).
- Whether you were running against the live `hunter.db` or the frozen Zenodo corpus.

## Reporting a finding

If you've run an independent analysis on the Zenodo corpus and found something interesting, open an issue with the analysis attached. Especially welcome: anything that contradicts a claim made in the docs.

## Contact

Honest critique, prior-art pointers, and replication attempts welcome. No cold pitches.

John Malpass · University College Dublin · johnjosephmalpass@gmail.com
