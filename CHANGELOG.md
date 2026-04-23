# Changelog

Notable changes to HUNTER. Format follows Keep a Changelog. Versions track the SHA-locked code state plus documentation and tooling around it.

## [Unreleased]

### Added
- `docs/HUNTER_ARCHITECTURE.md` walkthrough of the `hunter.py` engine file, including section-by-section navigation and rationale for the current layout.
- `pyproject.toml` with project metadata, dependencies, and tool configuration (pytest, ruff).
- `.editorconfig` for consistent indentation and line endings across editors.
- `.github/workflows/test.yml` for continuous testing on Python 3.10, 3.11, 3.12.
- Combined-324 narrative analysis. Earlier reported r = −0.49 (n = 61 main-pipeline) extended to combined corpus: r = −0.27 (n = 324). Sign holds; magnitude weaker on pooled sample. The mechanism-kill round is the specific channel through which clean stories are penalised.
- `LAUNCH_POST.md` first Substack post draft with three title and subtitle options.
- `publication_agent.py` (untracked) for generating Substack and X-ready cards from diamond-tier findings.

### Changed
- README tightened: cut a 170-word run-on sentence in "What this is", compressed the Key artifacts enumeration, removed the triple-parallel "Not X, not Y, not Z" structure, dropped em-dashes, restructured Reading order as guidance rather than a directive list.
- LAUNCH_POST scrubbed of em-dashes and rewritten with shorter sentence cadence for cold-reader entry.
- DB table count corrected across docs (was 43/226; actual count is 52).
- Diamond theses count corrected (was "Fifteen" in catalogue header; actual is 18 ≥ 75).
- Archive references rewritten so the gitignored local archive is not assumed accessible.
- EMPIRICAL_FINDINGS.md gained a §0 combined-324 picture; sections 1–9 preserved as historical n=61 analysis.
- MATH_VERIFICATION.md test 6 references updated to point at the public `V3_GOLDEN_*` config rather than a gitignored archive folder.

### Removed
- `HUNTER_STORY.pdf` and `docs/HUNTER_STORY.md`. Narrative pitch material withdrawn from public release.

### Fixed
- Pre-registration manifest archive references now resolve. Earlier docs pointed at gitignored paths.

## [1.0.0] — 2026-04-19

### Locked
- Pre-registration manifest at SHA-256 `f39d2f5ff6b3e695`.
- Core engine files locked: `hunter.py`, `prompts.py`, `config.py`, `theory.py`, `thesis_dedup.py`, `portfolio_feedback.py`, `cycle_detector.py`.
- Corpus frozen at 2026-03-31.
- Three null baselines committed (random-pair, within-silo, shuffled-label).
- Decision rules committed (primary wins, primary loses, must-not-do).

### Released
- HUNTER Cross-Silo Financial Corpus v1 to Zenodo under CC-BY-4.0 (DOI 10.5281/zenodo.19667567).
- Public prediction board live and empty by design at https://johnmalpass.github.io/hunter-research/.
- 12-week pre-registered out-of-sample study scheduled for June 1 through August 31, 2026.

## v2 (planned, post summer)

After the summer 2026 study reports in September, the SHA lock is released and the codebase is refactored:

- Split `hunter.py` into a `collision/` package with one module per matching strategy and one per kill round.
- Split `database.py` into a proper `db/` package.
- Add type annotations and a static type checker pass.
- Re-lock the manifest with new code hash for v2 evaluation period.

The framework, decision rules, and pre-registration discipline carry over. The locked v1 results stand on their own and inform v2 only through documented decisions in this CHANGELOG.
