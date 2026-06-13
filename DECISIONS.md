# DECISIONS

Log of every deviation from SPEC.md, each with one line of reasoning. Never deviate silently.

- **2026-06-13 — Project is named `hearsay`, not `earshot`.** The PyPI check (`curl https://pypi.org/pypi/earshot/json`) returned HTTP 200: `earshot` is taken by an active package (Python bindings for a Rust voice-activity-detection library, v0.2.0) in our exact domain, so even name-squatting nearby would be confusing. Five free alternatives were verified against PyPI (all returned 404): `hearsay`, `soundbite`, `transcrawl`, `audioscribe`, `mediamark` (`overhear`, `earworm`, `echolog` were taken). Picked **hearsay**: a real, memorable word that describes the product literally — speech someone heard, passed along in written form — and reads well as a command (`hearsay <url>`). Used consistently everywhere; only SPEC.md keeps the original `earshot` wording because it is verbatim.
- **2026-06-13 — `git init` + `.gitignore` moved from Phase 0's last task to STEP ZERO.** The "one commit per completed task" rule requires a repo (and a working .gitignore) to exist from the first task onward.
- **2026-06-13 — STEP ZERO's three files share one bootstrap commit (58127c1).** They are the documents that establish the one-commit-per-task process itself, so a single bootstrap commit predating the process is the honest representation.
