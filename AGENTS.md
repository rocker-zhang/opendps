# AGENTS.md — Development discipline for this repository

This project is an open-source reimplementation of NVIDIA's datacenter GPU
power-management stack (telemetry → profiling → dynamic power management),
targeting a **demo-grade working deliverable**. This file is the single source
of engineering discipline for both human contributors and AI agents working in
this repo. It is intentionally public-safe: **no internal infrastructure,
hostnames, IPs, credentials, or employer identifiers may ever appear here or in
any committed file.**

## 1. Contribution flow — three-layer review + push-gate

Every change passes three independent reviews before it can be pushed:

1. **Code review** (every change) — a reviewer agent reads the code layer:
   correctness, style, cleanup, resource handling, API misuse.
2. **Arc review** (every change) — a reviewer agent reads the whole arc:
   root-cause → fix → test → validation logic. Does the change actually solve
   the stated problem, and is the validation sound?
3. **Prose / tone review** (anything published: README, docs, issues, PRs) —
   a reviewer rewrites against the checklist in §6.
4. **Push-gate** — passing all three reviews does **not** authorize a push.
   Before any `push` / `force-push` / PR / issue / comment, tell the maintainer
   and wait for an explicit OK. Never push on your own initiative.
5. **Lab dual-review** — every lab/experiment result must be checked by **two**
   reviewers to confirm the run was valid and the result is not a false
   positive before the milestone counts as met.

## 2. Per-component isolation (anti-context-pollution)

This is a multi-component system. To keep one component's context from polluting
another:

- Each component / layer is developed by its **own dedicated subagent** with a
  scoped prompt. The orchestrating session does not hold every component's full
  detail at once.
- Large agent outputs (research reports, test logs) return **conclusions +
  key `file:line` references only**. Full detail goes to a temp file, never
  flushed back into the main context.
- One-shot large logs (e.g. a passing test suite) are not re-fed; grep the
  verdict line + key markers instead.

## 3. Privacy and attribution rules — mandatory

Every public-facing file (README, docs, code comments, commit messages, release notes)
must pass this checklist **before** commit:

**Hardware naming:**
- Use only the public NVIDIA model name: "B300 SXM6", "GB200", "A100" — no
  SKU suffixes (AC, NVL, etc.) that narrow down the supplier or organization.
- Do not write exact per-node or cluster GPU counts that reveal fleet scale.

**Organization / career:**
- Zero employer names, team names, cluster names, datacenter codenames, or
  internal project names in any committed file.
- Phrases like "our cluster", "in production at", "deployed at \[org\]",
  "tested on N nodes" imply organizational context — remove or generalize.
- Lab hardware references: use the public GPU model name only, never internal
  hostnames or lab-specific identifiers.

**Commit messages:**
- No AI-tool attribution trailers of any kind.
- Keep messages concise and technical — no workflow or toolchain commentary.

**CI / tooling:**
- Do not commit files whose primary content is a list of internal identifiers
  to protect against (e.g. a security scanner config with employer names as
  patterns). Such files are themselves sensitive and belong in private config.

---

## 4. Lab discipline — strict isolation

All lab work must be isolated:

- **Python work runs in a project-local `venv`**, never the global interpreter.
  Do not `pip install` into system Python. Pin deps in a `requirements.txt` /
  `pyproject.toml` per component.
- Native/CUDA builds go to a per-component build dir; do not install artifacts
  system-wide.
- **Known hardware constraint:** GB10 exposes power/clock/temp **telemetry**
  but does **not** support power-capping (`nvidia-smi -pl` → N/A) or clock-lock
  without privileges. The *monitor/profile* half of the system is demoable on
  GB10; the *control/enforce* half must be validated via software-throttling
  emulation or on a GPU that supports power limits.
- A milestone that needs a lab is **not done** until the lab run passes and two
  reviewers have signed off (see §1.5).

## 4. Research / landing discipline

- **Check prior art first** — before building, search for existing open
  implementations (DCGM, nvml bindings, gpu power tools) so we reuse, not
  reinvent. Cite what we deliberately reimplement and why.
- **Verify "suspected bug" against HEAD** — line numbers drift; read the whole
  function statically, don't grep the first few lines.
- **No premature "it's a bug" verdicts** — reproduce or state the evidence
  first. An investigation that ends in "false positive" is still a completed
  task; record the conclusion honestly.

## 5. Milestone definition-of-done

A milestone is complete only when:
1. Code merged to the working branch with the three-layer review passed.
2. Tests written and green (unit + the relevant lab run).
3. Lab run (if any) reproduced and dual-reviewed (§1.5).
4. Docs/README updated for the new capability.
5. Status updated in the project tracker.
6. Maintainer has given explicit OK to push (§1.4).

## 6. Prose / commit / tone rules

- **Language:** code, commits, PRs, issues, and public docs in **English**.
- **Commit sign-off:** author = the maintainer's own name + email. Do **not**
  add `Co-Authored-By: Claude`. The string "AI" must not appear in any commit
  message, PR body, or issue.
- **Tone checklist** (reject/rewrite if present): bold pseudo-headers
  (`**Possible X**`), empty `(A)/(B)/(C)` enumerations, "Happy to…" /
  "Please let me know…" sign-offs, excessive em-dashes, formal-corporate filler
  (Moreover, Additionally, In any case, That said, It is worth noting),
  robotic transitions (First… Second… Finally…).

## 7. Task tracking

- Multi-step work uses the task tracker; mark `in_progress` at start,
  `completed` when done (including false-positive investigations).
- Don't leave dead tasks `pending` — they mislead future readers.
