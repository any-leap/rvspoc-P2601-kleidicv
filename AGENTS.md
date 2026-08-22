<!--
SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

SPDX-License-Identifier: Apache-2.0
-->

# Agents Guide

## Repository Overview

This repository contains KleidiCV, an Arm image-processing library focused on
high-performance AArch64 implementations. The top-level build is CMake-based
and includes the main library, threading support, C examples, tests, and
benchmarks.

## Key Paths

- `CMakeLists.txt`: top-level build entry point
- `kleidicv/`: main library sources
- `kleidicv_thread/`: threading support
- `examples/`: example programs
- `test/`: API and unit tests
- `benchmark/`: benchmark targets
- `doc/`: build, test, integration, and platform documentation
- `.devcontainer/devcontainer.json`: development container configuration
- `docker/Dockerfile`: container image definition

## Workspace Tasks

Do not hardcode build or test commands in agent instructions. Use the tasks
defined in `.vscode/tasks.json` as the source of truth for common validation
flows.

Once a source file is edited run the `Format source` VS Code task.

The default validation for changes is to run the `All API tests` VS Code task but 
to keep test cycle short the `GTEST_FILTER` environment variable should be set
accordingly which tests to run.

## Development Environment

The devcontainer builds from `.devcontainer/Dockerfile`, enables `ccache`, and keeps
the workspace mounted at its real path instead of `/workspace`. The Docker
image installs the cross-compilation and LLVM tooling used by the project.
Agents should expect to be running inside a VS Code dev container unless a
task explicitly says otherwise.

## Working Notes For Agents

- Prefer small, targeted changes and avoid reformatting unrelated files.
- Check for existing uncommitted changes before editing shared setup files.
- Use the documentation in `doc/` for background, but prefer
  `.vscode/tasks.json` for build, test, formatting, and CI entry points.
- Before choosing validation steps, check whether an appropriate VS Code task
  already exists and use it when possible.
- If a task touches container or toolchain setup, review both
  `.devcontainer/devcontainer.json` and `docker/Dockerfile` together.


<!-- GLOBAL_AGENT_BASELINE_2026_06_30 -->

## Global Agent Baseline

- Follow the global rules in `/Users/t3st/.codex/AGENTS.md`; nearest project/subdirectory instructions still override for project-specific details, except global safety rules.
- New Web App / TanStack Start / full-stack React / shadcn/ui / Bun work should use the `$start-webapp` Golden Path before scaffolding. UI/UX work should use `frontend-design-ui-ux` and treat `.ulpi/design/` as the design source of truth.
- UI/Web changes require browser acceptance, not just tests/build: inspect the running app for aesthetics, usefulness, responsive viewports, loading/empty/error states, overlap/overflow, and clickable controls.
- For repeatable multi-step work, define the loop before executing: Goal, Context, Actions, Feedback signal, Stop condition, Safety rails, and Handoff. Iterate until verification passes, a blocker is explicit, or a safety boundary is reached.
- Loops should not stay in one context by default: use subagents/parallel agents for independent research, implementation, review, or failure investigation; split long-running or context-heavy loops into fresh threads with explicit handoff notes.
- Before calling work complete, report the standard verification gate as applicable: Type Check, Unit Test, Build, Invariants, and Browser Test. If a gate cannot run, state why and what substitute evidence was used.
- Treat test, lint, typecheck, hook, and CI failures as real signals. Investigate root cause before classifying a failure as pre-existing; do not bypass hooks or delete/suppress rules to land work.
- Prefer root-cause fixes over suppressions: avoid `biome-ignore`, `eslint-disable`, `ts-ignore`, broad excludes, silent catches, and rule downgrades. If a mainline tradeoff requires temporary deferral, record reason, impact, owner, removal trigger, and follow-up task.
- Do not treat placeholder implementations as complete: no fake data paths, TODO stubs, empty handlers, silent fallbacks, temporary hardcode, or unconnected UI/API without explicit owner, reason, risk, and removal trigger.
- Prefer mature, stable, well-maintained packages and framework features over custom implementations for solved problems. For Web UI, prefer shadcn/ui standard components, registry items, and blocks before hand-rolling generic primitives or large layout components.
- New projects must be git repositories: after scaffolding, check for `.git`; run `git init` if missing, and verify `.gitignore` excludes dependencies, build output, env files, local databases, and tool caches.
- Keep project instructions concise and operational. Prefer exact commands, `Always / Ask First / Never` boundaries, known pitfalls, and links to project docs over long duplicated process text.
