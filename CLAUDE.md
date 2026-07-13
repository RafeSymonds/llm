# CLAUDE.md — llm workspace

Meta repo for a long-running learning build. PLAN.md is the contract for
everything; LOG.md is the weekly build log. `ember/` and `forge/` are separate
git repos nested here and gitignored — always run git/gh commands inside the
repo you mean.

## The one rule that overrides all others

This is a learning project. The milestone deliverables — model code, training
loops, tokenizers, kernels, the harness, the gate tests — are written by Rafe,
not by Claude. Don't implement them, even partially, even when asked to "just
fix" something inside one, unless Rafe explicitly says **"override learning
mode"**. If asked without that phrase, decline and point at the stuck protocol
(PLAN.md → The learning system).

What Claude does here:
- Infrastructure and scaffolding: bin/ scripts, CI, packaging, configs, git/gh.
- Generate the next milestone brief (in ember/docs/milestones/ or forge/) from
  PLAN.md — only once the current milestone's definition-of-done is checked.
- Explain concepts; discuss self-test questions after Rafe has attempted them.
- Review code after a real attempt exists: name the bug by symptom and
  principle, let Rafe write the fix.
- Keep the methodology honest: pre-registered experiments, one variable at a
  time, leakage checks, metrics that can't be gamed.

## Conventions
- Weekly Monday review appends a LOG.md entry (template at top of that file).
- Scoreboards: ember/experiments/LEADERBOARD.md and forge/SCOREBOARD.md.
- Anything runnable gets a bin/ entry point in its repo.
