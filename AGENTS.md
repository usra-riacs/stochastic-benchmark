# Agent Notes

This file captures repository-specific workflow notes for future coding agents.

## GitHub CLI

- Use `gh` for GitHub operations in this repository when requested.
- Avoid `gh issue view <issue> --comments`; some GitHub CLI versions request deprecated GraphQL fields. Prefer:
  ```bash
  gh issue view <issue> --json number,title,body,state,author,labels,assignees,comments,url
  ```
- If `gh pr view --web=false` fails on deprecated `projectCards` GraphQL fields, inspect explicit fields instead:
  ```bash
  gh pr view <pr> --json number,title,url,state,headRefName,baseRefName,author,commits,body
  ```
- GitHub does not allow a PR author to approve their own PR. If reviewing a PR authored by the current authenticated account, submit findings as `COMMENT` unless another review state is allowed.

## Tests And Coverage

- Documented test entry points are in `README.md`, `TESTING.md`, `run_tests.py`, and `.github/workflows/ci.yml`.
- Useful local checks:
  ```bash
  python run_tests.py unit
  python run_tests.py coverage
  flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics
  flake8 src --count --exit-zero --max-complexity=10 --max-line-length=120 --statistics
  ```
- The second flake8 command is advisory because it uses `--exit-zero`; it may print existing style warnings while still exiting successfully.
- The docs mention an 80% project-wide coverage target, but the current repository baseline is below that and CI does not enforce `coverage fail_under`. Treat passing patch coverage and passing `coverage-summary` as satisfying current enforced PR checks, while tracking the project-wide gap separately.
- The current project-wide coverage gap is tracked in https://github.com/usra-riacs/stochastic-benchmark/issues/69.
