# Code Quality & Linter Guidelines for RetrievLab

## 1. Zero Lint Invariant
Every Python file committed to the repository must pass `uv run ruff check .` with **zero errors and zero warnings**.

---

## 2. String Formatting & F-Strings Rule (Ruff F541)
* **Never prefix a string with `f` if it does not contain `{}` placeholders.**
  * ❌ `print(f"\nQuery-Level Win/Loss Comparison:", flush=True)`
  * ✅ `print("\nQuery-Level Win/Loss Comparison:", flush=True)`
* For static titles, table headers, diagnostic section banners, report templates, and fixed logs, always use standard quotes `""` or `''`.
* Only use f-strings (`f"..."`) when interpolating dynamic variables or expressions (e.g. `f"Recall@5: {recall:.4f}"`).

---

## 3. Unused Code & Clean Scopes (Ruff F401 & F841)
* **No Unused Imports (F401)**: Never leave unused standard library or third-party imports (e.g. `sys`, `os`, `time`, `json`).
* **No Dead Variable Assignments (F841)**: Never assign values to variables (e.g. `t_start`, `offset`, `temp`) that are not subsequently read, passed to functions, or returned.

---

## 4. Mandatory Verification Workflow
Before committing any code or concluding any ticket:
1. Run `uv run ruff check .`
2. If trivial warnings exist, run `uv run ruff check --fix .`
3. Verify that `uv run ruff check .` prints `All checks passed!`
