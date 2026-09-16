# Rule: Mandatory Approval Before Writing Code (Exception: Unit Tests)

## Core Invariant
**Never write or modify code without prior user approval.**

1. **Mandatory Approval Requirement**: Before writing or modifying any non-test code (features, data loaders, algorithms, experiments, feature extractors, rankers), always present the proposed implementation details / sub-steps to the user and **wait for explicit user approval**.
2. **Exception for Test Cases**: Writing, updating, or running unit test cases (`tests/test_*.py`) to verify behavior or assert ground truths is exempt from requiring prior approval.
3. **Show & Break Down First**: Present a clear, granular breakdown of what is being built, how it is divided into sub-steps, and the exact files/interfaces to be modified or created.
4. **Incremental Execution**: Once approved, execute changes in small, trackable units.
