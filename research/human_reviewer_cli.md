# Stage B: human reviewer CLI and persistence

Stage B adds `review_store.py` and `scripts/review_tactical_labels.py` around the
unchanged Stage A workflow. There are no new adjudication states, taxonomy changes,
approvals, searches, training exports, or model training.

## Architecture and lifecycle

The CLI reuses the dataset-readiness inspector to validate the complete candidate,
feature, evidence and source bundle, including available or explicitly supplied
manifests. Parsed bytes must match the inspector's hashes. Every existing review
is revalidated against the current bundle with Stage A `validate_review`.

`review` requires an explicit tactical concept, decision, reviewer ID and rationale.
It calls `prepare_review` and then `record_review`, saving only the resulting draft.
`finalize` selects the exact submitted record by its canonical review hash and calls
`finalize_review`. The original timestamp, decision and evidence bindings survive.
Multiple distinct drafts are permitted. Exact duplicates and finalized conflicts
follow Stage A. To prevent accidental command retries, the CLI additionally rejects
submissions identical in all fields except timestamp/state. Changing the human
decision or rationale remains a distinct draft; this does not resolve final conflicts.

All taxonomy tactical concepts are accepted for explicit human decisions, as Stage A
permits. Lists display only supported automatic *proposals*, not hypothetical detector
coverage. Missing evidence is shown in lists and prevents review/show/finalization.
Abstentions and historical evidence can be reviewed under Stage A; no positive or
negative decision is inferred from either. Historical v1 remains training-ineligible.
Mate-to-CP alone cannot establish a missed forced mate. Reviewers must record
`unresolved` when they cannot establish a concept; the CLI cannot authenticate their
identity or independently establish chess correctness.

Finalization is not independent approval. The CLI has no approval input or command.
Finalized unresolved decisions remain ineligible. Finalized positive/negative
decisions still require independently established approval and existing eligibility
checks. No historical proposal or real candidate was human-labeled by this task.

## Commands

Run from the repository root. Direct `python scripts/review_tactical_labels.py ...`
execution also works. The parent directory of the store must already exist.

```powershell
# Explicit creation; refuses to overwrite any existing store, including corrupt data.
.venv\Scripts\python.exe -B -m scripts.review_tactical_labels init --store data/reports/reviews.jsonl

# Read-only; --store is optional, but a supplied missing/corrupt store is an error.
.venv\Scripts\python.exe -B -m scripts.review_tactical_labels list --store data/reports/reviews.jsonl
.venv\Scripts\python.exe -B -m scripts.review_tactical_labels show --store data/reports/reviews.jsonl --candidate-id <ID>

# Only after actual human examination; placeholders are not real review instructions.
.venv\Scripts\python.exe -B -m scripts.review_tactical_labels review --store data/reports/reviews.jsonl --candidate-id <ID> --concept hanging_piece --decision unresolved --reviewer-id <ACTUAL_REVIEWER_ID> --rationale "Insufficient defensive evidence."

# Explicit finalization of the exact draft hash printed by review/list/show.
.venv\Scripts\python.exe -B -m scripts.review_tactical_labels finalize --store data/reports/reviews.jsonl --candidate-id <ID> --review-hash <DRAFT_SHA256>

# Existing inspector can read this store directly. This supplies no approvals.
.venv\Scripts\python.exe -B -m scripts.inspect_training_readiness --adjudications-path data/reports/reviews.jsonl
```

All artifact-reading commands accept `--candidate-path`, `--feature-path`,
`--evidence-path`, `--source-dataset-path`, `--source-manifest-path`, and optional
`--candidate-manifest-path`, `--feature-manifest-path`, `--labeling-manifest-path`.
Defaults match the readiness CLI's saved pilot. Candidate IDs and review hashes are
different identities; finalization needs both. List/show expose review states and
decisions separately, including `draft:unresolved` and `finalized:unresolved`.

Exit 0 means success, 2 means invalid input/storage conflict/I/O failure (including
argparse errors), and 1 means an unexpected failure. Success never means approval.

## Storage contract and concurrency

The store is UTF-8 JSONL: one complete existing `AdjudicationRecord` per line,
sorted object keys, compact separators, literal Unicode and a final LF per record.
Collection order is preserved. An explicitly initialized empty store is zero bytes.
Missing stores fail; they never silently become empty. Unknown/missing fields,
duplicate JSON keys, blank records, invalid versions/states and invalid collections
are rejected. Limits match the pilot: 32 MiB and 10,000 records.

`load_reviews(path)` returns an immutable tuple. `load_snapshot(path)` additionally
returns the SHA-256 of the exact bytes read. `save_reviews(path, reviews,
expected_digest=...)` requires that digest; `None` only initializes an absent empty
store. No parent directories are created implicitly.

Writes validate/serialize before touching the destination, acquire a sibling
`<store>.lock` with exclusive creation, validate the existing store and compare its
digest **while holding the lock**, write a same-directory temporary file, flush and
fsync it, and atomically replace with `os.replace`. Ordinary failures remove temporary
files and release owned locks. Validation/serialization/fsync/replace failure preserves
the previous store. Readers see an old or new complete store on supported local
filesystems. A successful save returns the committed snapshot.

The digest detects stale readers; the exclusive lock prevents simultaneous cooperating
writers from both passing the digest check. A digest alone would not do this. All
writers must use this API and the same canonical destination. Path resolution unifies
ordinary relative/symlink aliases; hard-link aliases, external editors and unreliable
network filesystems are outside this protocol. Do not modify the store externally
while reviews are being written.

Process death may leave a lock and temporary file. Locks are never automatically
stolen. Stop all writers, inspect the destination and leftover files, and only then
manually remove a confirmed stale lock. A crash after replacement but before output
may mean the operation committed: inspect the store before retrying. File fsync does
not promise directory-entry durability across power loss on every platform.

Artifact files must remain immutable during CLI operations. The CLI revalidates the
entire bundle immediately before saving and rejects observed changes; artifact files
are not part of the store lock, so this is not a transactional lock against external
artifact writers. Stage A's existing limitation also remains: source bytes and links
are checked, but the source PGN is not replayed to re-prove positions. Review hashes
protect integrity, not reviewer expertise or identity.

## Verification

```powershell
.venv\Scripts\python.exe -B -m pytest tests/ml/test_review_workflow.py tests/ml/test_review_store.py tests/ml/test_review_cli.py tests/ml/test_dataset_readiness.py -q
.venv\Scripts\python.exe -B -m pytest -q
```

New fixtures are temporary and synthetic. They do not write production datasets or
approve any real reviews. Test results are reported separately after execution.

## Stage C2: separate trusted-operator CLI

The lifecycle continues from a finalized human decision to independent authorization
of its exact canonical hash, then existing eligibility validation. The reviewer CLI
still cannot approve. Use `scripts.approve_tactical_review` as a separately trusted
operator, with registry directory access denied to annotation submitters. The tool
does not configure permissions or authenticate identities. Protect the directory as
well as the registry, since atomic replacement creates a new temporary file there.

From the repository root (placeholder values below require actual human decisions):

```powershell
.venv\Scripts\python.exe -B -m scripts.approve_tactical_review init --approvals-path <PROTECTED_DIRECTORY>/approvals.json
.venv\Scripts\python.exe -B -m scripts.approve_tactical_review list --approvals-path <PROTECTED_DIRECTORY>/approvals.json
.venv\Scripts\python.exe -B -m scripts.approve_tactical_review inspect --adjudications-path <REVIEWS.jsonl> --review-hash <FINALIZED_REVIEW_SHA256>
.venv\Scripts\python.exe -B -m scripts.approve_tactical_review approve --adjudications-path <REVIEWS.jsonl> --approvals-path <PROTECTED_DIRECTORY>/approvals.json --review-hash <FINALIZED_REVIEW_SHA256> --confirm-independent-approval --operator-id <INDEPENDENT_OPERATOR_ID> --independent-justification "<Independent basis for this exact positive or negative decision>"
.venv\Scripts\python.exe -B -m scripts.inspect_training_readiness --adjudications-path <REVIEWS.jsonl> --approvals-path <PROTECTED_DIRECTORY>/approvals.json
```

Direct `python scripts/approve_tactical_review.py ...` execution also works.
`inspect` and `approve` accept the same candidate/feature/evidence/source/manifest
path options as the reviewer CLI. Supply a supported v2 bundle explicitly: the default
saved pilot is historical v1 and cannot be approved. `inspect` is read-only and can
show a blocked draft or unresolved record for diagnosis; it never promises eligibility.
`list` lists registry hashes only, not verified eligible labels. Use the readiness
inspector with the complete matching input bundle to calculate actual label counts.

`approve` requires affirmative action and an independent justification; no decision
is defaulted, inferred or finalized. Nonempty text is an operator assertion, not
machine verification of independence or chess correctness. For ambiguous mate-to-CP
evidence, independently establish and explain the concept decision; otherwise preserve
`unresolved`. A concept-specific verified negative can be authorized just like a
positive, but an absent proposal or low-error control is not a negative label.

The operator ID must differ from the annotating reviewer ID, but alternate identifiers
cannot prove separate people. Identity and justification appear in success output
only; the existing registry schema remains an array of review hashes. Retain required
authenticated audit context externally. Registry initialization cannot overwrite
existing data, repeated approval is rejected, and existing hashes are preserved.

Any contradictory or unresolved decision for the selected candidate/concept blocks
new issuance. There is no final-review revision, disagreement-resolution, or approval
revocation command. Do not silently discard conflicting drafts. Freeze review/evidence
inputs during approval and approved collections during downstream use; new later
disagreements require human governance, not automatic rewriting of prior approvals.

The registry uses exclusive locking plus stale-content checks and atomic replacement.
It rechecks reviews and artifacts before committing, but does not transactionally lock
external artifact writers. Stale locks require investigation with writers stopped.
The full persistence/trust limitations are documented in `human_adjudication.md`.

Exit 0 means the requested operation succeeded; 2 means invalid input, conflict or
I/O failure; 1 means unexpected failure. No command trains or exports data.
`training_ready` retains its limited meaning: at least one eligible label with valid
integrity, not sufficient data, balanced concepts or leakage-safe train/test splits.

Focused C2 verification:

```powershell
.venv\Scripts\python.exe -B -m pytest tests/ml/test_approval_workflow.py tests/ml/test_approval_store.py tests/ml/test_approval_cli.py tests/ml/test_dataset_readiness.py -q
```
