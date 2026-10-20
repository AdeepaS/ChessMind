# Human adjudication and training eligibility

`labeling/adjudication.py` adds `human_adjudication_v1` records and the
`adjudicated_concepts_v1` eligibility policy. It does not annotate real candidates,
export training data, train a model, or change proposal rules.

Engine verification records search results and legal continuations. Rule proposals
are provisional interpretations of those results. Human adjudication is a separate
decision about one candidate and one taxonomy concept. Training eligibility requires
that decision plus validated evidence and independently authorized review approval.
Neither engine `verified` nor rule `proposed` means ground truth.

## Record and review policy

`AdjudicationRecord` is frozen and validates at construction. It contains:

- Stable candidate ID and an existing `MistakeConcept` identifier.
- Explicit `positive`, `negative`, or `unresolved` decision.
- Reviewer identifier, nonempty rationale, and UTC ISO-8601 review timestamp.
- `draft` (default) or `finalized` review state.
- Adjudication schema and eligibility-policy versions.
- Canonical JSON SHA-256 hashes of the candidate, feature and rule-evidence records,
  and byte SHA-256 hashes of the original source dataset and source manifest.

A negative is explicit and concept-specific. An absent annotation, empty proposal
list, low-error control, or abstention never supplies a negative target. Other
concepts on the same move remain unknown. Finalized unresolved decisions remain
ineligible. Drafts can coexist, but the complete submitted review collection must
contain at most one finalized record per candidate/concept, even if duplicates
agree. Conflicts fail closed; this version provides no resolution workflow.

## Trust boundary

A name, timestamp, rationale or self-declared independence flag cannot authenticate
a human review. `is_training_eligible` therefore requires `approved_review_hashes`:
a separately trusted frozenset of canonical annotation hashes. Its default is empty.
An external review authority must establish reviewer identity, independence from
automatic proposal generation, and final approval before authorizing an exact hash.
Do not populate this set from submitted annotations or let an annotator approve
their own submission. The hash binds the reviewer, decision, rationale and evidence;
changing any field invalidates approval.

No authentication service or trusted approval registry is implemented here. Without
such independently established approval, all decisions fail eligibility. Hashes
provide integrity, not proof of expertise, independence, or chess correctness.
The predicate cannot defend against a caller fabricating its trusted approval set
or hiding a conflicting review: pass the complete collection, not a selected subset.
For this MVP the approval set is an explicit trust input, not an authentication
system or a requirement to deploy a review service. It can be maintained by a
trusted operator after independent review. Exposing it directly to untrusted
annotation submitters would bypass the approval safeguard.

## Creating one annotation

After a real human review, construct `AdjudicationRecord` with that reviewer's actual
identifier, rationale, timestamp and explicit concept decision. Use
`record_sha256(candidate)`, `record_sha256(feature)`, and
`record_sha256(evidence)` for the three record references; use
`hashlib.sha256(source_bytes).hexdigest()` for each source artifact. Start as a draft;
finalize only when review is complete. Serialize with `json.dumps(asdict(record))`;
load with `AdjudicationRecord(**json.loads(text))`, which rejects unknown fields.

An independently authorized authority then approves `record_sha256(asdict(record))`.
Call `is_training_eligible(record, reviews=complete_reviews, candidate=candidate,
feature=feature, evidence=evidence, source_dataset=source_bytes,
source_manifest=manifest_bytes, approved_review_hashes=trusted_hashes)`.
Only finalized positive/negative decisions can return true. This describes an API
workflow, not an instruction to fabricate approval or annotate existing data.

The predicate recomputes candidate identity and features using existing validation,
checks version/linkage and canonical hashes, validates saved scores and legal PVs,
and checks actual source bytes against provenance and manifest game/index linkage.
It also requires valid upstream input-hash references, unchanged automatic-review
metadata, and legal tactical evidence continuations. Upstream candidate/feature
file hashes are references only: those complete files are not supplied here;
the exact individual records are bound by the approved annotation's hashes.
It does not open the paths stored in records or perform engine searches. Source
bytes are integrity-checked; it does not reparse the source PGN to re-prove every
position. Independent review must examine the linked evidence and source context.

## Historical records and limits

Supported evidence uses `mistake_proposals_v1` and `conservative_tactics_v2`.
An unreviewed saved v1 proposal is not an `AdjudicationRecord`, lacks independent
approval, and has an unsupported evidence rule version. It cannot be a training
target. Historical artifacts remain unchanged; any future v2 evidence preparation
must be separate and preserve original provenance. No automatic migration exists.

A v2 abstention may support a separately finalized, independently approved human
annotation; the abstention alone remains ineligible. The original automatic status
and `unreviewed` flag are preserved as historical facts, not overwritten to simulate
human review. Adjudication does not require a positive rule proposal.

This foundation does not establish class coverage, leakage-safe splits, reviewer
agreement, an exporter, or model readiness. Some saved mate-to-CP cases may remain
unresolved even after review. Only deterministic synthetic tests create eligible
fixture annotations; no real dataset labels are created.

## Task 9: dataset readiness inspection

From the repository root, inspect the existing saved pilot without engine searches
or file outputs:

```powershell
.venv\Scripts\python.exe -B -m scripts.inspect_training_readiness
```

The CLI prints deterministic, sorted JSON to stdout. It has no output-file option,
so it cannot overwrite an input artifact. Exit **0** means a valid inspection,
including a valid empty or zero-eligible dataset. Exit **2** means invalid inputs
(including missing files, unsupported schemas, malformed JSON, conflicting finalized
reviews, or integrity failures); exit **1** indicates an unexpected execution failure.
Argument usage errors also exit 2. Do not use exit 0 alone as permission to train:
inspect `training_ready` and the scope described below.

Defaults use the existing candidate, feature, proposal, source PGN and source
manifest paths. Override them with `--candidate-path`, `--feature-path`,
`--evidence-path`, `--source-dataset-path` and `--source-manifest-path`.
Adjacent `candidate_manifest.json`, `feature_manifest.json` and
`labeling_manifest.json` are verified when present; explicit paths can be supplied
with `--candidate-manifest-path`, `--feature-manifest-path` and
`--labeling-manifest-path`. A referenced label-quality report is hash-checked too.
Recorded source paths are provenance text, never automatically opened.

Supply actual review inputs only after independent review:

```powershell
.venv\Scripts\python.exe -B -m scripts.inspect_training_readiness --adjudications-path <reviews.jsonl> --approvals-path <trusted-approval-hashes.json>
```

The angle-bracket paths are placeholders, not bundled review artifacts.
Adjudications are JSONL `AdjudicationRecord` objects. Approvals are a JSON array of
canonical annotation SHA-256 strings, supplied by a separately trusted operator.
The CLI never creates approvals or derives them from submissions. Supplying a list
does not authenticate its author: the caller must preserve the trust boundary
described above. A missing optional argument means no reviews/approvals; explicitly
supplying a nonexistent file is an input error. Submit the complete review collection
so conflicting finalized decisions cannot be hidden by selecting a subset.

Report fields:

- `integrity_valid` and `integrity_errors`: collection-wide validation. Invalid
  input clears counts and eligibility results; partial successes are not usable.
- `artifact_sha256`: actual input-byte hashes, including supplied review/approval
  files and available manifests.
- `counts`: source/candidate games, candidates, feature-complete candidates,
  evidence coverage, engine verification statuses, provisional proposal counts,
  historical evidence counts, and all four proposal statuses (including zeros).
- `versions`: candidate, feature, evidence, rule, adjudication and eligibility
  policy version inventories.
- `adjudication_status`: draft/finalized decisions, approved/unapproved and
  ineligible counts, plus supplied/unused approval hashes. Approval alone is not
  eligibility; drafts and unresolved decisions remain ineligible.
- `eligible_labels`: total candidate/concept decisions accepted by the existing
  `is_training_eligible` predicate, with explicit positive and negative counts per
  concept. Unannotated concepts stay unknown.
- `blocking_reasons`: reasons no eligible labels are available, or an
  `artifact_integrity_failure`. Historical evidence can coexist with eligible
  current evidence but is always excluded from eligibility.
- `training_ready`: true only when integrity passes and at least one label passes
  the existing eligibility policy. **This is a label-readiness gate, not a claim
  of adequate data, class coverage, safe partitions or readiness to fit a model.**
  `readiness_scope` repeats this limitation in the machine-readable report.

The inspector reuses strict bounded JSONL reading, canonical candidate/feature
validation, source hashes and manifest game/index linkage, saved score/PV validation,
current tactical rules and the adjudication predicate. It performs no engine search,
training export, splitting, source PGN replay or new chess adjudication. Artifacts
are limited to 32 MiB each; candidate/feature/evidence JSONL uses the existing
10,000-line and 2 MiB-line pilot limits, and adjudications use a 10,000-line limit.
Inputs are rehashed before returning to detect changes during inspection.
Current-rule evidence is checked against replayed rule decisions. Historical v1
evidence is structurally/linkage validated and inventoried, not migrated or accepted
as eligible. Unknown versions fail collection integrity.

The saved pilot should report 213 feature-complete candidates, 24 engine-verified
records, four historical provisional proposals, zero adjudications and **zero eligible
labels**, with `training_ready = false`. The remaining 189 candidates have no saved
verification record. Engine verification and legal continuations are evidence, not
independent confirmation of a concept. Zero eligible labels block supervised
training; neither abstention, `requires_review`, nor a low-error control supplies a
negative target.

Before training, prepare supported current-version evidence separately, obtain
explicit independent human concept decisions with rationale and provenance, finalize
them, and establish separate approval for their exact hashes. Then assess coverage,
review quality and leakage-safe partitions. Those later steps are not implemented
by this inspector.

Focused verification:

```powershell
.venv\Scripts\python.exe -B -m pytest tests/ml/test_dataset_readiness.py -q
```

Positive and negative eligible fixtures in these tests are explicitly synthetic.
They do not create real reviewed labels or change the saved artifacts.
