# Stage C3.2: guarded evidence regeneration

Implemented entry point (production execution is deferred to C3.3):

```powershell
.venv\Scripts\python.exe -B -m scripts.regenerate_tactical_evidence --output-dir data/labeled/c3_v2
```

Run from the repository root. The destination parent must exist and the destination
must be absent. Defaults retain the complete production candidate/feature inputs,
source PGN/manifest and historical evidence bundle. Adjacent candidate, feature and
labeling manifests and the historical quality report are required. CLI path overrides
support isolated fixture bundles; the cohort size and engine policy are not configurable.

The wrapper validates historical integrity, requires 24 verified v1 records and
checks the existing round-robin selector's exact ID order before engine launch.
It fingerprints all protected inputs and pins the audited Stockfish 19 executable
SHA-256. It owns a new engine, uses depth 12, Threads 1 and Hash 64, and validates
MultiPV 1 and unrestricted strength in captured engine metadata. MultiPV is managed
by python-chess's single-PV analyse call. Engine resource identity is captured even
though the evaluation cache has zero capacity. Historical verification is never
passed into the generator; both endpoint requests use the existing verifier.

The unchanged labeling pipeline writes to a private staging directory. Acceptance
requires successful verification of the exact cohort, legal PVs, achieved depth 12
on nonterminal endpoints, valid terminal depth-zero handling, matching original
comparison fields, matching engine profiles and full readiness integrity/current-rule
replay. Zero proposals are allowed. No reviews or approvals are supplied or generated;
eligible labels must remain zero. All protected byte hashes are checked again before
the staged bundle is renamed to its final destination.

Output contains the existing three labeling files plus `regeneration_audit.json`:
historical and generated artifact hashes, protected input hashes, exact cohort IDs
and canonical digest, engine identity/settings, timestamp, Git revision when available,
ML implementation hashes, and acceptance summary. The audit is an execution record,
not authenticated attestation or human approval. The evidence schema is unchanged.

Failures return nonzero (2 for errors, 130 for interruption) and do not publish the
accepted destination. Temporary staging is cleaned on ordinary errors/interruption.
Process termination or power failure can leave staging behind; investigate it rather
than treating it as accepted evidence. Inputs and the destination parent must remain
quiescent. Hash checks detect persistent external edits but are not filesystem locks.
Publication uses same-volume rename on the intended Windows environment; this does
not claim universal network-filesystem or power-loss durability. Resolved upstream
directories, their descendants/ancestors, and existing outputs are rejected.

Finite search and FEN repetition-history limitations remain unchanged. Engine
verification and conservative proposals are not human labels. Existing v1 review
hashes cannot authorize v2 records; use Stage A/B and trusted Stage C2 approval only
after real independent review. A later engine binary change needs an explicit policy
update, not a command-line bypass of the pinned hash.

Focused validation:

```powershell
.venv\Scripts\python.exe -B -m pytest tests/ml/test_regeneration.py -q
```

The new suite uses only synthetic fixtures and mocked engine searches. Existing
pipeline tests include their own synthetic real-engine smoke test. No test invokes
production regeneration.
