# ORBIT-Causal 0.3.7 release notes

This protocol-hardening release converts the blocked 250K Windows result into
prespecified validation work. C1 is the primary candidate, C3 is its superiority
baseline, and C6 remains a negative guarded-policy ablation. A pre-run selection
contract binds the official source, config, deterministic row window, candidate
roles, and release thresholds before locked outcomes are evaluated.

Bounded Criteo runs now support deterministic SplitMix64 priority windows through
`--sample-offset`. The new 250K qualification window uses offset 250,000, which is
disjoint from the v0.3.6 offset-zero pilot. The admitted row-ID set is hashed and
checked against the independently authenticated selection contract.

The primary C1-minus-C3 comparison uses paired pretreatment-profile cluster
bootstrap inference. C1 must also pass standalone value, calibration, slice,
operational, integrity, schema, evaluator, and abstention gates. C1-minus-C5 and
C1-minus-C6 are secondary diagnostics and cannot replace the primary comparison.

The Windows figure durability fix exercised in the v0.3.6 isolated overlay is now
part of canonical `reporting.py`: temporary figure files are opened in `rb+` before
`fsync`, avoiding the native Windows `EBADF` failure.

## Preserved v0.3.6 history

This release is a qualification-focused recovery of the preserved 0.3.2 candidate. It adds no new modeling family. The material changes close operational and statistical claim gaps.

Every Criteo experiment now performs the pinned official-file admission inside the
same run before loading a bounded sample or full benchmark. A schema-compatible
lookalike can no longer receive a public-benchmark label. Source digest and admission
evidence are bound into the sealed run outputs.
Validation and loading operate on the same authenticated private snapshot, closing
the file-replacement window between admission and training.
The standalone official-file validator now uses the same snapshot rule, so it cannot
parse one file and later authenticate replacement bytes.
Oversized inputs are rejected as soon as they cross the pinned byte bound instead of
being copied to exhaustion. Scoring CSVs are likewise bounded and frozen before their
header and rows are parsed.

Profile collision auditing now rechecks original typed values. Adjacent integers above
`2**53` can no longer be silently merged after float projection. Report regeneration
uses a byte snapshot instead of shared hard links, and the sdist now carries the CI
workflow and scoring example required for a complete source handoff.

Release smoke helpers do not inject the repository `src` directory ahead of an
installed wheel. Qualification derives package version and test count from the
candidate, isolates coverage state, fingerprints `.github`, verifies mandatory sdist
handoff files, rebuilds the wheel from the sdist and requires byte identity, and keeps
report limitations aligned with the executed data role. Artifact manifests now bind
the exact ORBIT-Causal release in addition to the numerical runtime dependencies.
The generated dependency lock includes hashes for every admitted distribution, and
both installation and vulnerability audit fail closed when hashes are missing.
Artifact-only smoke environments use a separate hash-complete runtime lock and are
removed sequentially, reducing peak qualification storage without reusing installs. Artifact
environments live outside the evidence output, cleanup is registered for failed runs, and the
manifest is written only after their absence is verified.
Child-process run staging and rollback directories are also removed and checked before evidence is
sealed, preventing transient copies from leaking into a release qualification output.
Smoke run directories are created outside the evidence output and removed as ephemeral test state;
the qualification logs retain the independent seal-verification results and hashes.

The public Criteo validator now reads `.csv` and `.csv.gz`, streams the entire file, and can require the official v2.1 contract: 13,979,592 rows, 311,422,618 compressed bytes, and SHA-256 `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`.

Bounded `--max-rows` experiments no longer take a potentially ordered prefix. They stream the complete source and retain a deterministic priority sample keyed by original row number, preserving both randomized treatment arms and row-level provenance.

`G3_value` now passes only when the paired pretreatment-profile cluster-bootstrap confidence interval for C6 minus C3 has a positive lower bound. A candidate with positive standalone value but no demonstrated improvement is held.

Supported release scope is CPU execution on Python 3.11 and 3.12. This local qualification does not claim a native Windows, GitHub-hosted, GPU/CUDA, full Criteo training, financial-client, or production-impact result unless the corresponding evidence is explicitly recorded.
# v0.3.9 locked benchmark evaluation

- Freezes the v0.3.8 E7/C3 portfolio artifact rather than retraining after holdout review.
- Adds a pre-specified value non-inferiority and calibration-superiority protocol.
- Uses a new disjoint 2,000,000-row priority window and 4,000 paired profile-cluster draws.
- Adds calibration-RMSE influence-function inference and deterministic replay auditing.
- The locked public-benchmark protocol passed all co-primary gates and guardrails.
