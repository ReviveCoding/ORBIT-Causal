# ORBIT-Causal

**Orthogonal, Risk-Bounded Intervention Targeting**

ORBIT-Causal is a reproducible offline causal ML benchmark for heterogeneous treatment-effect estimation, calibrated uplift scoring, and capacity-constrained intervention targeting. The final v0.3.9 release evaluates a frozen candidate once on a disjoint 2,000,000-row holdout from the randomized Criteo Uplift v2.1 dataset.

## Final locked result

| Metric | E7 candidate | C3 baseline |
|---|---:|---:|
| Policy value at 20% capacity | **0.006041** | 0.005644 |
| Policy value 95% CI | **[0.005510, 0.006527]** | [0.005130, 0.006142] |
| Calibration RMSE | **0.005088** | 0.009470 |

Combined frozen-portfolio inference took **25.41 ms per 1,000 rows** on the audited environment.

The paired policy-value difference was `0.000397`, with a 95% confidence interval of `[0.000111, 0.000682]`. All six pre-specified release gates passed, prior evaluation overlap was zero, and the independent replay audit passed.

The locked verdict is `RELEASE_ELIGIBLE` for this public-data benchmark. It is not evidence of online advertising lift, production ROI, or external validity beyond the benchmark.

## Repository contents

- `src/orbit_causal/`: reusable training, calibration, policy, scoring, and verification code
- `tests/`: unit, property, integration, and release-contract tests
- `configs/`: reproducible demo and Criteo configurations
- `scripts/run_locked_holdout.py`: v0.3.9 single-use locked evaluation
- `scripts/audit_locked_replay.py`: independent replay verification
- `artifacts/frozen_candidate_portfolio.joblib`: frozen E7/C3 candidate artifact
- `contracts/`: parent and final selection contracts
- `results/`: compact final metrics, bootstrap draws, calibration, and slice evidence
- `audit/`: final audit report, evidence inventory, run seal, and replay audit

Raw Criteo data and the 130.8MB row-level replay are intentionally excluded. The replay is generated output and exceeds GitHub's regular 100MiB per-file limit. The compact evidence retained here is sufficient to inspect the reported model performance; a full replay can be regenerated from the pinned source and contracts.

## Reproduce the package tests

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-deps .
pytest -q
```

## Reproduce the locked evaluation

Obtain `criteo-research-uplift-v2.1.csv.gz` from the dataset provider and verify its pinned identity through the runner. Then run:

```bash
PYTHONPATH=src python scripts/run_locked_holdout.py seal \
  --data /path/to/criteo-research-uplift-v2.1.csv.gz \
  --config configs/criteo_v2.json \
  --package-dir src/orbit_causal \
  --artifact artifacts/frozen_candidate_portfolio.joblib \
  --parent-contract contracts/v0.3.8-selection-contract.json \
  --contract /trusted/path/v0.3.9-selection-contract.json
```

The locked runner refuses a source, artifact, code fingerprint, configuration, contract, or row set that differs from the sealed protocol. See `python scripts/run_locked_holdout.py --help` for the evaluation command and required authenticated contract hash.

## Method and data references

- [Criteo Uplift Prediction Dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)
- [Criteo Uplift v2 benchmark paper](https://arxiv.org/abs/2111.10106)
- [Practical evaluation of heterogeneous treatment-effect estimation](https://proceedings.mlr.press/v151/xu22c.html)
- [Causal isotonic calibration](https://proceedings.mlr.press/v202/van-der-laan23a.html)

## License and third-party material

The original ORBIT-Causal software is released under the MIT License. The
Criteo dataset and any covered data-derived material remain subject to Criteo's
CC BY-NC-SA 4.0 terms. See `LICENSE` and `THIRD_PARTY_NOTICES.md` for details.

## Version

Final benchmark release: `v0.3.9`
