# ORBIT-Causal v0.3.9 Locked 2M Holdout Audit

## Decision

The pre-specified v0.3.9 benchmark protocol completed successfully. E7 passed
both co-primary gates and every locked guardrail on the previously unused
2,000,000-row Criteo priority window `[750000, 2750000)`.

**Verdict: `RELEASE_ELIGIBLE` for the public-benchmark subset claim.**

This verdict does not establish production business utility. The public Criteo
data do not contain the intervention costs, customer utility, or operating
constraints needed to define a production non-inferiority margin.

## Locked protocol

| Item | Pre-specified value |
|---|---|
| Candidate | Frozen v0.3.8 E7 artifact |
| Baseline | C3 |
| Holdout | 2,000,000 previously unused rows |
| Priority-rank window | `[750000, 2750000)` |
| Policy capacity | 20% |
| Value gate | E7−C3 one-sided 95% lower bound > `-0.00025` |
| Calibration gate | RMSE(E7)−RMSE(C3) one-sided 95% upper bound < `0` |
| Bootstrap | 4,000 paired pretreatment-profile cluster draws |
| Calibration inference | Paired cluster bootstrap of the empirical influence function |
| Decision rule | Both co-primary gates and all guardrails must pass |

The contract was sealed before outcome-bearing holdout rows were loaded.

- Selection contract SHA-256: `6df2b1271058421ed3f4e3640678e421468b0de7ec1c934b967c12014cc67967`
- Frozen E7/C3 artifact SHA-256: `047dbfd1f0ae48418b0de15e6f5da299a5d6fb575ecf81790afacf7a0a5df004`
- Official Criteo v2.1 source SHA-256: `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`
- Holdout row-ID SHA-256: `0c0316d2091404b376fd361244d3c29c30968c151c4ec4c20b7e7e867b124a91`

Changing the sealed value margin from `-0.00025` to `-0.00050` caused the
runner to reject the contract before creating an output directory.

## Results

| Metric | E7 | C3 | E7−C3 |
|---|---:|---:|---:|
| AIPW policy value | 0.006041 | 0.005644 | **+0.000397** |
| Two-sided 95% CI | [0.005510, 0.006527] | [0.005130, 0.006142] | [0.000111, 0.000682] |
| Calibration RMSE | **0.005088** | 0.009470 | **−0.004382** |
| Calibration difference, two-sided 95% CI | | | **[−0.005921, −0.002863]** |

Co-primary gate statistics:

- Value non-inferiority one-sided 95% lower bound: `+0.000147571`
- Locked value margin: `-0.000250000`
- Calibration superiority one-sided 95% upper bound: `-0.003140244`
- Locked calibration margin: `0`

Both comparisons passed in the pre-specified direction. The two-sided interval
for policy value difference was also above zero, but superiority was not the
locked value claim and is reported only as secondary characterization.

## Guardrails

| Gate | Result |
|---|---|
| Integrity and disjointness | PASS |
| Value non-inferiority | PASS |
| Calibration superiority | PASS |
| Candidate standalone value | PASS |
| Slice regression guardrail | PASS |
| Operational limits | PASS |

Additional observations:

- Holdout rows: `2,000,000`
- Unique pretreatment-profile clusters: `1,956,145`
- Policy disagreement rate: `8.0304%`
- Inference time: `25.413 ms / 1,000 rows`
- Frozen artifact size: `5,047,429 bytes`
- Locked execution time: `447.49 seconds`

## Independent replay audit

A separate audit implementation read the sealed row-level replay and
independently recomputed:

1. deterministic SplitMix64 holdout membership;
2. disjointness from the prior three 250K windows;
3. row-level doubly robust effect scores;
4. candidate and baseline AIPW policy scores;
5. both ten-bin calibration RMSE values and their influence functions;
6. every one of the 4,000 paired cluster-bootstrap draws;
7. decision quantiles, file inventory hashes, and run seal.

Audit results:

- Prior-window overlap: `0`
- Duplicate holdout row IDs: `0`
- Regenerated bootstrap draws: `4,000`
- Maximum absolute draw difference: `6.1821×10⁻¹⁵`
- Sealed evidence files verified: `7`
- Independent replay status: `PASS`
- Run seal SHA-256: `965afd52f3649f4d58c98217286c8a4bb38a5661cbac6e2421a8d6211b2a5fa4`

## Interpretation boundary

The result supports releasing E7 as the benchmark candidate under the locked
public-data protocol. It does not justify silently converting the benchmark
margin into a production margin, and it does not replace a cost-aware external
validation study. A full 13.98M-row characterization may be run next as a
secondary descriptive analysis, but it must not alter this locked verdict.

## Method references

- [Criteo Uplift Prediction Dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)
- [Xu and Yadlowsky, A Practical Evaluation of Heterogeneous Treatment Effect Estimation](https://proceedings.mlr.press/v151/xu22c.html)
- [van der Laan et al., Causal Isotonic Calibration for Heterogeneous Treatment Effects](https://proceedings.mlr.press/v202/van-der-laan23a.html)
- [ICH E9 Statistical Principles for Clinical Trials](https://database.ich.org/sites/default/files/E9_Guideline.pdf)
