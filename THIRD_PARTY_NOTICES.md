# Third-Party Notices

The MIT License in `LICENSE` applies to the ORBIT-Causal software authored for
this repository. It does not replace or override the terms governing
third-party data or data-derived material identified below.

## Criteo Uplift Modeling Dataset v2.1

- Provider: Criteo AI Lab
- Dataset page: https://ailab.criteo.com/criteo-uplift-prediction-dataset/
- License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0
  International (CC BY-NC-SA 4.0)
- License text: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
- Associated work: Eustache Diemert, Artem Betlei, Christophe Renaudin, and
  Massih-Reza Amini, "A Large Scale Benchmark for Uplift Modeling," AdKDD and
  TargetAd Workshop, KDD 2018.

The raw Criteo dataset is not included in this repository. Users must obtain it
from the provider and comply with the provider's terms.

`artifacts/frozen_candidate_portfolio.joblib` contains model parameters learned
from the Criteo Uplift Modeling Dataset v2.1. It is included solely to support
reproducibility of this noncommercial research benchmark. To the extent this
artifact constitutes Adapted Material under the dataset license, it is made
available under CC BY-NC-SA 4.0, not the repository's MIT License. The artifact
does not contain the raw dataset and has not been endorsed by Criteo.

The repository's configurations, contracts, aggregate metrics, calibration
tables, slice summaries, bootstrap draws, and audit records were produced from
experiments using that dataset. They do not include raw row-level Criteo data.

No third-party trademarks or patent rights are granted by this notice.
