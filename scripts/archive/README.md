# Archived investigation scripts

These produced the conclusions the project rests on and are kept so those
conclusions stay checkable. **They are not part of the working pipeline**, and
several were written against setup definitions that have since been replaced —
expect to adjust them before re-running.

| Script | What it established | Why it is archived |
|---|---|---|
| `04_train.py` | Model ladder, 15-config sweep, CPCV | Pools both exit families into one model — the flaw `10_family_models.py` fixes |
| `05_barrier_sensitivity.py` | 36 brackets across all setups | Applies one bracket to every setup; `11_reversion_lab.py` does it per family |
| `06_diagnostics.py` | Model vs simple rules; top-decile composition | Superseded by `12_reversion_final.py`'s same-day portfolio benchmark |
| `07_holdout_all.py` | All models on the 2021-25 holdout | Uses the pool benchmark now known to inflate results |
| `08_veto_analysis.py` | The "model as veto" hypothesis | Dead end: strong on holdout, did not reproduce on dev |
| `09_target_variants.py` | Alternative label targets | Dead end: no variant's top decile cleared zero |

`reports/archive/` holds their outputs, which reference setups that no longer exist.
