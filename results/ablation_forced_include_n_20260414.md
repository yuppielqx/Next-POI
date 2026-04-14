# forced_include_n Ablation (2026-04-14)

## Setting

We study how many nearby POIs should be forcibly preserved in the final LLM shortlist.
This parameter is controlled by `forced_include_n` in `src/config.py`.

Compared settings:

- `forced_include_n = 0`
- `forced_include_n = 3`
- `forced_include_n = 5`

## Overall Results

| Setting | Hit@1 | Hit@5 | Hit@10 | N@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.2265 | 0.4196 | 0.5011 | 0.3568 | 0.3116 |
| 3 | 0.2250 | 0.4274 | 0.5063 | 0.3586 | 0.3123 |
| 5 | 0.2242 | 0.4166 | 0.5011 | 0.3561 | 0.3107 |

## Key Findings

1. `forced_include_n = 3` gives the best overall ranking quality.
   - It achieves the strongest `Hit@5`, `Hit@10`, `N@10`, and `MRR`.
   - Although `Hit@1` is slightly below the `0` setting, the gap is very small (`0.2250` vs `0.2265`).

2. `forced_include_n = 0` is slightly better only on strict top-1 precision.
   - This suggests that removing forced nearby candidates can keep the first rank marginally cleaner.
   - However, it also leaves some useful nearby destinations unrecovered deeper in the list.

3. `forced_include_n = 5` is too aggressive.
   - Forcing too many nearby POIs into the shortlist introduces ranking noise.
   - This weakens both top-rank precision and downstream list quality.

## Interpretation

The results show a precision-recall trade-off in shortlist construction:

- With `0`, the shortlist relies more heavily on user history and collaborative signals.
- With `5`, nearby candidates are over-injected and compete with stronger historical candidates.
- With `3`, the model gets enough geographic coverage to recover plausible local destinations, without overwhelming the shortlist.

This behavior is consistent with the current pipeline design:

- nearby candidates help candidate recall;
- but excessive forced inclusion can dilute the signal seen by the final LLM ranker;
- therefore, a moderate nearby quota provides the best balance.

## Recommended Default

We recommend using `forced_include_n = 3` as the default configuration.

## Snapshot References

- `results/snapshots/20260414_ablation_forced0`
- `results/snapshots/20260414_ablation_forced3`
- `results/snapshots/20260414_llm_timeaware_nearby_source_forced5`
