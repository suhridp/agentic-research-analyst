# Verifier stress test (28 sentences from `data/synthetic`, seed 0)

Accept rate % [Wilson 95% CI]. Higher is better for originals/paraphrases; for every other row the number is a **false-accept rate** (lower is better).

| Claim type | should be | n | lexical_v1 (numbers only) | lexical_v2 (+negation) |
|---|---|---|---|---|
| original | accepted | 28 | 100 [88, 100] | 100 [88, 100] |
| number_change | rejected | 10 | 0 [0, 28] | 0 [0, 28] |
| negation | rejected | 7 | 100 [65, 100] | 0 [0, 35] |
| direction_flip | rejected | 7 | 100 [65, 100] | 100 [65, 100] |
| hedge_removal | rejected | 3 | 100 [44, 100] | 100 [44, 100] |
| qualifier_drop | rejected | 6 | 100 [61, 100] | 100 [61, 100] |
| term_swap | rejected | 28 | 100 [88, 100] | 100 [88, 100] |
