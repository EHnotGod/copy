# Construction Pool Audit (automatic first pass)

Semantic-similarity rules nominate samples for REVIEW. Distribution drift is recorded separately and does not by itself reject a sample. Only schema failures or exact raw-prompt matches are automatically DROP; explicit manual verdicts are applied from the override file.

## Summary

| Benchmark | Raw | Construction | KEEP | REVIEW | DROP |
|---|---:|---:|---:|---:|---:|
| CoreBench | 1080 | 400 | 383 | 0 | 17 |
| WISE | 1000 | 400 | 369 | 0 | 31 |

CoreBench source-model metadata: {"Gemini": 400}

## Highest CoreBench raw-similarity candidates

| Construction | Raw | Score | Reasons |
|---|---|---:|---|
| R-LR-G024 | R-LR-090 | 0.249 | possible_raw_structural_overlap, manual_verdict:same four antique keys on velvet setup as raw R-LR-090 |
| R-LR-G026 | R-LR-025 | 0.225 | possible_raw_structural_overlap, manual_verdict:same row-of-alchemical-vessels logic template as raw R-LR-025 |
| R-LR-G008 | R-LR-080 | 0.214 | possible_raw_structural_overlap, manual_verdict:same row-of-jars logic template and entities as raw R-LR-080 |
| R-LR-G004 | R-LR-080 | 0.212 | possible_raw_structural_overlap, manual_verdict:same row-of-jars logic template and entities as raw R-LR-080 |
| R-LR-G025 | R-LR-048 | 0.184 | possible_raw_structural_overlap, manual_verdict:reuses the robot logic-puzzle setting from raw R-LR-048 |
| C-MR-G035 | C-MR-063 | 0.154 | possible_raw_structural_overlap, manual_verdict:reuses the same laptop/desk/lamp relation scene family as raw C-MR-063 |
| C-MR-G010 | C-MR-063 | 0.151 | possible_raw_structural_overlap, manual_verdict:reuses the same laptop/desk/lamp relation scene family as raw C-MR-063 |
| R-GR-G033 | R-GR-001 | 0.147 | possible_raw_structural_overlap, manual_verdict:same elemental weapon-forge generalization family as raw R-GR-001 |
| R-GR-G034 | R-GR-003 | 0.147 | possible_raw_structural_overlap, manual_verdict:same parameterized bio-garden plant simulator family as raw R-GR-003 |
| R-AR-G010 | R-AR-008 | 0.144 | possible_raw_structural_overlap, manual_verdict:same spider-web source analogy as raw R-AR-008 |
| R-LR-G020 | R-LR-025 | 0.142 | possible_raw_structural_overlap, manual_verdict:same row-of-vessels logic template and entities as raw R-LR-025 |
| R-PR-G008 | R-PR-088 | 0.142 | possible_raw_structural_overlap, manual_verdict:same sushi rolling and slicing procedure as raw R-PR-088 |
| C-MR-G037 | C-MR-063 | 0.137 | possible_raw_structural_overlap, manual_verdict:laboratory relation scene is materially distinct from the nearest office raw case |
| R-LR-G023 | R-LR-025 | 0.136 | possible_raw_structural_overlap, manual_verdict:same row-of-flasks logic template and entities as raw R-LR-025 |
| R-GR-G017 | R-GR-004 | 0.132 | possible_raw_structural_overlap, manual_verdict:reuses the same runic magical-forge generalization family as raw R-GR-004 |

## Highest WISE raw-similarity candidates

| Construction | Raw | Score | Reasons |
|---|---|---:|---|
| 99 | 108 | 0.543 | possible_same_raw_knowledge_target, manual_verdict:same Leaning Tower of Pisa target as raw 108 |
| 286 | 976 | 0.503 | possible_same_raw_knowledge_target, manual_verdict:same lead-iodide precipitation reaction as raw 976 |
| 1 | 238 | 0.497 | possible_same_raw_knowledge_target, manual_verdict:same kiwi target as raw 238 |
| 258 | 997 | 0.449 | possible_same_raw_knowledge_target, manual_verdict:same liquid-bromine appearance as raw 997 |
| 277 | 967 | 0.416 | possible_same_raw_knowledge_target, manual_verdict:same scratched-galvanized-steel corrosion target as raw 967 |
| 161 | 760 | 0.379 | possible_same_raw_knowledge_target, manual_verdict:same honey-bee waggle-dance knowledge as raw 760 |
| 270 | 961 | 0.376 | possible_same_raw_knowledge_target, manual_verdict:same steel-wool-in-oxygen reaction as raw 961 |
| 267 | 992 | 0.372 | possible_same_raw_knowledge_target, manual_verdict:same dry-ice sublimation target as raw 992 |
| 104 | 96 | 0.366 | possible_same_raw_knowledge_target, manual_verdict:same Golden Gate Bridge target as raw 96 |
| 50 | 135 | 0.353 | possible_same_raw_knowledge_target, manual_verdict:same Commonwealth remembrance-flower association as raw 135 |
| 214 | 832 | 0.351 | possible_same_raw_knowledge_target, manual_verdict:same swing-at-maximum-angle state as raw 832 |
| 257 | 991 | 0.350 | possible_same_raw_knowledge_target, manual_verdict:same sodium-water reaction as raw 991 |
| 148 | 52 | 0.341 | possible_same_raw_knowledge_target, manual_verdict:same torii target as raw 52 |
| 228 | 850 | 0.341 | possible_same_raw_knowledge_target, manual_verdict:same thin-film-interference visual knowledge as raw 850 |
| 303 | 908 | 0.338 | possible_same_raw_knowledge_target, manual_verdict:same copper/nitric-acid reaction as raw 908 |
