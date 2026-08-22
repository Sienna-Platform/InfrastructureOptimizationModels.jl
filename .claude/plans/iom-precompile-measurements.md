# IOM precompile workload — measurements

Machine: MacBook-Pro-6.local (Apple Silicon), Julia 1.12.5, macOS (Darwin 25.5.0).

Environment notes (pre-existing drift fixed before baseline, unrelated to the workload):
- Stale untracked `test/Manifest.toml` pinned JSON 1.4.0 vs IS4-branch IS 3.6.0 requiring ≥1.5 → manifest regenerated (`rm test/Manifest.toml` + `Pkg.instantiate()`).
- Measurement scripts' `import MathOptInterface as MOI` fails in the test env (MOI is not a direct test dep) → replaced with `const MOI = JuMP.MOI` in both scripts.

## Baseline (unmodified tree, commit 337dfaf)

| metric | value |
|---|---|
| IOM own precompile (s, from Pkg log) | 9.4 |
| Pkg.precompile wall, warm no-op (s) | 1.26 |
| LOAD_SECONDS | 2.69 |
| FIRST_CALL_SECONDS | 2.39 |
| SECOND_CALL_SECONDS | 0.0 |
| TOTAL_TTFX_SECONDS | 7.43 |
| FIRST_TYPESET_A | 2.40 |
| SECOND_TYPESET_B | 0.374 |
| REPEAT_TYPESET_A | 0.0 |
| pkgimage size (newest .dylib) | 5.1 MB |

## After (workload in tree, 2026-07-24)

| metric | baseline | after | delta |
|---|---|---|---|
| IOM own precompile (s) | 9.4 | 10.4 | +1.0 |
| LOAD_SECONDS | 2.69 | 2.68 | ~0 |
| FIRST_CALL_SECONDS | 2.39 | 0.81 | **−66%** |
| SECOND_CALL_SECONDS | 0.0 | 0.0 | none |
| TOTAL_TTFX_SECONDS | 7.43 | 5.69 | −23% |
| FIRST_TYPESET_A | 2.40 | 0.77 | **−68%** |
| SECOND_TYPESET_B | 0.374 | 0.418 | ~same (inherent per-type residual) |
| pkgimage size (.dylib) | 5.1 MB | 9.0 MB | +3.9 MB |

Escape hatch: with `precompile_workload = false` in `test/LocalPreferences.toml`, IOM
precompiles in 8.1 s (workload skipped) and loads clean; pref file removed afterwards
(cache for the default config is reused, no extra recompile).

Acceptance: **PASS.**
1. FIRST_CALL_SECONDS −66% and FIRST_TYPESET_A −68% — both ≥50% (criterion met; the
   container prototype's −79% included a slower machine's larger shared-compile share).
2. SECOND_CALL_SECONDS at ~0 ms — no runtime regression from locked-in specializations.
3. Precompile increase +1.0 s — well under the 15 s budget.
