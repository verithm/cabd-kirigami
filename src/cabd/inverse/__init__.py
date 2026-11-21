"""Inverse design: target (theta2, u2, log10 dE_r) -> design variables, with the four methods, scored by the referee.

Rules fixed 2026-11-16 (training plan doc), identical for every method:
  core    objective (smooth max of tau-normalized errors), P(bi>mono) >= 0.9 constraint, SLSQP from 20 Sobol starts,
          300 s wall-clock cap per query, candidate selection, feasible()
  run     one CLI for inverse v1 (512 training geometries, 5 seeds, dev / final queries) and v2 (models refitted on
          1024 / 1536 geometries, seed 0, the 40 ansys40 queries)
  score   referee columns (v1: referees fitted on the 512 geometries; v2: referees fitted on the same training size)
  patch   rerun single queries whose wall-clock cap was eaten by Windows standby
Results: results/inverse/*.csv (v1) and results/inverse/v2/*.csv (v2), one row per candidate.
"""
