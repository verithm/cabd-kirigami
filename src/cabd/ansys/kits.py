"""Declarative description of every Ansys kit that was handed out (built by cabd.ansys.kit).

A kit is a folder that one PC runs unattended (run_collect.bat -> collect.ps1 -> Workbench job.wbjn -> Mechanical
mech_job.py -> MAPDL post_job.inp): the solver files of ansys_kit/, jobs.csv (geometries in run order), landmarks.csv
(points whose displacement is extracted), kit_config.json (mesh rules, limits), README.md (Korean guide for the
person running it) and step/<geom_id>.step. Every kit uses the same solver settings, so all results merge.

Groups (one entry each; `kits` are the folder names):
  base     PC1-3   train core 512 + test 256 (job k -> PC k mod 3 + 1, so finished jobs stay close to a Sobol prefix
                   even if one PC falls behind; order train -> test stage 1 -> test stage 2). collect.ps1 as in ansys_kit.
  ext1     PC4-6   params_train_ext1 (TR0512..TR1023), round robin from priority 768. collect.ps1 with two fixes:
                   a state list that a per-job variable overwrote (PowerShell names are case-insensitive), and a
                   coarse retry that cannot mesh is not_converged when the first run left a curve (it repeats on every run).
  ext2     PC7-9   params_train_ext2 (TR1024..TR1536), same as ext1 from priority 1280.
  retry1   1 PC    the 5 ext1 geometries whose solver never started on PC5/PC6 (status "error: no solver directory" on
                   both states, all within one minute on PC5 2026-11-12 22:57-22:58 and at the first job on PC6: a start
                   failure, not a geometry problem). Same solver files and collect.ps1 as ext, so results merge directly.
  final    1 PC    finishes every state still missing on PC1-3 (built from their results; see kit.build_final).
  verify   ALL (one PC) or V1-V3: inverse-design verification kit 1, the rank-1 candidate of each method for the 40
                   ansys40 queries (seed 0, final query set; rule fixed 2026-11-17 before any Ansys result).
                   Geometry ids V<method><query>: G = GP, R = ROM+GP, M = MLP, C = cVAE (VG0018 = GP candidate for TE0018);
                   identical geometries (|du| < 1e-6) are solved once and share an id. Whole queries per PC (contiguous
                   blocks), so every PC finishes complete four-method comparisons.
  verify2  P1-P3 (or ALL): kit 2, forward models refitted on 1024 and 1536 training geometries, same rule:
                   2 sizes x 4 methods x 40 = 320 candidates; ids <size><method><query>, size W = 1024, X = 1536.
                   A query without a feasible rank-1 candidate has nothing to solve (counted as a failure when scoring).

`version` is the kit_version written to kit_config.json; the historical dates are kept so a rebuild of a historical
kit is identical to the kit that was handed out (None = today's date, for a new kit).
"""

CONFIG = dict(
    global_size_mm=0.6,
    hinge_mesh_attempts="[(0.1, 8), (0.1, 16), (0.15, 8), (0.15, 12), (0.2, 8)]",  # (in-plane mm, layers in y)
    retry_mesh="(0.2, 4)",  # hinge mesh of the retry after a failed solve
    node_limit=127000, job_timeout_min_per_state=90, post_timeout_min=30,
)
# files copied unchanged into every kit (collect.ps1 and README.md are written per kit)
SOLVER_FILES = ["run_collect.bat", "job.wbjn", "mech_job.py", "post_job.inp", "template.wbpz",
                "make_result_zip.bat", "make_result_zip.ps1"]
METHODS = {"gp": "G", "rom_gp": "R", "mlp": "M", "cvae": "C"}      # job order within a query
SIZES2 = {1024: "W", 1536: "X"}                                     # verification kit 2 size letters

GROUPS = {
    "base": dict(kind="base", kits=["CABD_collect_PC1", "CABD_collect_PC2", "CABD_collect_PC3"],
                 version="collect-v3 (2026-11-04)", step="base", assignment="assignment_all.csv"),
    "ext1": dict(kind="ext", stage=1, kits=["CABD_collect_PC4", "CABD_collect_PC5", "CABD_collect_PC6"],
                 version="collect-v4-ext1 (2026-11-09)", step="ext1", assignment="assignment_ext1.csv"),
    "ext2": dict(kind="ext", stage=2, kits=["CABD_collect_PC7", "CABD_collect_PC8", "CABD_collect_PC9"],
                 version="collect-v4-ext2 (2026-11-11)", step="ext2", assignment="assignment_ext2.csv"),
    "retry1": dict(kind="retry", label="retry1", from_group="ext1", geoms=["TR0514", "TR0705", "TR0708", "TR0711", "TR0714"],
                   kits=["CABD_collect_retry1"], version="collect-v4-retry1 (2026-11-17)", step="ext1"),
    "final": dict(kind="final", kits=["CABD_final"], version="final-v1 (2026-11-08)", step="base",
                  sources=("base", (1, 2, 3))),
    "verify": dict(kind="verify", kit=1, kits=["CABD_verify_V1", "CABD_verify_V2", "CABD_verify_V3"],
                   version="verify-v1 (2026-11-17)", step="verify"),
    "verify_all": dict(kind="verify", kit=1, kits=["CABD_verify_ALL"], version="verify-v1 (2026-11-17)", step="verify"),
    "verify2": dict(kind="verify", kit=2, kits=["CABD_verify2_P1", "CABD_verify2_P2", "CABD_verify2_P3"],
                    version="verify-v2 (2026-11-21)", step="verify2"),
}
