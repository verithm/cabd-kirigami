# Mechanical scripting for one (geometry, state) run (runs inside Mechanical via SendCommand).
# Prepended by job.wbjn: job_header.py values (X_W, X_D, T_H, T_A, mesh settings, MESH_ONLY), LOG_PATH, STATE.
import time

TAG = STATE


def log(msg):
    f = open(LOG_PATH, "a")
    f.write("%s %s\n" % (TAG, msg))
    f.close()


def fail(msg):
    log("ERROR " + msg)
    raise Exception(msg)


model = ExtAPI.DataModel.Project.Model
analysis = model.Analyses[0]
settings = analysis.AnalysisSettings
TOL = 1e-3  # mm
SOLVE_START_TRIES = 3


def selection(ids):
    sel = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)
    sel.Ids = list(ids)
    return sel


# ---- bodies: map the 6 named STEP bodies (names come from make_step_files.py)
NAMES = ["hinge_tip_upper", "hinge_tip_lower", "hinge_apex", "arm_upper", "arm_lower", "frame"]
bodies = {}
for b in model.Geometry.GetChildren(DataModelObjectCategory.Body, True):
    hits = [n for n in NAMES if n in b.Name]
    if len(hits) != 1:
        fail("unexpected body name %s" % b.Name)
    bodies[hits[0]] = b
if sorted(bodies) != sorted(NAMES):
    fail("bodies found: %s" % sorted(bodies))
geo = dict((n, bodies[n].GetGeoBody()) for n in NAMES)
log("bodies " + " ".join("%s:id%d:vol%.4f" % (n, geo[n].Id, geo[n].Volume) for n in NAMES))

# ---- materials: slightly wet = soft hinges + rigid frame/arms; fully wet = soft everywhere
for n in NAMES:
    soft = n.startswith("hinge") or STATE == "full"
    bodies[n].Material = "Tough Black Soft" if soft else "Tough Black Rigid"
log("materials " + " ".join("%s=%s" % (n, bodies[n].Material) for n in NAMES))

# ---- contacts: regenerate bonded contacts for the new geometry (expect 6 hinge-frame/arm faces)
for g in list(model.Connections.Children):
    g.Delete()
model.Connections.CreateAutomaticConnections()
contacts = model.Connections.GetChildren(DataModelObjectCategory.ContactRegion, True)
log("contacts %d types=%s" % (len(contacts), ",".join(sorted(set(str(c.ContactType) for c in contacts)))))
if len(contacts) != 6 or any(str(c.ContactType) != "Bonded" for c in contacts):
    fail("expected 6 bonded contacts")

# ---- boundary conditions: same rule as the shared project
#   fixed support = back-wall outer face (x = X_W) of the frame
#   displacement +15 mm in X = the two depth-direction edges of the apex-hinge inner face (x = X_D)
fix_faces = [f.Id for f in geo["frame"].Faces if abs(f.Centroid[0] - X_W) < TOL]
disp_edges = [e.Id for e in geo["hinge_apex"].Edges
              if all(abs(v.X - X_D) < TOL for v in e.Vertices) and abs(e.Length - 6.0) < TOL]
log("fix_faces=%s disp_edges=%s" % (fix_faces, disp_edges))
if len(fix_faces) != 1 or len(disp_edges) != 2:
    fail("could not locate support face / displaced edges")
for ch in analysis.Children:
    kind = ch.GetType().Name
    if kind == "FixedSupport":
        ch.Location = selection(fix_faces)
    elif kind == "Displacement":
        ch.Location = selection(disp_edges)
        log("displacement X=%s" % ch.XComponent.Output.DiscreteValues)
if model.NamedSelections:
    for ns in list(model.NamedSelections.Children):
        ns.Delete()
for name, ids in [("FIX_FACE", fix_faces), ("DISP_EDGES", disp_edges)]:
    ns = model.AddNamedSelection()
    ns.Name = name
    ns.Location = selection(ids)
    ns.SendToSolver = True

# ---- mesh: global GLOBAL_SIZE as in the shared project. Hinges: MultiZone (hex) with a uniform in-plane size
#      and a fixed number of layers along y: first of HINGE_MESH_ATTEMPTS that meshes cleanly within NODE_LIMIT.
#      Sweep fails on the tip hinges and tets exceed the node limit (dev tests on TR0362, 2026-11-02).
mesh = model.Mesh
mesh.ElementSize = Quantity(GLOBAL_SIZE, "mm")
HINGES = ["hinge_tip_upper", "hinge_tip_lower", "hinge_apex"]
SOLID_TYPES = ("kHex20", "kTet10", "kWedge15", "kPyramid13")


def hinge_controls(size, divs):
    for ch in list(mesh.Children):
        ch.Delete()
    for n in HINGES:
        sz = mesh.AddSizing()
        sz.Location = selection([geo[n].Id])
        sz.ElementSize = Quantity(size, "mm")
        m = mesh.AddAutomaticMethod()
        m.Location = selection([geo[n].Id])
        m.Method = MethodType.MultiZone
        es = mesh.AddSizing()
        es.Location = selection([e.Id for e in geo[n].Edges if abs(e.Length - 6.0) < TOL])
        es.Type = SizingType.NumberOfDivisions
        es.NumberOfDivisions = divs
        es.Behavior = SizingBehavior.Hard


def hinge_mesh_info():
    try:
        data = ExtAPI.DataModel.MeshDataByName("Global")
        out = []
        for n in ["hinge_tip_upper", "hinge_tip_lower", "hinge_apex"]:
            reg = data.MeshRegionById(geo[n].Id)
            kinds = {}
            for i in reg.ElementIds:
                k = str(data.ElementById(i).Type)
                kinds[k] = kinds.get(k, 0) + 1
            out.append("%s:%s" % (n, ",".join("%s=%d" % kv for kv in sorted(kinds.items()))))
        return " ".join(out)
    except Exception as exc:
        return "n/a (%s)" % exc


def hinges_solid():
    # GenerateMesh can "succeed" with a failed sweep (only face/edge elements in that body): check each hinge
    try:
        data = ExtAPI.DataModel.MeshDataByName("Global")
        for n in HINGES:
            reg = data.MeshRegionById(geo[n].Id)
            if not any(str(data.ElementById(i).Type) in SOLID_TYPES for i in reg.ElementIds):
                return False
        return True
    except Exception:
        return False


def clear_messages():
    try:
        ExtAPI.Application.Messages.Clear()
    except Exception:
        pass


def mesh_errors(since):
    msgs = list(ExtAPI.Application.Messages)[since:]
    texts = [" ".join(m.DisplayString.split())[:160] for m in msgs if str(m.Severity) == "Error"]
    return [t for t in texts if t]  # Mechanical re-lists earlier errors with an empty text after Clear()


# (in-plane size, layers along y) in order of preference; shape-check failures move to the next one
attempts = [(h, d, "multizone_%gmm_%dl" % (h, d)) for h, d in HINGE_MESH_ATTEMPTS]
mesh_rule = None
for size, divs, label in attempts:
    hinge_controls(size, divs)
    t0 = time.time()
    clear_messages()
    n_msg = len(list(ExtAPI.Application.Messages))
    try:
        mesh.GenerateMesh()
        nodes, elems = mesh.Nodes, mesh.Elements
    except Exception as exc:
        nodes, elems = 0, 0
        log("mesh %s raised %s" % (label, exc))
    errs = mesh_errors(n_msg)
    solid = hinges_solid()
    log("mesh %s nodes=%d elements=%d hinges_solid=%s mesh_s=%.1f errors=%s" % (label, nodes, elems, solid, time.time() - t0, errs))
    if 0 < nodes <= NODE_LIMIT and solid and not errs:
        mesh_rule = label
        break
if mesh_rule is None:
    fail("no mesh within %d nodes" % NODE_LIMIT)
log("mesh_rule=%s nodes=%d elements=%d" % (mesh_rule, nodes, elems))
log("hinge_mesh " + hinge_mesh_info())


def set_prop(name, value):
    try:
        setattr(settings, name, value)
    except Exception as exc:
        log("FAILED set %s (%s)" % (name, exc))


def configure():
    # benchmark kit setting (survey 1.8)
    set_prop("AutomaticTimeStepping", AutomaticTimeStepping.On)
    set_prop("DefineBy", TimeStepDefineByType.Substeps)
    set_prop("InitialSubsteps", 40)
    set_prop("MinimumSubsteps", 20)
    set_prop("MaximumSubsteps", 160)
    set_prop("ContactData", False)
    # Program Controlled picks PCG for the hex hinge mesh; PCG fails on every substep (~5 min each) before
    # falling back to sparse (dev test TR0362, 2026-11-02). The shared project ran sparse as well.
    set_prop("SolverType", SolverType.Direct)


def completed_time():
    # time of the last converged substep = column 7 of the solution monitor file (file.mntr)
    # (a text search of solve.out is unreliable: its header says "TERMINATE ANALYSIS IF NOT CONVERGED")
    done = 0.0
    try:
        for line in open(analysis.WorkingDir + "file.mntr"):
            f = line.split()
            if len(f) >= 7 and f[0].isdigit() and f[1].isdigit():
                done = float(f[6])
    except Exception:
        pass
    return done


def converged():
    return completed_time() >= 0.9999


if MESH_ONLY:
    log("mesh_only stop")
else:
    for ch in list(analysis.Solution.Children):
        if ch.GetType().Name != "SolutionInformation":
            ch.Delete()
    # Retry on failure = coarser hinge mesh. Convergence tests (TR0000, 2026-11-02): the hinges distort sooner on finer
    # meshes, and the former fixed-75-substep retry took 2.3 h to stop earlier than the auto run.
    configure()
    for mode in ["auto", "auto_coarse"]:
        if mode == "auto_coarse":
            size, divs = RETRY_MESH
            hinge_controls(size, divs)
            mesh_rule = "multizone_%gmm_%dl" % (size, divs)
            try:
                mesh.GenerateMesh()
            except Exception as exc:
                log("mesh %s raised %s" % (mesh_rule, exc))
            if not hinges_solid():
                log("retry mesh failed %s" % mesh_rule)
                break
            log("mesh_rule=%s nodes=%d elements=%d" % (mesh_rule, mesh.Nodes, mesh.Elements))
        # A solve that stops before its first converged substep says nothing about the geometry (seen: "memory size
        # requested is not available", TE0110, 2026-11-02): same mesh again, never the coarser retry.
        for start_try in range(SOLVE_START_TRIES):
            analysis.Solution.ClearGeneratedData()
            t0 = time.time()
            analysis.Solution.Solve(True)
            status = str(analysis.Solution.Status)
            ok = status == "Done" and converged()
            log("solve mode=%s solve_s=%.1f status=%s completed_time=%.4f ok=%s" % (mode, time.time() - t0, status, completed_time(), ok))
            for msg in ExtAPI.Application.Messages:
                if str(msg.Severity) in ("Error", "Warning"):
                    log("message %s: %s" % (msg.Severity, " ".join(msg.DisplayString.split())[:300]))
            if ok or completed_time() > 0:
                break
            if start_try < SOLVE_START_TRIES - 1:
                time.sleep(60)
        if ok:
            break
        if completed_time() <= 0:
            log("solver_start_failed mode=%s" % mode)
            break
        if mode == "auto":
            # keep the failed auto run (its substeps up to the failure are still data) before the retry
            import System.IO
            keep = LOG_PATH.replace("wb_log.txt", "%s_auto_failed" % STATE)
            System.IO.Directory.CreateDirectory(keep)
            for f in ["file.rst", "ds.dat", "solve.out", "file.mntr"]:
                src = analysis.WorkingDir + f
                if System.IO.File.Exists(src):
                    System.IO.File.Copy(src, keep + "\\" + f, True)
            log("auto_failed_dir=%s" % keep)
    log("final_mode=%s ok=%s" % (mode, ok))
    log("solver_dir=%s" % analysis.WorkingDir)
