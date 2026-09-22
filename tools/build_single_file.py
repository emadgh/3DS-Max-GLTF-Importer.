from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "gltf_importer_legacy.py"
OUT = ROOT / "gltf_importer.py"

src = LEGACY.read_text(encoding="utf-8")

# Single-file build: integrate the fast mesh builder directly into the original
# importer, so gltf_importer.py has no sibling-module dependency.
src = src.replace("import traceback\n", "import traceback\nimport time\n", 1)
src = src.replace("self.topology = 'quads'  # 'triangles', 'quads', 'ngons'", "self.topology = 'triangles'  # 'triangles', 'quads', 'ngons'", 1)
src = src.replace("cmb_topology.SelectedIndex = 1  # Default to Quads", "cmb_topology.SelectedIndex = 0  # Default to Triangles (native glTF topology / fast path)", 1)

FAST_BLOCK = r'''# Large-mesh performance tuning.
FAST_EXPLICIT_NORMAL_VERTEX_LIMIT = 100000

if HAS_MAX:
    try:
        rt.execute(r"""
            global GLTFImporter_SetNormalsFast
            fn GLTFImporter_SetNormalsFast obj normalArray =
            (
                local n = normalArray.count
                if n > obj.numverts do n = obj.numverts
                for i = 1 to n do setNormal obj i normalArray[i]
            )

            global GLTFImporter_SetUVsFast
            fn GLTFImporter_SetUVsFast obj uvArray =
            (
                meshop.setNumMaps obj 2 keep:true
                meshop.setMapSupport obj 1 true
                meshop.defaultMapFaces obj 1
                local n = uvArray.count
                local mapCount = meshop.getNumMapVerts obj 1
                if n > mapCount do n = mapCount
                for i = 1 to n do meshop.setMapVert obj 1 i uvArray[i]
            )
        """)
    except Exception:
        pass


def _phase(log, name, started):
    elapsed = time.perf_counter() - started
    log.info(f"{name}: {elapsed:.2f}s")
    log.flush()
    return time.perf_counter()


def _create_max_mesh(name, positions, normals, uvs, indices, opts, log, transform=None):
    """Create a 3ds Max mesh using bulk pymxs/MAXScript operations."""
    scale = opts.get_scale_value()

    if not HAS_MAX:
        log.info(f"[parse-only] Mesh '{name}': {len(positions)} verts, {len(indices) // 3} tris")
        log.mesh_count += 1
        return None

    indices = _validate_mesh_data(name, positions, normals, uvs, indices, opts, log)
    if indices is None:
        return None

    num_verts = len(positions)
    num_faces = len(indices) // 3
    if num_verts == 0 or num_faces == 0:
        log.warn(f"Skipping empty mesh '{name}'")
        return None

    started = time.perf_counter()
    redraw_disabled = False
    imported_explicit_normals = False

    try:
        try:
            rt.disableSceneRedraw()
            redraw_disabled = True
        except Exception:
            pass

        max_vertices = []
        append_vert = max_vertices.append
        for pos in positions:
            cx, cy, cz = _convert_position(pos[0], pos[1], pos[2], opts)
            append_vert(rt.Point3(cx * scale, cy * scale, cz * scale))

        max_faces = []
        append_face = max_faces.append
        if opts.flip_normals:
            for i in range(0, num_faces * 3, 3):
                append_face(rt.Point3(indices[i] + 1, indices[i + 2] + 1, indices[i + 1] + 1))
        else:
            for i in range(0, num_faces * 3, 3):
                append_face(rt.Point3(indices[i] + 1, indices[i + 1] + 1, indices[i + 2] + 1))

        mesh = rt.mesh(vertices=max_vertices, faces=max_faces)
        mesh.name = name
        started = _phase(log, f"'{name}' bulk geometry", started)

        if uvs and len(uvs) == num_verts:
            max_tverts = [
                rt.Point3(uv[0], (1.0 - uv[1]) if opts.flip_uvs_v else uv[1], 0.0)
                for uv in uvs
            ]
            try:
                rt.meshop.setNumMaps(mesh, 2, keep=True)
                rt.meshop.setMapSupport(mesh, 1, True)
                rt.meshop.defaultMapFaces(mesh, 1)
                rt.setMesh(mesh, tverts=max_tverts)
            except Exception:
                try:
                    rt.GLTFImporter_SetUVsFast(mesh, max_tverts)
                except Exception:
                    rt.meshop.setNumMaps(mesh, 2)
                    rt.meshop.setMapSupport(mesh, 1, True)
                    rt.meshop.setNumMapVerts(mesh, 1, num_verts)
                    rt.meshop.defaultMapFaces(mesh, 1)
                    for i, uv in enumerate(max_tverts):
                        rt.meshop.setMapVert(mesh, 1, i + 1, uv)
            started = _phase(log, f"'{name}' UVs", started)

        if normals and len(normals) == num_verts:
            if num_verts <= FAST_EXPLICIT_NORMAL_VERTEX_LIMIT:
                normal_start = time.perf_counter()
                max_normals = []
                append_normal = max_normals.append
                for nrm in normals:
                    nx, ny, nz = _convert_normal(nrm[0], nrm[1], nrm[2], opts)
                    append_normal(rt.Point3(nx, ny, nz))
                try:
                    rt.GLTFImporter_SetNormalsFast(mesh, max_normals)
                except Exception:
                    for i, nrm in enumerate(max_normals):
                        rt.setNormal(mesh, i + 1, nrm)
                imported_explicit_normals = True
                _phase(log, f"'{name}' explicit normals", normal_start)
            else:
                log.warn(
                    f"'{name}': Skipping {num_verts} explicit glTF normals in fast mode "
                    f"(limit {FAST_EXPLICIT_NORMAL_VERTEX_LIMIT}); using Auto-Smooth instead"
                )

        if transform:
            _apply_node_transform(mesh, transform, opts)

        rt.update(mesh)
        started = _phase(log, f"'{name}' mesh update", started)

        try:
            actual_faces = rt.getNumFaces(mesh)
            actual_verts = rt.getNumVerts(mesh)
            if actual_faces != num_faces:
                log.warn(f"'{name}': Expected {num_faces} faces, Max reports {actual_faces}")
            log.info(f"'{name}': Created {actual_verts} verts, {actual_faces} faces")
        except Exception:
            pass

        try:
            mesh.backfacecull = False
        except Exception:
            try:
                rt.setProperty(mesh, "backfacecull", False)
            except Exception:
                pass

        if opts.weld_vertices:
            weld_start = time.perf_counter()
            try:
                before = rt.getNumVerts(mesh)
                rt.meshop.weldVertsByThreshold(mesh, rt.getNumVerts(mesh), opts.weld_threshold)
                after = rt.getNumVerts(mesh)
                welded = before - after
                if welded > 0:
                    log.weld_count += welded
                    log.info(f"Welded {welded} vertices on '{name}'")
            except Exception as e:
                log.warn(f"Weld failed on '{name}': {e}")
            _phase(log, f"'{name}' weld", weld_start)

        if opts.auto_smooth and not imported_explicit_normals:
            smooth_start = time.perf_counter()
            try:
                rt.addModifier(mesh, rt.Smooth(autoSmooth=True, threshold=opts.smooth_angle))
                _phase(log, f"'{name}' auto-smooth", smooth_start)
            except Exception as e:
                log.warn(f"Auto-Smooth failed on '{name}': {e}")

        if opts.topology != 'mesh':
            topo_start = time.perf_counter()
            try:
                if opts.topology == 'triangles':
                    # glTF mode 4 is already triangles. Do not run Turn_to_Poly
                    # merely to recreate the topology we already have.
                    rt.convertToPoly(mesh)
                    log.info(f"'{name}': Editable Poly (triangles, fast conversion)")
                elif opts.topology == 'quads':
                    ttp = rt.Turn_to_Poly()
                    ttp.limitPolySize = True
                    ttp.maxPolySize = 4
                    rt.addModifier(mesh, ttp)
                    rt.convertToPoly(mesh)
                    log.info(f"'{name}': Editable Poly (quads)")
                else:
                    rt.convertToPoly(mesh)
                    log.info(f"'{name}': Editable Poly (n-gons)")
            except Exception as e:
                log.warn(f"Failed topology conversion on '{name}': {e}")
                try:
                    rt.convertToPoly(mesh)
                except Exception:
                    pass
            _phase(log, f"'{name}' topology conversion", topo_start)

        log.mesh_count += 1
        log.flush()
        return mesh

    except Exception as e:
        log.error(f"Failed to create mesh '{name}': {e}")
        return None
    finally:
        if redraw_disabled:
            try:
                rt.enableSceneRedraw()
            except Exception:
                pass
        try:
            rt.redrawViews()
        except Exception:
            try:
                rt.completeRedraw()
            except Exception:
                pass
'''

start = src.index("def _create_max_mesh(")
end = src.index("\ndef _apply_node_transform", start)
src = src[:start] + FAST_BLOCK + "\n\n" + src[end + 1:]

# Keep WinForms objects and Python callbacks alive for the lifetime of the form.
show_marker = "def show_ui():"
if show_marker in src and "_UI_KEEPALIVE = []" not in src:
    src = src.replace(show_marker, "_UI_KEEPALIVE = []\n\n\n" + show_marker, 1)

# Prevent a second import from being triggered while the synchronous import is
# running, then restore the form even if Max or the importer raises an exception.
handler_start = src.index("    def on_import(*args):")
handler_end = src.index("    def on_close(*args):", handler_start)
handler = src[handler_start:handler_end]
handler = handler.replace(
    "        filepaths = list(_file_paths)\n\n        lbl_status.Text",
    "        filepaths = list(_file_paths)\n\n        btn_import.Enabled = False\n        form.UseWaitCursor = True\n\n        lbl_status.Text",
    1,
)
handler = handler.rstrip() + r'''
        finally:
            # 3ds Max can temporarily disable an owned WinForms window while
            # executing synchronous scene operations. Always restore it here.
            try:
                form.Enabled = True
                btn_import.Enabled = True
                form.UseWaitCursor = False
                txt_custom.Enabled = (cmb_scale.SelectedIndex == 3)
            except Exception:
                pass
            try:
                rt.enableSceneRedraw()
                rt.redrawViews()
            except Exception:
                pass
            try:
                form.Refresh()
                form.Activate()
                rt.dotNetClass("System.Windows.Forms.Application").DoEvents()
            except Exception:
                pass

'''
src = src[:handler_start] + handler + src[handler_end:]

# Retain callbacks and the form itself. PyMXS/.NET event bridges can otherwise
# lose Python callback objects after a long synchronous import + GC cycle.
wire_marker = "    # Show and parent to Max window\n    form.Show()"
keepalive = r'''    # Strong references for the .NET event bridge / callbacks.
    global _UI_KEEPALIVE
    try:
        _UI_KEEPALIVE[:] = [
            item for item in _UI_KEEPALIVE
            if item[0] is not None and not item[0].IsDisposed
        ]
    except Exception:
        _UI_KEEPALIVE[:] = []
    _UI_KEEPALIVE.append((
        form,
        on_add_files, on_add_folder, on_remove, on_clear,
        on_scale_change, on_import, on_close,
    ))

    # Show and parent to Max window
    form.Show()'''
if wire_marker not in src:
    raise RuntimeError("Could not find UI show marker")
src = src.replace(wire_marker, keepalive, 1)

# Make the single-file nature obvious in the header.
src = src.replace("Version: 0.3.0", "Version: 0.4.0-fast-single", 1)
src = src.replace(
    "Imports GLTF/GLB files into 3ds Max with geometry and PBR materials.",
    "Imports GLTF/GLB files into 3ds Max with geometry and PBR materials.\nSingle-file fast build with bulk large-mesh construction and reusable UI.",
    1,
)

OUT.write_text(src, encoding="utf-8", newline="\n")
print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")
