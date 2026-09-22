"""
Performance-oriented entry point for GLTF Importer for 3ds Max.

This module preserves the public API/UI from gltf_importer_legacy.py and replaces
only the mesh-construction hot path. Geometry is sent to MAXScript in bulk rather
than issuing a pymxs call for every vertex/face/UV/normal.

The legacy module is loaded explicitly from the directory containing this file.
This is important in 3ds Max because Run Script / Script Editor execution does not
always add the script directory to Python's sys.path.
"""

import importlib.util
import os
import sys
import time


# Explicit normals are particularly expensive in 3ds Max because setNormal()
# creates explicit normal data. For very large meshes, use Max's smoothing
# calculation instead. Small meshes keep exact glTF normals.
FAST_EXPLICIT_NORMAL_VERTEX_LIMIT = 100000


def _load_legacy_module():
    script_file = globals().get("__file__")
    if not script_file:
        raise RuntimeError(
            "3ds Max did not provide __file__. Please run gltf_importer.py with "
            "Scripting > Run Script instead of pasting its contents into the console."
        )

    script_dir = os.path.dirname(os.path.abspath(script_file))
    legacy_path = os.path.join(script_dir, "gltf_importer_legacy.py")

    if not os.path.isfile(legacy_path):
        raise FileNotFoundError(
            "gltf_importer_legacy.py was not found beside gltf_importer.py.\n"
            f"Expected: {legacy_path}"
        )

    module_name = "_gltf_importer_legacy_local"
    spec = importlib.util.spec_from_file_location(module_name, legacy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for: {legacy_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_legacy = _load_legacy_module()

# Re-export the legacy module's public API so existing usage continues to work.
for _name in dir(_legacy):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_legacy, _name)

rt = _legacy.rt
HAS_MAX = _legacy.HAS_MAX


if HAS_MAX:
    # Compatibility helpers. The normal helper is used only for meshes below
    # FAST_EXPLICIT_NORMAL_VERTEX_LIMIT; large meshes intentionally skip explicit
    # normals because this operation is the dominant import bottleneck.
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
    """Create one 3ds Max mesh using bulk pymxs/MAXScript operations."""
    scale = opts.get_scale_value()

    if not HAS_MAX:
        log.info(f"[parse-only] Mesh '{name}': {len(positions)} verts, {len(indices) // 3} tris")
        log.mesh_count += 1
        return None

    indices = _legacy._validate_mesh_data(name, positions, normals, uvs, indices, opts, log)
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

        # Geometry: build Python lists of MAXScript Point3 values and cross the
        # pymxs boundary once instead of once per vertex/face.
        max_vertices = []
        append_vert = max_vertices.append
        for pos in positions:
            cx, cy, cz = _legacy._convert_position(pos[0], pos[1], pos[2], opts)
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

        # UVs: glTF indices address POSITION and TEXCOORD together, so when
        # counts match, map-face topology is identical to geometry topology.
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
                    # Last-resort compatibility path for older Max versions.
                    rt.meshop.setNumMaps(mesh, 2)
                    rt.meshop.setMapSupport(mesh, 1, True)
                    rt.meshop.setNumMapVerts(mesh, 1, num_verts)
                    rt.meshop.defaultMapFaces(mesh, 1)
                    for i, uv in enumerate(max_tverts):
                        rt.meshop.setMapVert(mesh, 1, i + 1, uv)
            started = _phase(log, f"'{name}' UVs", started)

        # Exact glTF normals are retained for modest meshes. On large meshes,
        # setNormal() is pathologically expensive in Max because it creates and
        # maintains explicit-normal data for each vertex.
        if normals and len(normals) == num_verts:
            if num_verts <= FAST_EXPLICIT_NORMAL_VERTEX_LIMIT:
                normal_start = time.perf_counter()
                max_normals = []
                append_normal = max_normals.append
                for nrm in normals:
                    nx, ny, nz = _legacy._convert_normal(nrm[0], nrm[1], nrm[2], opts)
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
            _legacy._apply_node_transform(mesh, transform, opts)

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

        # If exact normals were skipped, apply Max's native angle-based smoothing.
        # This keeps the fast path visually useful without building 500k+ explicit
        # normal records.
        if opts.auto_smooth and not imported_explicit_normals:
            smooth_start = time.perf_counter()
            try:
                rt.addModifier(mesh, rt.Smooth(autoSmooth=True, threshold=opts.smooth_angle))
                _phase(log, f"'{name}' auto-smooth", smooth_start)
            except Exception as e:
                log.warn(f"Auto-Smooth failed on '{name}': {e}")

        # GLTF primitive mode 4 is already triangles. For the Triangles option,
        # Turn_to_Poly(maxPolySize=3) is redundant and very expensive on large
        # meshes; convert directly to Editable Poly instead.
        if opts.topology != 'mesh':
            topo_start = time.perf_counter()
            try:
                if opts.topology == 'triangles':
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


# Functions such as import_file() were defined in the legacy module and resolve
# _create_max_mesh from that module's globals. Replace that symbol with our fast
# implementation so both the UI and public API automatically use it.
_legacy._create_max_mesh = _create_max_mesh


if __name__ == "__main__":
    show_ui()
