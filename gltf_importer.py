"""
Performance-oriented entry point for GLTF Importer for 3ds Max.

This module preserves the public API/UI from gltf_importer_legacy.py and replaces
only the mesh-construction hot path. Geometry is sent to MAXScript in bulk rather
than issuing a pymxs call for every vertex/face/UV/normal.
"""

import time

from gltf_importer_legacy import *  # noqa: F401,F403
import gltf_importer_legacy as _legacy

rt = _legacy.rt
HAS_MAX = _legacy.HAS_MAX


if HAS_MAX:
    # Keep the remaining unavoidable per-normal fallback loop inside MAXScript.
    # This avoids one Python -> MAXScript transition per normal.
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
        # Helpers are an optimization only; Python fallbacks below remain valid.
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

    try:
        # Avoid repeated viewport invalidation while the mesh is being assembled.
        try:
            rt.disableSceneRedraw()
            redraw_disabled = True
        except Exception:
            pass

        # Build MAXScript values in Python, then cross the pymxs boundary once.
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

        # GLTF indices address all vertex attributes together, so when UV count
        # matches POSITION count the UV face topology is identical to geometry.
        if uvs and len(uvs) == num_verts:
            max_tverts = [
                rt.Point3(uv[0], (1.0 - uv[1]) if opts.flip_uvs_v else uv[1], 0.0)
                for uv in uvs
            ]
            try:
                # Build map faces in one native call, then replace all tverts in
                # one setMesh call. This avoids per-face and per-UV pymxs calls.
                rt.meshop.setNumMaps(mesh, 2, keep=True)
                rt.meshop.setMapSupport(mesh, 1, True)
                rt.meshop.defaultMapFaces(mesh, 1)
                rt.setMesh(mesh, tverts=max_tverts)
            except Exception:
                # Compatibility fallback: still only one Python -> MAXScript call;
                # the per-UV loop executes inside MAXScript.
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

        # Preserve explicit GLTF normals. There is no equivalent documented bulk
        # setNormal call, so run the loop on the MAXScript side when possible.
        if normals and len(normals) == num_verts:
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
            started = _phase(log, f"'{name}' explicit normals", started)

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

        if opts.auto_smooth and not normals:
            try:
                rt.addModifier(mesh, rt.Smooth(autoSmooth=True, threshold=opts.smooth_angle))
            except Exception:
                pass

        # Keep the original topology semantics. This phase is timed separately
        # because quad reconstruction can dominate very large imports.
        if opts.topology != 'mesh':
            topo_start = time.perf_counter()
            try:
                if opts.topology == 'triangles':
                    ttp = rt.Turn_to_Poly()
                    ttp.limitPolySize = True
                    ttp.maxPolySize = 3
                    rt.addModifier(mesh, ttp)
                    rt.convertToPoly(mesh)
                    log.info(f"'{name}': Editable Poly (triangles)")
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


# Patch the implementation used by functions defined in the legacy module.
_legacy._create_max_mesh = _create_max_mesh


if __name__ == "__main__":
    show_ui()
