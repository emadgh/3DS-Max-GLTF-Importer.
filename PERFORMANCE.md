# Large-mesh performance test

This branch changes only the mesh-construction hot path and preserves the existing importer UI/API.

## Why the current importer becomes extremely slow

The original implementation calls `pymxs.runtime.setVert`, `setFace`, `meshop.setMapVert`, `meshop.setMapFace`, and `setNormal` once per item. On a textured mesh with roughly 500,000 vertices, that creates hundreds of thousands to millions of Python -> MAXScript boundary crossings on the 3ds Max main thread.

The optimized path instead:

- creates all geometry with one `mesh(vertices=..., faces=...)` call;
- builds UV face topology in bulk because GLTF indices address POSITION and TEXCOORD attributes together;
- replaces UV vertices in bulk where supported, with a MAXScript-side loop as compatibility fallback;
- assigns explicit normals in a single MAXScript call whose loop executes inside MAXScript;
- disables scene redraw while each mesh is assembled;
- prints timings for geometry, UVs, normals, mesh update, weld, and topology conversion.

## Important: topology conversion

The current UI defaults to **Quads**. GLTF geometry is triangle-based, so this causes `Turn_to_Poly` to scan and reconstruct topology after import. On very large meshes this can still be expensive even after mesh creation is optimized.

For the first benchmark, import the same model twice:

1. **Triangles** topology
2. **Quads** topology

The new timing lines will show whether any remaining delay is in geometry construction or in `Turn_to_Poly`.

## Test procedure

Use the same ~500k-vertex GLTF/GLB that previously required 3-4 minutes. Record:

- total import time;
- `bulk geometry` time;
- `UVs` time;
- `explicit normals` time;
- `topology conversion` time;
- whether 3ds Max remains responsive enough to repaint the importer window between phases;
- whether geometry, UVs, materials, transforms, and normals match the original importer.

This change has been syntax-checked outside 3ds Max, but it still requires an in-application test because `pymxs` and MAXScript mesh behavior can only be validated inside 3ds Max.
