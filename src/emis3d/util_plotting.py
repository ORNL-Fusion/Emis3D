# Util_plotting.py
"""
Contains defintions used with various plotting routines
"""

from raysect.core.math import AffineMatrix3D


def get_to_world(node):
    """
    Compose cumulative local-to-world AffineMatrix3D by walking the parent
    chain.

    Raysect's SceneGraphNode stores only its own local .transform; there is no
    built-in .to_world property. We climb the parent chain collecting
    transforms, then multiply top-down (root-to-leaf) so that the outermost
    transform is applied last:

        world_point = T_root_child * ... * T_parent * T_node * local_point

    The Raysect World root has no .parent attribute at all (unlike Node
    subclasses which have parent=None), so we guard the loop with hasattr().
    """
    transforms = []
    n = node
    while n is not None and hasattr(n, "parent") and n.parent is not None:
        transforms.append(n.transform)
        n = n.parent
    result = AffineMatrix3D()  # identity
    for t in reversed(transforms):  # root-to-leaf order
        result = result * t
    return result


def proj(local_pt, to_world=None):
    """Converts Point3Ds to the world coordinates"""
    if to_world is not None:
        local_pt = local_pt.transform(to_world)
    return local_pt


def draw_Cherab_box(
    ax, camera, colors=["black", "tab:red", "tab:green", "tab:blue"], to_world=False
):
    """
    Draws a box given an input Point3D
    """

    w_ = None
    if to_world:
        w_ = get_to_world(camera)

    box = camera.children[0]
    values = extract_csg_bounds(box)

    for ii, val in enumerate(values):
        v_low = proj(val["lower"], to_world=w_)
        v_up = proj(val["upper"], to_world=w_)

        x = [v_low.x, v_up.x]
        y = [v_low.y, v_up.y]
        z = [v_low.z, v_up.z]

        edges = [
            ([x[0], y[0], z[0]], [x[1], y[0], z[0]]),
            ([x[0], y[1], z[0]], [x[1], y[1], z[0]]),
            ([x[0], y[0], z[1]], [x[1], y[0], z[1]]),
            ([x[0], y[1], z[1]], [x[1], y[1], z[1]]),
            ([x[0], y[0], z[0]], [x[0], y[1], z[0]]),
            ([x[1], y[0], z[0]], [x[1], y[1], z[0]]),
            ([x[0], y[0], z[1]], [x[0], y[1], z[1]]),
            ([x[1], y[0], z[1]], [x[1], y[1], z[1]]),
            ([x[0], y[0], z[0]], [x[0], y[0], z[1]]),
            ([x[1], y[0], z[0]], [x[1], y[0], z[1]]),
            ([x[0], y[1], z[0]], [x[0], y[1], z[1]]),
            ([x[1], y[1], z[0]], [x[1], y[1], z[1]]),
        ]

        l_ = True
        label = val["name"]
        print(label)
        color = "Black"
        if ii < len(colors):
            color = colors[ii]

        for s, e in edges:
            if color is not None:
                ax.plot3D(*zip(s, e), label=label, color=color)
            else:
                ax.plot3D(*zip(s, e))
            if l_:
                l_ = False
                label = "__no_legend__"


def extract_csg_bounds(obj, path="root", results=None, visited=None):
    """
    Recursively traverse a Raysect CSG tree and extract:
        - primitive name
        - lower point
        - upper point
        - tree path

    Returns:
        list of dicts
    """

    if results is None:
        results = []

    if visited is None:
        visited = set()

    # Prevent infinite loops
    if id(obj) in visited:
        return results

    visited.add(id(obj))

    # -------------------------------------------------
    # If this object has bounds (leaf primitive)
    # -------------------------------------------------
    if hasattr(obj, "lower") and hasattr(obj, "upper"):
        results.append(
            {
                "path": path,
                "name": getattr(obj, "name", None),
                "lower": obj.lower,
                "upper": obj.upper,
            }
        )

    # -------------------------------------------------
    # If this is a CSG node (Subtract, Union, Intersect)
    # -------------------------------------------------
    if hasattr(obj, "primitive_a"):
        extract_csg_bounds(
            obj.primitive_a,
            path=f"{path}.primitive_a",
            results=results,
            visited=visited,
        )

    if hasattr(obj, "primitive_b"):
        extract_csg_bounds(
            obj.primitive_b,
            path=f"{path}.primitive_b",
            results=results,
            visited=visited,
        )

    return results
