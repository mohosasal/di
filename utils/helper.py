import dgl

# Placeholder for helper functions (e.g., path validation)
def validate_path(path: list[int], graph: dgl.DGLGraph) -> bool:
    for i in range(len(path) - 1):
        if not graph.has_edges_between(path[i], path[i + 1]):
            return False
    return True