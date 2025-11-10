import torch


def dfs(adjacency: torch.Tensor,
        visited: torch.Tensor,
        in_queue: torch.Tensor,
        start_idx: int,
        queue: list):
    stack = [start_idx]
    while stack:
        node = stack.pop()
        if visited[node]:
            continue
        visited[node] = True
        queue.append(node)

        neighbors = torch.where(adjacency[node] == 1)[0]
        neighbors = neighbors.sort(descending=True)[0]

        for neighbor in neighbors:
            if not visited[neighbor] and not in_queue[neighbor]:
                stack.append(neighbor)
                in_queue[neighbor] = True


def sort_by_node_degree(node_attr: torch.Tensor,
                        node_mask: torch.Tensor,
                        adjacency: torch.Tensor):
    L = adjacency.shape[0]
    device = adjacency.device

    sorted_indices = torch.arange(L, device=device)
    reverse_indices = torch.arange(L, device=device)

    valid_indices = torch.where(node_mask)[0]
    if valid_indices.numel() == 0:
        return sorted_indices, reverse_indices

    degree_vector = adjacency[valid_indices].sum(dim=1)
    atom_types = node_attr[valid_indices, 0]

    sort_keys = torch.stack([-degree_vector, atom_types.float()], dim=1)
    sort_order = torch.argsort(sort_keys, dim=0, stable=True)[:, 0]

    sorted_global = valid_indices[sort_order]
    sorted_indices[:len(sorted_global)] = sorted_global
    reverse_indices[sorted_global] = torch.arange(len(sorted_global), device=device)

    return sorted_indices, reverse_indices


def sort_by_node_degree_dfs(node_attr: torch.Tensor,
                            node_mask: torch.Tensor,
                            adjacency: torch.Tensor):
    L = node_attr.shape[0]
    device = node_attr.device

    sorted_indices = torch.full((L,), -1, dtype=torch.long, device=device)
    reverse_indices = torch.full((L,), -1, dtype=torch.long, device=device)

    valid_indices = torch.where(node_mask)[0]
    if valid_indices.numel() == 0:
        return sorted_indices, reverse_indices

    degree_vector = node_attr[valid_indices, 2]
    atom_ids = valid_indices

    max_degree = degree_vector.max()
    max_degree_indices = atom_ids[degree_vector == max_degree].sort()[0]

    visited = torch.zeros(L, dtype=torch.bool, device=device)
    in_queue = torch.zeros(L, dtype=torch.bool, device=device)
    in_dfs_path = torch.zeros(L, dtype=torch.bool, device=device)

    queue = []
    dfs_path = []

    for idx in max_degree_indices:
        queue.append(idx.item())
        in_queue[idx] = True

    for idx in queue:
        if not visited[idx]:
            dfs(adjacency, visited, in_queue, idx, dfs_path)

    for i in dfs_path:
        in_dfs_path[i] = True

    remaining = torch.arange(L, device=device)[~in_dfs_path]
    full_order = dfs_path + remaining.tolist()

    sorted_tensor = torch.tensor(full_order, dtype=torch.long, device=device)
    sorted_indices[:] = sorted_tensor
    reverse_indices[sorted_tensor] = torch.arange(L, device=device)

    return sorted_indices, reverse_indices


def sort_by_node_degree_dfs_ring(node_attr: torch.Tensor,
                                 node_mask: torch.Tensor,
                                 adjacency: torch.Tensor):
    L = node_attr.shape[0]
    device = node_attr.device

    sorted_indices = torch.full((L,), -1, dtype=torch.long, device=device)
    reverse_indices = torch.full((L,), -1, dtype=torch.long, device=device)

    valid_indices = torch.where(node_mask)[0]
    if valid_indices.numel() == 0:
        return sorted_indices, reverse_indices

    degree_vector = node_attr[valid_indices, 2]
    atom_ids = valid_indices

    max_degree = degree_vector.max()
    max_degree_indices = atom_ids[degree_vector == max_degree].sort()[0]

    ring_atoms = valid_indices[node_attr[valid_indices, 8] == 2]

    initial_queue = torch.cat((max_degree_indices, ring_atoms)).sort()[0]

    visited = torch.zeros(L, dtype=torch.bool, device=device)
    in_queue = torch.zeros(L, dtype=torch.bool, device=device)
    in_dfs_path = torch.zeros(L, dtype=torch.bool, device=device)

    queue = []
    dfs_path = []

    for idx in initial_queue:
        queue.append(idx.item())
        in_queue[idx] = True

    for idx in queue:
        if not visited[idx]:
            dfs(adjacency, visited, in_queue, idx, dfs_path)

    for i in dfs_path:
        in_dfs_path[i] = True

    remaining = torch.arange(L, device=device)[~in_dfs_path]
    full_order = dfs_path + remaining.tolist()

    sorted_tensor = torch.tensor(full_order, dtype=torch.long, device=device)
    sorted_indices[:] = sorted_tensor
    reverse_indices[sorted_tensor] = torch.arange(L, device=device)

    return sorted_indices, reverse_indices
