"""
    Collator for Conformer, with precomputed graph-based sortings.
"""

import torch
from typing import Dict, Sequence
from torch_geometric.data import Data
from rdkit import Chem
from torch_geometric.utils import degree
import numpy as np
import warnings

from ..modules.utils import get_sigma_and_epsilon
from ..modules.collating_utils import get_adjacency, valid_length_to_mask
from .sorting_utils import (
    sort_by_node_degree,
    sort_by_node_degree_dfs,
    sort_by_node_degree_dfs_ring,
)

warnings.simplefilter("ignore", UserWarning)


atomic_symbol_to_number = {
    'H': 0, 'C': 5, 'N': 6, 'O': 7, 'F': 8
}

atomic_number_to_symbol = {
    0: 'H', 5: 'C', 6: 'N', 7: 'O', 8: 'F'
}

bond_type_map = {
    Chem.BondType.SINGLE: 0,
    Chem.BondType.DOUBLE: 1,
    Chem.BondType.TRIPLE: 2,
    Chem.BondType.AROMATIC: 3
}


def get_laplacian_eigenvectors(adjacency: np.ndarray) -> np.ndarray:
    A = adjacency
    l, _ = A.shape
    epsilon = 1e-8
    D = np.diag(1 / np.sqrt(A.sum(axis=1) + epsilon))
    L = np.eye(l) - D @ A @ D
    w, v = np.linalg.eigh(L)
    return v


class Collator:
    def __init__(self, max_nodes: int = None, max_edges: int = None) -> None:
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.keys = None
        self.max_degree = None

    @torch.no_grad()
    def __call__(self, mol_sq: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        keys = mol_sq[0].keys()
        self.keys = keys

        mol_sq = [self._transform(self._get_pyg_data(mol)) for mol in mol_sq]

        num_mol = len(mol_sq)
        num_nodes = [len(mol.node_type) for mol in mol_sq]
        max_nodes = max(num_nodes) if self.max_nodes is None else self.max_nodes

        node_type = torch.zeros((num_mol, max_nodes), dtype=torch.long)
        lap_eigenvectors = torch.zeros((num_mol, max_nodes, max_nodes), dtype=torch.float32)
        adjacency = torch.zeros((num_mol, max_nodes, max_nodes), dtype=torch.float32)
        conformer = torch.zeros((num_mol, max_nodes, 3), dtype=torch.float)
        labels = torch.tensor([mol.labels for mol in mol_sq], dtype=torch.float) if "labels" in keys else None
        node_attr = torch.zeros((num_mol, max_nodes, 9), dtype=torch.long) if "node_attr" in keys else None
        edge_attr = torch.zeros((num_mol, max_nodes, max_nodes, 4), dtype=torch.long)
        num_near_edges = torch.zeros((num_mol, max_nodes), dtype=torch.long)
        sigma = torch.zeros((num_mol, max_nodes, max_nodes), dtype=torch.float32)
        epsilon = torch.zeros((num_mol, max_nodes, max_nodes), dtype=torch.float32)

        for i, mol in enumerate(mol_sq):
            L = num_nodes[i]
            adj = get_adjacency(L, mol.edge_index)
            adjacency[i, :L, :L] = adj
            lap_eigenvectors[i, :L, :L] = torch.from_numpy(get_laplacian_eigenvectors(adj.numpy()))

            if hasattr(mol, "sigma") and hasattr(mol, "epsilon"):
                sigma[i, :L, :L] = mol.sigma
                epsilon[i, :L, :L] = mol.epsilon

            node_type[i, :L] = mol.node_type + 1
            if "conformer" in keys:
                conformer[i, :L] = mol.conformer
            if "node_attr" in keys:
                node_attr[i, :L] = mol.node_attr + 1
            if "num_near_edges" in keys:
                num_near_edges[i, :L] = mol.num_near_edges

        # === 原始 11 个字段 ===
        res_dic = {
            "node_type": node_type,
            "node_mask": valid_length_to_mask(num_nodes, max_nodes),
            "adjacency": adjacency,
            "lap_eigenvectors": lap_eigenvectors,
            "edge_attr": edge_attr,
            "num_near_edges": num_near_edges,
            "sigma": sigma,
            "epsilon": epsilon,
        }
        if labels is not None:
            res_dic["labels"] = labels
        if conformer is not None:
            res_dic["conformer"] = conformer
        if node_attr is not None:
            res_dic["node_attr"] = node_attr

        # === 三种基于图结构的排序 ===
        sorted_indices_all = {f"sorted_indices_{i}": torch.full((num_mol, max_nodes), -1, dtype=torch.long)
                              for i in range(3)}
        reverse_indices_all = {f"reverse_indices_{i}": torch.full((num_mol, max_nodes), -1, dtype=torch.long)
                               for i in range(3)}

        for i in range(num_mol):
            L = num_nodes[i]
            mask_i = res_dic["node_mask"][i, :L]
            node_attr_i = node_attr[i, :L] if node_attr is not None else None
            adj_i = adjacency[i, :L, :L]

            s0, r0 = sort_by_node_degree(node_attr_i, mask_i, adj_i)
            s1, r1 = sort_by_node_degree_dfs(node_attr_i, mask_i, adj_i)
            s2, r2 = sort_by_node_degree_dfs_ring(node_attr_i, mask_i, adj_i)

            sorted_indices_all["sorted_indices_0"][i, :L] = s0
            reverse_indices_all["reverse_indices_0"][i, :L] = r0
            sorted_indices_all["sorted_indices_1"][i, :L] = s1
            reverse_indices_all["reverse_indices_1"][i, :L] = r1
            sorted_indices_all["sorted_indices_2"][i, :L] = s2
            reverse_indices_all["reverse_indices_2"][i, :L] = r2

        res_dic["sorted_indices_dict"] = sorted_indices_all
        res_dic["reverse_indices_dict"] = reverse_indices_all

        return res_dic

    def _get_pyg_data(self, mol: Dict) -> Data:
        return Data(
            node_type=torch.tensor(mol["node_type"], dtype=torch.long),
            edge_index=torch.tensor(mol["edge_index"], dtype=torch.long),
            node_attr=torch.tensor(mol["node_attr"], dtype=torch.long) if "node_attr" in self.keys else None,
            edge_attr=torch.tensor(mol["edge_attr"], dtype=torch.long),
            conformer=torch.tensor(mol["conformer"], dtype=torch.float32),
            labels=torch.tensor(mol["labels"], dtype=torch.float32) if "labels" in self.keys else None,
        )

    def _transform(self, data: Data) -> Data:
        if not data.edge_attr.numel():
            data.edge_type = torch.empty((0,), dtype=torch.long)
        elif data.edge_attr.dim() == 1:
            data.edge_type = data.edge_attr.view(-1)[0]
        else:
            data.edge_type = data.edge_attr[:, 0]

        d = degree(data.edge_index[1], num_nodes=len(data.node_type), dtype=torch.long)
        max_degree = d.max().item()
        num_near_edges = max_degree - d
        data.num_near_edges = num_near_edges

        epsilon, sigma = get_sigma_and_epsilon(data)
        data.epsilon = torch.sqrt((epsilon.unsqueeze(1) * epsilon.unsqueeze(0)))
        data.sigma = ((sigma.unsqueeze(1) + sigma.unsqueeze(0)) / 2)
        return data
