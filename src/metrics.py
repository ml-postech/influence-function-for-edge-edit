import torch
import torch.nn as nn
from torch_geometric.loader import RandomNodeLoader

def make_metric_fns(eval_node_idxs, exact_k_hop, edge_index):
    metric_fns = {
                    'mean_validation_loss': lambda m, g: mean_validation_loss(m, g),
                    'feature_ablation':     lambda m, g: feature_ablation(eval_node_idxs, m, g, exact_k_hop),
                    'dirichlet_energy':     lambda m, g: dirichlet_energy(m, g, edge_index),
                }
    return metric_fns


class NodeRepClass:
    def __init__(self, model, graph, k_hop_neighbors_list):
        self.model = model
        self.graph = graph.clone()
        self.k_hop_neighbors_list = k_hop_neighbors_list

    def get_node_rep(self, node_feature):
        self.graph.x[self.k_hop_nodes] = node_feature
        return self.model(self.graph)[self.node_idx]
    
    def set_node_idx(self, node_idx):
        self.node_idx = node_idx
        self.k_hop_nodes = self.k_hop_neighbors_list[node_idx]


def dirichlet_energy(model, graph, edge_index) -> torch.Tensor:
    """
    Compute the Dirichlet energy of a graph embedding.
    Returns:`
    --------
    energy : torch.Tensor
        The scalar Dirichlet energy of the graph.
    """
    x = model(graph)
    row, col = edge_index  # source (i), target (j)
    
    # Difference between neighboring node embeddings
    diff = x[row] - x[col]
    
    # Squared L2 norm of differences
    diff_squared = (diff ** 2).sum(dim=1)
    
    # Average over all nodes
    num_nodes = x.size(0)
    energy = diff_squared.sum() / num_nodes

    return energy


def pairwise_cosine_similarity(matrix):
    """
    Computes the pairwise cosine similarity between rows of an n x d matrix.

    Args:
        matrix (torch.Tensor): An n x d tensor.

    Returns:
        torch.Tensor: An n x n tensor containing pairwise cosine similarities.
    """
    # Normalize each row to unit length
    norms = torch.norm(matrix, dim=1, keepdim=True)  # (n x 1)
    normalized_matrix = matrix / norms  # (n x d)

    # Compute the pairwise cosine similarity (n x n)
    similarity_matrix = torch.mm(normalized_matrix, normalized_matrix.T)

    return similarity_matrix

def mean_validation_loss(model, graph, val_idxs=None):
    criterion = nn.CrossEntropyLoss()

    model.eval()

    if val_idxs is None:
        val_logit = model(graph)[graph.val_mask]
        val_target = graph.y[graph.val_mask]
    else:
        val_logit = model(graph)[val_idxs]
        val_target = graph.y[val_idxs]

    val_loss = criterion(val_logit, val_target)

    return val_loss

def batch_mean_validation_loss(model, graph):
    criterion = nn.CrossEntropyLoss()

    model.eval()
    graph = graph.clone().to("cpu")
    loader = RandomNodeLoader(graph, num_parts=5, shuffle=False, num_workers=0)

    t_val_loss, t_val_num = 0, 0
    for idx, batch in enumerate(loader):
        if batch.val_mask.sum() == 0:
            continue
        batch = batch.to("cuda")
        b_val_logit = model(batch)[batch.val_mask]
        b_val_target = batch.y[batch.val_mask]

        b_val_loss = criterion(b_val_logit, b_val_target)
        b_val_num = batch.val_mask.sum()

        t_val_loss += b_val_loss * b_val_num
        t_val_num += b_val_num

    mean_val_loss = t_val_loss / t_val_num
    return mean_val_loss

def mean_test_loss(model, graph):
    criterion = nn.CrossEntropyLoss()

    model.eval()
    test_logit = model(graph)[graph.test_mask]
    test_target = graph.y[graph.test_mask]
    test_loss = criterion(test_logit, test_target)

    return test_loss

def feature_eliminate(graph, eliminate_idxs):
    new_graph = graph.clone()
    new_graph.x[eliminate_idxs] = 0

    return new_graph

def feature_ablation(node_idxs, model, graph, k_hop_neighbors):
    # The contribution of node feature to node representation can be obtained as:
    # GNN(G) - GNN(G'), where G' is the copy of G in which target node feature is eleminated.
    origin_node_rep = model(graph)
    result_list = []

    for v in node_idxs:
        if k_hop_neighbors[v].numel() == 0:
            continue
        node_rep = origin_node_rep[v]
        feature_eliminated_graph = feature_eliminate(graph, k_hop_neighbors[v])
        feature_eliminated_rep = model(feature_eliminated_graph)[v]
        feature_contribution = torch.norm(node_rep - feature_eliminated_rep, p='fro')
        result_list.append(feature_contribution)
    result_tensor = torch.stack(result_list)

    return result_tensor.mean()