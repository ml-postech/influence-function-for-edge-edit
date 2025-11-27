import torch
from collections import defaultdict
from torch_sparse import SparseTensor
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import k_hop_subgraph
from collections import defaultdict

def edge_rewiring(graph, edge_to_remove, edge_to_insert):
    tmp_graph = graph.clone()
    edge_index = tmp_graph.edge_index
    edges = edge_index.T
    edge_weight = tmp_graph.edge_weight
    
    removed_mask = torch.logical_not((edge_index[:, :, None] == edge_to_remove.T[:, None, :]).all(dim=0).any(dim=1))
    removed_edges = edges[removed_mask]
    removed_edge_weight = edge_weight[removed_mask]

    inserted_edges = torch.cat([removed_edges, edge_to_insert], dim=0)
    inserted_edge_weight = torch.cat([removed_edge_weight, torch.ones((edge_to_insert.shape[0])).to(removed_edge_weight.device)], dim=0)

    tmp_graph.edge_index = inserted_edges.T
    tmp_graph.edge_weight = inserted_edge_weight

    return tmp_graph

def find_nodes_within_k_hop(graph, k):
    edge_index = graph.edge_index
    num_nodes = graph.num_nodes

    edge_with_self_loop = add_self_loops(edge_index,num_nodes=num_nodes)[0]

    # adjacency matrix sparse 형태로 생성
    adj = SparseTensor(row=edge_with_self_loop[0], col=edge_with_self_loop[1], sparse_sizes=(num_nodes, num_nodes))

    # 1-hop 이웃부터 시작
    current_adj = adj

    for _ in range(k - 1):
        current_adj = current_adj @ adj

    # SparseTensor에서 edge 목록 추출
    k_row, k_col, _ = current_adj.coo()

    # set 연산으로 (정확한 k-hop) = (1~k-hop) - (1~k-1-hop)
    k_edges = set(zip(k_row.tolist(), k_col.tolist()))

    # Dictionary 형태로 변환
    k_hop_neighbors = defaultdict(list)
    for src, dst in k_edges:
        k_hop_neighbors[src].append(dst)
    for i in range(graph.num_nodes):
        k_hop_neighbors[i] = torch.tensor(k_hop_neighbors[i]).to(graph.x.device)

    return k_hop_neighbors



def find_k_hop_neighbors_bfs(graph, k):
    k_hop_neighbors = defaultdict(list)

    for node_idx in range(graph.num_nodes):
        # k-hop 서브그래프의 노드 집합 추출
        subset, _, _, _ = k_hop_subgraph(
            node_idx, k, graph.edge_index, relabel_nodes=False
        )
        k_hop_neighbors[node_idx] = subset.to(graph.x.device)

    return k_hop_neighbors

def find_k_hop_neighborhoods(graph, k):
    edge_index = graph.edge_index
    num_nodes = graph.num_nodes

    edge_with_self_loop = add_self_loops(edge_index,num_nodes=num_nodes)[0]

    # adjacency matrix sparse 형태로 생성
    adj = SparseTensor(row=edge_with_self_loop[0], col=edge_with_self_loop[1], sparse_sizes=(num_nodes, num_nodes))

    # 1-hop 이웃부터 시작
    current_adj = adj

    for _ in range(k - 1):
        prev_adj = current_adj.clone()
        current_adj = current_adj @ adj

    # SparseTensor에서 edge 목록 추출
    k_row, k_col, _ = current_adj.coo()
    prev_row, prev_col, _ = prev_adj.coo()

    # set 연산으로 (정확한 k-hop) = (1~k-hop) - (1~k-1-hop)
    k_edges = set(zip(k_row.tolist(), k_col.tolist()))
    prev_edges = set(zip(prev_row.tolist(), prev_col.tolist()))
    exact_k_edges = k_edges - prev_edges

    # Dictionary 형태로 변환
    k_hop_neighbors = defaultdict(list)
    for src, dst in exact_k_edges:
        k_hop_neighbors[src].append(dst)
    for i in range(graph.num_nodes):
        k_hop_neighbors[i] = torch.tensor(k_hop_neighbors[i]).to(graph.x.device)

    return k_hop_neighbors

def remove_edge(graph, removed_edge):
    tmp_graph = graph.clone()
    edges = tmp_graph.edge_index.T
    edge_weights = tmp_graph.edge_weight
    mask = torch.logical_not(torch.logical_or(
        torch.all(edges == removed_edge, dim=1),
        torch.all(edges == torch.flip(removed_edge, dims=[0]), dim=1)
    ))

    tmp_graph.x = graph.x
    tmp_graph.edge_index = edges[mask].T.to(graph.x.device)
    tmp_graph.edge_weight = edge_weights[mask].to(graph.x.device)

    return tmp_graph

def add_edge(graph, added_edge):
    tmp_graph = graph.clone()
    edges = tmp_graph.edge_index.T
    edge_weights = tmp_graph.edge_weight
    
    new_edges = torch.cat((edges, added_edge.unsqueeze(0), added_edge[[1,0]].unsqueeze(0)), dim=0).T
    new_edge_weight = torch.cat((edge_weights, torch.ones((2,)).to(edge_weights.device)), dim=0)

    tmp_graph.x = graph.x
    tmp_graph.edge_index = new_edges
    tmp_graph.edge_weight = new_edge_weight

    return tmp_graph

def add_zero_weight_edge(graph, added_edge):
    tmp_graph = graph.clone()
    edges = tmp_graph.edge_index.T
    edge_weights = tmp_graph.edge_weight.detach()
    
    new_edges = torch.cat((edges, added_edge.unsqueeze(0), added_edge[[1,0]].unsqueeze(0)), dim=0).T
    new_edge_weight = torch.cat((edge_weights, torch.zeros((2,)).to(edge_weights.device)), dim=0)

    tmp_graph.x = graph.x
    tmp_graph.edge_index = new_edges
    tmp_graph.edge_weight = new_edge_weight

    return tmp_graph

def add_zero_weight_edges(graph, added_edges):
    tmp_graph = graph.clone()
    new_edges = tmp_graph.edge_index.T
    new_edge_weights = tmp_graph.edge_weight.detach()
    
    for edge in added_edges:
        new_edges = torch.cat((new_edges, edge.unsqueeze(0), edge[[1,0]].unsqueeze(0)), dim=0)
        new_edge_weights = torch.cat((new_edge_weights, torch.zeros((2,)).to(new_edge_weights.device)), dim=0)

    tmp_graph.x = graph.x
    tmp_graph.edge_index = new_edges.T
    tmp_graph.edge_weight = new_edge_weights

    return tmp_graph


def remove_node(graph, removed_node):
    tmp_graph = graph.clone()
    edges = tmp_graph.edge_index.T
    edge_weights = tmp_graph.edge_weight

    mask = torch.logical_not((graph.edge_index == removed_node).max(dim=0)[0])

    tmp_graph.x = graph.x
    tmp_graph.edge_index = edges[mask].T.to(graph.x.device)
    tmp_graph.edge_weight = edge_weights[mask].to(graph.x.device)

    return tmp_graph