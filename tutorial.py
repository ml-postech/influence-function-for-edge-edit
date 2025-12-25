import torch
from src import GNN, DataLoader, train, eval_model, make_metric_fns
from src.utils import set_seed, random_planetoid_splits, get_edge_removal_candidates, get_edge_insertion_candidates, get_eval_node_idxs
from src.graph_utils import find_k_hop_neighborhoods
from calculate_influence import GraphInfluenceModule
import torch.optim as optim
import os.path as osp

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')

    # Model Arguments
    parser.add_argument('--model', type=str, default='GCN', choices=['SGC', 'GCN'])
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--linear', type=int, default=0)
    parser.add_argument('--bias', type=int, default=0)

    # Learning Arguments
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--weight_decay', type=float, default=0.001)

    # Influence Function Arguments
    parser.add_argument('--hessian_type', type=str, default='GNH', choices=['hessian', 'GNH'])
    parser.add_argument('--damp', type=float, default=0.1)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--lissa_iter', type=int, default=10000)
    parser.add_argument('--pbrf_weight_decay', type=float, default=0.0)
    parser.add_argument('--eval_metric', type=str, default='mean_validation_loss', choices=['dirichlet_energy', 'feature_ablation', 'mean_validation_loss'])
    parser.add_argument("--num_folds", type=int, default=1)
    parser.add_argument("--num_insertion_candidates", type=int, default=100)
    parser.add_argument("--num_removal_candidates", type=int, default=100)
    
    args = parser.parse_args()
    args.linear = bool(args.linear)
    args.bias = bool(args.bias)
    print(args)
    
    # Set up the dataset
    dataset = DataLoader(args.dataset, root='datasets')
    args.num_classes = dataset.num_classes
    data = dataset[0]
    data.edge_weight = torch.ones((data.edge_index.shape[1], ))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.edge_weight = data.edge_weight.to(device)
    data.y = data.y.to(device)

    SEEDS=[1941488137]
    seed = SEEDS[0]

    percls_trn = int(round(0.6*len(data.y)/dataset.num_classes))
    val_lb = int(round(0.2*len(data.y)))
    data = random_planetoid_splits(data, dataset.num_classes, percls_trn, val_lb, seed)
    
    # Set up the model
    set_seed(seed)
    model = GNN(
                name=args.model,
                in_dim=dataset.num_node_features, 
                hidden_dim=args.hidden_dim, 
                num_classes=dataset.num_classes, 
                num_layers=args.num_layers,
                linear=args.linear,
                bias=args.bias
            )

    # Train the model from scratch
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ori_best_val_loss = torch.inf
    for epoch in range(1,args.epochs+1):
        train(data, model, optimizer, device)
        result = eval_model(data, model, device)
        train_loss, val_loss, test_loss, train_acc, val_acc, test_acc = result

        # Save the model with the best validation loss
        if ori_best_val_loss > val_loss:
            ori_best_result = result
            ori_best_val_loss = val_loss
            ori_best_state_dict = {k: v.clone().detach() for k, v in model.state_dict().items()}

        if epoch % 100 == 0:
            print("-----------------------------------------------")
            print(f"Epoch: {epoch}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, test_loss: {test_loss:.4f}")
            print(f"Train acc: {train_acc*100:.2f}%, val acc: {val_acc*100:.2f}%, test_acc: {test_acc*100:.2f}%")
            print("-----------------------------------------------")

    print("-----------------------------------------------")
    print(f"Best results, train loss: {ori_best_result[0]:.4f}, val loss: {ori_best_result[1]:.4f}, test_loss: {ori_best_result[2]:.4f}")
    print(f"Train acc: {ori_best_result[3]*100:.2f}%, val acc: {ori_best_result[4]*100:.2f}%, test_acc: {ori_best_result[5]*100:.2f}%")
    print("-----------------------------------------------")

    model.load_state_dict(ori_best_state_dict)

    set_seed(seed)

    # Get eval node indices for the metric
    eval_node_idxs = get_eval_node_idxs(data, args.eval_metric, seed)

    # Get exact k-hop neighbors if needed
    if args.eval_metric == "feature_ablation":
        exact_k_hop_neighbors = find_k_hop_neighborhoods(data, args.num_layers)
    else:
        exact_k_hop_neighbors = None

    # Create metric functions
    metric_fns = make_metric_fns(eval_node_idxs, exact_k_hop_neighbors, data.edge_index)
    metric_fn = metric_fns[args.eval_metric]

    # Set the edge candidates for removal and insertion
    removal_candidates = get_edge_removal_candidates(data, args.num_removal_candidates)
    insertion_candidates = get_edge_insertion_candidates(data, args.num_insertion_candidates)

    # Reshape candidates to have batch dimension of 1 edge per batch
    removal_candidates = removal_candidates.view(-1, 1, 2)
    insertion_candidates = insertion_candidates.view(-1, 1, 2)

    # Calculate the influence function using GraphInfluenceModule
    print("Calculating influence for edge removal...")
    influence_module = GraphInfluenceModule(model, data, args, args.eval_metric, args.num_folds, eval_node_idxs, metric_fn)
    mvl_removal_inf, removal_retrain_inf, removal_perturb_inf, scale, inv_hvp_norm, avg_influenced = influence_module.calculate_influence(removal_candidates, 'edge_removal')

    print("Calculating influence for edge insertion...")
    mvl_insertion_inf, insertion_retrain_inf, insertion_perturb_inf, scale, inv_hvp_norm, avg_influenced = influence_module.calculate_influence(insertion_candidates, 'edge_insertion')

    print(f"Removal influence shape: {mvl_removal_inf.shape}")
    print(f"Insertion influence shape: {mvl_insertion_inf.shape}")
    print(f"Average removal influence: {mvl_removal_inf.mean():.6f}")
    print(f"Average insertion influence: {mvl_insertion_inf.mean():.6f}")