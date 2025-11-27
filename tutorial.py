import torch
from src import GNN, DataLoader, train, eval_model
from src.utils import set_seed, random_planetoid_splits
from improve_gnns import get_edge_removal_candidates, get_edge_insertion_candidates, get_influence
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
    parser.add_argument("--num_folds", type=int, default=1)
    parser.add_argument("--num_insertion_candidates", type=int, default=10000)
    parser.add_argument("--num_removal_candidates", type=int, default=10000)
    parser.add_argument("--insertion_ratio", type=float, default=0.1)
    parser.add_argument("--removal_ratio", type=float, default=0.1)
    
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
    # Set the edge candidates for removal and insertion
    removal_candidates = get_edge_removal_candidates(data, args.num_removal_candidates)
    insertion_candidates = get_edge_insertion_candidates(data, args.num_insertion_candidates)
    # Calculate the influence function
    checkpoint_dir = osp.join('checkpoints', args.dataset, f"{args.model}_{args.num_layers}_{args.hidden_dim}_{args.linear}_{args.bias}", f"{args.lr}_{args.epochs}_{args.weight_decay}")
    mvl_removal_inf, mvl_insertion_inf = get_influence(model, data, args, checkpoint_dir, seed, device, "mean_validation_loss", num_folds=args.num_folds, insertion_candidates=insertion_candidates, removal_candidates=removal_candidates)
    print(1)