import torch
import time
import torch.nn as nn
import os.path as osp
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import grad
from tqdm import tqdm
from torch_influence import BaseObjective, LiSSAInfluenceModule
from src import train, train_pbrf, mean_validation_loss, DataLoader, GNN, make_metric_fns
from src.graph_utils import *
from src.utils import *
import argparse


class CrossEntropyObjective(BaseObjective):
    def __init__(self, args):
        self.pbrf_wd = args.pbrf_weight_decay

    def train_outputs(self, model, batch):
        return model(batch)[batch.train_mask]

    def train_loss_on_outputs(self, outputs, batch):
        return F.cross_entropy(outputs, batch.y[batch.train_mask])  # mean reduction required

    def train_regularization(self, params):
        return self.pbrf_wd/2 * torch.square(params.norm())
    
    def train_loss_without_reg(self, model, batch):
        outputs = self.train_outputs(model, batch)
        return self.train_loss_on_outputs(outputs, batch)

    def test_loss(self, model, params, batch):
        val_output = model(batch)[batch.val_mask]
        return F.cross_entropy(val_output, batch.y[batch.val_mask])  # no regularization in test loss
    
    def indiv_train_loss(self, model, params, batch, idx):
        train_output = model(batch)[batch.train_mask]
        train_y = batch.y[batch.train_mask]
        train_loss = F.cross_entropy(train_output[idx], train_y[idx])
        return train_loss + self.train_regularization(params)
    

class GraphInfluenceModule:
    def __init__(self, model, graph, args, eval_metric, num_folds, eval_node_idxs, metric_fn):
        self.model = model
        self.graph = graph
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        self.inv_hvp = None
        self.nodes_within_km1_hop = None
        self.nodes_within_k_hop = None
        self.validation_splits = None
        self.eval_metric = eval_metric
        self.num_folds = num_folds
        self.metric_fn = metric_fn

        self.eval_node_idxs = eval_node_idxs
        self.exact_k_hop_neighbors = self._load_exact_k_hop_neighbors()
        self.get_validation_splits()
    
    def get_validation_splits(self):
        if self.validation_splits is None:
            num_vals = self.graph.val_mask.sum()
            val_idxs = self.graph.val_mask.nonzero().squeeze()
            num_per_split = int(num_vals/self.num_folds)
            shuffled_val_idxs = val_idxs[torch.randperm(num_vals)]

            validation_splits = []
            for i in range(self.num_folds):
                if i == self.num_folds - 1:
                    validation_splits.append(shuffled_val_idxs)
                else:
                    validation_splits.append(shuffled_val_idxs[:num_per_split])
                    shuffled_val_idxs = shuffled_val_idxs[num_per_split:]
                

            self.validation_splits = validation_splits

        return self.validation_splits
        
    def get_retraining_influence(self, targets, influence_type, params):
        """
        target: the target to estimate influence
        influence_type: the type of graph element. Choices: {'edge_removal', 'edge_insertion'}
        """
        origin_logit = self.model(self.graph)

        if influence_type == 'edge_removal':
            perturbed_logit = self.get_perturbed_logit(self.model, self.graph, removed_edge=targets)
        elif influence_type == 'edge_insertion':
            perturbed_logit = self.get_perturbed_logit(self.model, self.graph, added_edge=targets)
        
        influenced_nodes = []
        for target in targets:
            inf_nodes = torch.unique(torch.cat([self.nodes_within_km1_hop[target[0].item()], self.nodes_within_km1_hop[target[1].item()]])).to(torch.long)
            influenced_nodes.append(inf_nodes)
        influenced_nodes = torch.unique(torch.cat(influenced_nodes, dim=-1))
        
        influenced_mask = torch.zeros_like(self.graph.train_mask)
        influenced_mask[influenced_nodes] = 1
        train_influenced_mask = torch.logical_and(influenced_mask, self.graph.train_mask)
        train_influenced_nodes = train_influenced_mask.nonzero().squeeze(1)
        
        if train_influenced_nodes.numel() == 0:
            return [0 for i in range(self.num_folds)], 0
        else:
            origin_indiv_grad = self.get_indiv_grad(origin_logit[train_influenced_nodes], self.graph.y[train_influenced_nodes], params)
            perturbed_grad = self.get_indiv_grad(perturbed_logit[train_influenced_nodes], self.graph.y[train_influenced_nodes], params)

            k_fold_edge_influence = []
            for i in range(self.num_folds):
                edge_influence = 0
                for inv_hvp_elem, origin_indiv_elem, perturbed_elem in zip(self.inv_hvp[i], origin_indiv_grad, perturbed_grad): 
                    elem_influence = inv_hvp_elem * (origin_indiv_elem.sum(dim=0)-perturbed_elem.sum(dim=0))
                    edge_influence += elem_influence.sum()
                edge_influence = edge_influence / self.graph.train_mask.sum()
                k_fold_edge_influence.append(edge_influence.item())

            return k_fold_edge_influence, train_influenced_nodes.numel()
    
    def get_perturbing_influence(self, targets, influence_type):
        """
        target: the target to estimate influence
        influence_type: the type of graph element. Choices: {'edge_removal', 'edge_insertion'}
        """
        if influence_type == 'edge_removal':
            removed_edge_idx = []
            for target in targets:
                _, r_edge_idx = get_edge_weight(self.graph, target)
                removed_edge_idx.append(r_edge_idx)
            removed_edge_idx = torch.cat(removed_edge_idx, dim=-1)

            k_fold_perturb_effect = []
            for i in range(self.num_folds):
                eval_grad = self.weight_grad[i][removed_edge_idx]
                perturb_effect = eval_grad.sum() * -1
                k_fold_perturb_effect.append(perturb_effect.item())
        elif influence_type == 'edge_insertion':
            added_edge_idx = []
            for target in targets:
                _, a_edge_idx = get_edge_weight(self.graph_with_dummy_edges, target)
                added_edge_idx.append(a_edge_idx)
            added_edge_idx = torch.cat(added_edge_idx)
            
            k_fold_perturb_effect = []
            for i in range(self.num_folds):
                eval_grad = self.weight_grad_with_dummy_edges[i][added_edge_idx]
                perturb_effect = eval_grad.sum()
                k_fold_perturb_effect.append(perturb_effect.item())

        return k_fold_perturb_effect

    def calculate_influence(self, candidates, influence_type):
        """
        candidates: list containing the targets to estimate the influence
        influence_type: the type of graph element. Choices: {'edge_removal', 'edge_insertion'}
        """
        self.get_inv_hvp()

        if "edge" in influence_type:
            self.get_nodes_within_km1_hop()
        elif "node" in influence_type:
            self.get_nodes_within_k_hop()
        
        if influence_type in ["edge_insertion"]:
            self.get_weight_grad_with_dummy_edges(candidates.view(-1,2))

        params = [p for p in self.model.parameters() if p.requires_grad]

        total_inf_list = []
        retrain_inf_list = []
        perturb_inf_list = []
        total_num_influenced_nodes = 0

        for target in tqdm(candidates):
            retrain_inf, num_influenced_nodes = self.get_retraining_influence(target, influence_type, params)
            retrain_inf = torch.tensor(retrain_inf)
            total_num_influenced_nodes += num_influenced_nodes
            retrain_inf_list.append(retrain_inf)

            perturb_inf = self.get_perturbing_influence(target, influence_type)
            perturb_inf = torch.tensor(perturb_inf)
            perturb_inf_list.append(perturb_inf)

            total_inf = retrain_inf + perturb_inf
            total_inf_list.append(total_inf)

        retrain_inf_list = torch.stack(retrain_inf_list)
        perturb_inf_list = torch.stack(perturb_inf_list)
        total_inf_list = torch.stack(total_inf_list)
        
        return total_inf_list, retrain_inf_list, perturb_inf_list, self.module.scale, self.inv_hvp_norm, num_influenced_nodes/candidates.shape[0]
    
    def _load_exact_k_hop_neighbors(self):
        if self.eval_metric == 'feature_ablation':
            return find_k_hop_neighborhoods(self.graph, self.args.num_layers)
        else:
            return None

    def _create_lissa_module(self):
        return LiSSAInfluenceModule(
            graph=self.graph,
            model=self.model,
            objective=CrossEntropyObjective(self.args),
            train_loader=None,
            test_loader=None,
            device=self.device,
            damp=self.args.damp,
            repeat=1,
            lissa_iter = self.args.lissa_iter,
            scale=self.args.scale,
            depth=None,
            gnh=True if self.args.hessian_type=='GNH' else False,
            full_batch=True
        )

    def get_inv_hvp(self):
        if self.inv_hvp is None:
            self.module = self._create_lissa_module()
            eval_result, weight_grad, inv_hvp, inv_hvp_norm = self.approximate_inv_hvp(
                self.model, self.graph, self.module, self.eval_metric, self.num_folds, self.validation_splits
            )

            params = [p for p in self.model.parameters() if p.requires_grad]
            
            reshaped_inv_hvp = []
            for i in range(self.num_folds):
                reshaped_inv_hvp.append(reshape_like_params(inv_hvp[i], params))
            self.inv_hvp = reshaped_inv_hvp
            self.weight_grad = weight_grad
            self.inv_hvp_norm = inv_hvp_norm
    
    def get_nodes_within_k_hop(self):
        if self.nodes_within_k_hop is None:
            self.nodes_within_k_hop = find_nodes_within_k_hop(self.graph, self.args.num_layers)
    
    def get_nodes_within_km1_hop(self):
        if self.args.dataset == "Squirrel":
            # To do: Integrate across all datasets.
            self.nodes_within_km1_hop = find_k_hop_neighbors_bfs(self.graph, self.args.num_layers-1)
        if self.nodes_within_km1_hop is None:
            self.nodes_within_km1_hop = find_nodes_within_k_hop(self.graph, self.args.num_layers-1)

    def get_weight_grad_with_dummy_edges(self, insertion_candidates):
        self.graph_with_dummy_edges = add_zero_weight_edges(self.graph, insertion_candidates)
        self.graph_with_dummy_edges.edge_weight.requires_grad = True
        
        weight_grads = []
        if self.eval_metric == "mean_validation_loss":
            
            for i in range(self.num_folds):
                valid_idxs = self.validation_splits[i]
                eval_result = mean_validation_loss(self.model, self.graph_with_dummy_edges, valid_idxs)
                weight_grad = grad(eval_result, self.graph_with_dummy_edges.edge_weight)[0]
                weight_grads.append(weight_grad)
        else:
            eval_result = self.get_eval_result(self.model, self.graph_with_dummy_edges)
            weight_grad = grad(eval_result, self.graph_with_dummy_edges.edge_weight)[0]
            weight_grads.append(weight_grad)
        
        self.weight_grad_with_dummy_edges = weight_grads
    
    def get_perturbed_logit(self, model, graph, removed_edge=None, removed_node=None, added_edge=None):
        perturbed_graph = graph.clone()
        if removed_edge is not None:
            for edge in removed_edge:
                perturbed_graph = remove_edge(perturbed_graph, edge)
            perturbed_logit = model(perturbed_graph)
        elif added_edge is not None:
            for edge in added_edge:
                perturbed_graph = add_edge(perturbed_graph, edge)
            perturbed_logit = model(perturbed_graph)
        
        return perturbed_logit

    def get_eval_result(self, model, graph):
        graph.edge_weight.requires_grad = True
        eval_result = self.metric_fn(model, graph)

        return eval_result

    def approximate_inv_hvp(self, model, graph, module, eval_metric, num_folds, validation_splits):
        eval_results = []
        weight_grads = []
        inv_hvps = []
        inv_hvp_norms = []
        if eval_metric == 'mean_validation_loss':
            graph.edge_weight.requires_grad = True
            
            for i in range(num_folds):
                params = list(model.parameters())
                valid_idxs = validation_splits[i]
                eval_result = mean_validation_loss(model, graph, valid_idxs)
                param_grad = grad(eval_result, params, retain_graph=True)
                flatten_vec = flatten_params_like(param_grad, params)
                weight_grad = grad(eval_result, graph.edge_weight)[0]
                inv_hvp, inv_hvp_norm = module.stest(grad_eval=flatten_vec)

                eval_results.append(eval_result)
                weight_grads.append(weight_grad)
                inv_hvps.append(inv_hvp)
                inv_hvp_norms.append(inv_hvp_norm)
        elif eval_metric in ['feature_ablation','dirichlet_energy']:
            graph.edge_weight.requires_grad = True
            params = list(model.parameters())
            eval_result = metric_fn(model, graph)
            param_grad = grad(eval_result, params, retain_graph=True)
            flatten_vec = flatten_params_like(param_grad, params)
            weight_grad = grad(eval_result, graph.edge_weight)[0]
            inv_hvp, inv_hvp_norm = module.stest(grad_eval=flatten_vec)

            eval_results.append(eval_result)
            weight_grads.append(weight_grad)
            inv_hvps.append(inv_hvp)
            inv_hvp_norms.append(inv_hvp_norm)
        else:
            raise ValueError
        
        return eval_results, weight_grads, inv_hvps, inv_hvp_norms

    def get_indiv_grad(self, logits, targets, params):
        criterion = nn.CrossEntropyLoss()
        results = [[] for _ in range(len(params))]

        for i in range(targets.numel()):
            indiv_loss = criterion(logits[i], targets[i])
            indiv_grad = grad(indiv_loss, params, retain_graph=True)

            indiv_grad_detached = [g.detach() for g in indiv_grad]

            for j, paramwise_grad in enumerate(indiv_grad_detached):
                results[j].append(paramwise_grad)

        tensor_results = []
        for result in results:
            tensor_results.append(torch.stack(result))

        return tensor_results

def calculate_loo(model, graph, candidate_edges, args, seed, model_save_dir, metric_fn, element_type):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    evaluation_result = metric_fn(model, graph)

    loo_results = []
    for candidate_edge in tqdm(candidate_edges):
        if element_type == 'edge_removal':
            perturbed_graph = remove_edge(graph, candidate_edge)
            perturbed_graph.edge_weight = perturbed_graph.edge_weight.detach()
        elif element_type == 'edge_insertion':
            perturbed_graph = add_edge(graph, candidate_edge)
            perturbed_graph.edge_weight = perturbed_graph.edge_weight.detach()
        else:
            raise ValueError

        set_seed(seed)
        new_model = GNN(
                name=args.model,
                in_dim=dataset.num_node_features, 
                hidden_dim=args.hidden_dim, 
                num_classes=dataset.num_classes, 
                num_layers=args.num_layers,
                linear=args.linear,
                bias=args.bias
            )
        
        edge_perturb_model_path = osp.join(model_save_dir, f'{candidate_edge[0]}_{candidate_edge[1]}.pth')
        if osp.isfile(edge_perturb_model_path):
            edge_perturb_state_dict = torch.load(edge_perturb_model_path, weights_only=True)
            new_model.load_state_dict(edge_perturb_state_dict)
            new_model = new_model.to(device)
        else:
            new_model = new_model.to(device)
            new_optimizer = optim.SGD(new_model.parameters(), lr=args.lr, weight_decay=args.damp)

            new_model.train()
            for _ in range(args.epochs):
                train_loss, _, _, _, _, _ = train(perturbed_graph, new_model, new_optimizer, device)
            torch.save(new_model.state_dict(), edge_perturb_model_path)

        new_model.eval()
        perturbed_result = metric_fn(new_model, perturbed_graph)

        loo_result = perturbed_result-evaluation_result
        loo_results.append(loo_result.item())

    return loo_results

def calculate_pbrf(model, graph, candidate_edges, args, seed, model_dir, metric_fn, element_type):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_result = metric_fn(model, graph)

    km1_hop_neighbors = find_k_hop_neighbors_bfs(graph, args.num_layers-1)
    y_s = model(graph)
    theta_s = flatten_parameters(model).detach()
    loss_func = nn.CrossEntropyLoss()

    train_y_s = y_s[graph.train_mask]
    train_target = graph.y[graph.train_mask]

    bregman_grad = grad(loss_func(train_y_s, train_target), train_y_s)[0]
    y_s = y_s.detach()

    pbrf_results = []
    pbrf_nip_results = []
    pbrf_nrt_results = []
    for edge_idx, candidate_edge in enumerate(tqdm(candidate_edges)):
        data.x.requires_grad = False
        data.edge_weight.requires_grad = False

        if candidate_edge.dim() == 2:
            influenced_nodes = []
            for target in candidate_edge:
                i_nodes = torch.unique(torch.cat([km1_hop_neighbors[target[0].item()], km1_hop_neighbors[target[1].item()]])).to(torch.long)
                influenced_nodes.append(i_nodes)
            influenced_nodes = torch.unique(torch.cat(influenced_nodes, dim=-1))
        else:
            influenced_nodes = torch.unique(torch.cat([km1_hop_neighbors[candidate_edge[0].item()], km1_hop_neighbors[candidate_edge[1].item()]])).to(torch.long)
        influenced_mask = torch.zeros_like(graph.train_mask)
        influenced_mask[influenced_nodes] = 1
        train_influenced_mask = torch.logical_and(influenced_mask, graph.train_mask)
        train_influenced_nodes = train_influenced_mask.nonzero().squeeze(1)

        if element_type == 'edge_removal':
            perturbed_graph = graph.clone()
            for edge in candidate_edge:
                perturbed_graph = remove_edge(perturbed_graph, edge)
        elif element_type == 'edge_insertion':
            perturbed_graph = graph.clone()
            for edge in candidate_edge:
                perturbed_graph = add_edge(perturbed_graph, edge)
        else:
            raise ValueError
        
        if train_influenced_nodes.numel() == 0:
            model.eval()
            perturbed_result = metric_fn(model, perturbed_graph)
            perturbed_result_nip = metric_fn(model, graph)
            perturbed_result_nrt = metric_fn(model, perturbed_graph)
        else:
            set_seed(seed)
            new_model = GNN(
                    name=args.model,
                    in_dim=dataset.num_node_features, 
                    hidden_dim=args.hidden_dim, 
                    num_classes=dataset.num_classes, 
                    num_layers=args.num_layers,
                    linear=args.linear,
                    bias=args.bias,
                    num_heads=args.num_heads
                )
            
            edges_name = ''
            for edge in candidate_edge:
                edge_name = f'{edge[0]}_{edge[1]}_'
                edges_name += edge_name
            edges_name = edges_name[:-1] + '.pth'
            edge_perturb_model_path = osp.join(model_dir, edges_name)
            if osp.isfile(edge_perturb_model_path):
                edge_perturb_state_dict = torch.load(edge_perturb_model_path, weights_only=True)
                new_model.load_state_dict(edge_perturb_state_dict)
                new_model = new_model.to(device)
            else:
                new_model.load_state_dict(model.state_dict())
                new_model = new_model.to(device)

                new_optimizer = optim.SGD(new_model.parameters(), lr=args.lr, weight_decay=args.pbrf_weight_decay)

                new_model.train()
                for epoch in range(args.pbrf_epochs):
                    train_loss, remove_loss, add_loss, train_acc, val_acc, test_acc = train_pbrf(train_influenced_nodes, graph, perturbed_graph, new_model, new_optimizer, device, y_s, theta_s, bregman_grad, args)

                torch.save(new_model.state_dict(), edge_perturb_model_path)

            new_model.eval()
            perturbed_result     = metric_fn(new_model, perturbed_graph)
            perturbed_result_nip = metric_fn(new_model, graph)
            perturbed_result_nrt = metric_fn(model, perturbed_graph)

        pbrf_result = perturbed_result-eval_result
        pbrf_results.append(pbrf_result.item())
        pbrf_nip_results.append((perturbed_result_nip-eval_result).item())
        pbrf_nrt_results.append((perturbed_result_nrt-eval_result).item())

    return pbrf_results, pbrf_nip_results, pbrf_nrt_results


def get_pbrf(args, model, data, candidate_edges, seed, dirs, element_type):
    print('Calculate PBRF...')
    start_time = time.time()
    edge_pbrf, act_nip, act_nrt = calculate_pbrf(model, data, candidate_edges, args, seed, dirs["pbrf_model"], metric_fn, element_type)
    print(f'Consumed time: {time.time()-start_time:.2f}s')

    return edge_pbrf, act_nip, act_nrt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora_public')
    parser.add_argument('--model', type=str, default='GCN', choices=['SGC', 'GCN', 'GAT', 'ChebNet'])
    parser.add_argument('--hessian_type', type=str, default='GNH', choices=['hessian', 'GNH'])
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--damp', type=float, default=0.1)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--lissa_iter', type=int, default=10000)
    parser.add_argument('--eval_metric', type=str, default='mean_validation_loss', choices=['dirichlet_energy', 'feature_ablation', 'mean_validation_loss'])
    parser.add_argument('--linear', type=int, default=0)
    parser.add_argument('--bias', type=int, default=0)
    parser.add_argument('--pbrf_epochs', type=int, default=1000)
    parser.add_argument('--pbrf_weight_decay', type=float, default=0.0)
    parser.add_argument("--element_type", type=str, default='edge_edit', choices=['edge_removal', 'edge_insertion', 'edge_edit'])
    parser.add_argument("--num_insertion_candidates", type=int, default=50)
    parser.add_argument("--num_removal_candidates", type=int, default=50)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--check_runtime", type=int, default=0)
    parser.add_argument("--json_config", type=str, default="none")
    parser.add_argument("--fig_title", type=str, default="none")
    parser.add_argument("--num_group_elem", type=int, default=1)

    args = parser.parse_args()
    args.linear = bool(args.linear)
    args.bias = bool(args.bias)
    print(args)

    dirs = make_dirs(args)
    save_config(args, osp.join(dirs['result'], 'config.json'), dirs)

    if args.json_config != "none":
        import json
        from types import SimpleNamespace

        json_config = args.json_config
        with open(args.json_config, 'r') as f:
            args = json.load(f)
        args = SimpleNamespace(**args)     
        args.json_config = json_config 
        if "fig_title" not in vars(args).keys():  
            args.fig_title = args.eval_metric
        print(args)

    WD = args.weight_decay
    PBRF_WD = args.pbrf_weight_decay
    if args.hessian_type == 'hessian':
        print('Warning. args.damp should be the same with args.weight_decay when args.hessian_type is hessian.')
        print(f'Original damp: {args.damp}, adjusted damp: {args.weight_decay}')
        args.damp = args.weight_decay

    dataset = DataLoader(args.dataset, root='datasets')
    args.num_classes = dataset.num_classes
    data = dataset[0]
    data.edge_weight = torch.ones((data.edge_index.shape[1], ))

    SEEDS=[1941488137,4198936517,983997847,4023022221,4019585660,2108550661,1648766618,629014539,3212139042,2424918363]
    seed = SEEDS[0]

    vanilla_dir = dirs["vanilla"]
    vanilla_path = osp.join(vanilla_dir, f"{seed}.pth")
    
    eval_node_idxs = get_eval_node_idxs(data, args.eval_metric, seed)

    if 'public' not in args.dataset:
        percls_trn = int(round(0.6*len(data.y)/dataset.num_classes))
        val_lb = int(round(0.2*len(data.y)))
        data = random_planetoid_splits(data, dataset.num_classes, percls_trn, val_lb, seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.edge_weight = data.edge_weight.to(device)
    data.y = data.y.to(device)

    set_seed(seed)
    model = GNN(
                name=args.model,
                in_dim=dataset.num_node_features, 
                hidden_dim=args.hidden_dim, 
                num_classes=dataset.num_classes, 
                num_layers=args.num_layers,
                linear=args.linear,
                bias=args.bias,
                num_heads=args.num_heads
            )
    if osp.isfile(vanilla_path):
        model_state_dict = torch.load(vanilla_path, weights_only=True)
        model.load_state_dict(model_state_dict)
        model = model.to(device)
    else:
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(1,args.epochs+1):
            train_loss, val_loss, test_loss, train_acc, val_acc, test_acc = train(data, model, optimizer, device)
            if epoch % 100 == 0:
                print("-----------------------------------------------")
                print(f"Epoch: {epoch}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, test_loss: {test_loss:.4f}")
                print(f"Train acc: {train_acc*100:.2f}%, val acc: {val_acc*100:.2f}%, test_acc: {test_acc*100:.2f}%")
                print("-----------------------------------------------")
        torch.save({k: v.clone().detach() for k, v in model.state_dict().items()}, vanilla_path)

    save_name = f'influence_vs_pbrf'
    save_name_nip = f'retraining_effect'
    save_name_nrt = f'perturbing_effect'
    save_name_rtpt = f'retraining_vs_perturbing'

    set_seed(seed)
    if args.eval_metric == "feature_ablation":
        exact_k_hop_neighbors = find_k_hop_neighborhoods(data, args.num_layers)
    else:
        exact_k_hop_neighbors = None

    metric_fns = make_metric_fns(eval_node_idxs, exact_k_hop_neighbors, data.edge_index)
    metric_fn = metric_fns[args.eval_metric]

    if args.element_type in ['edge_removal', 'edge_edit']:
        set_seed(seed)
        num_candidates = args.num_removal_candidates * args.num_group_elem
        candidates = get_edge_removal_candidates(data, num_candidates)
        candidates = candidates.view(args.num_removal_candidates, args.num_group_elem, 2)

        num_removal_candidates = num_candidates
        removal_candidates = candidates
    if args.element_type in ['edge_insertion', 'edge_edit']:
        set_seed(seed)
        num_candidates = args.num_insertion_candidates * args.num_group_elem
        candidates = get_edge_insertion_candidates(data, num_candidates*2)[:num_candidates]
        candidates = candidates.view(args.num_insertion_candidates, args.num_group_elem, 2)

        num_insertion_candidates = num_candidates
        insertion_candidates = candidates
        
    print(f'Calculate the Influence of {args.element_type}...')
    start_time = time.time()
    influence_module = GraphInfluenceModule(model, data, args, args.eval_metric, 1, eval_node_idxs, metric_fn)
    if args.element_type == 'edge_edit':
        r_total_inf, r_retrain_inf, r_perturb_inf, module_scale, inv_hvp_norm, num_ins = influence_module.calculate_influence(removal_candidates, 'edge_removal')
        i_total_inf, i_retrain_inf, i_perturb_inf, module_scale, inv_hvp_norm, num_ins = influence_module.calculate_influence(insertion_candidates, 'edge_insertion')
        
        total_inf = torch.cat((r_total_inf, i_total_inf), dim=0)
        retrain_inf = torch.cat((r_retrain_inf, i_retrain_inf), dim=0)
        perturb_inf = torch.cat((r_perturb_inf, i_perturb_inf), dim=0)
    else:
        total_inf, retrain_inf, perturb_inf, module_scale, inv_hvp_norm, num_ins = influence_module.calculate_influence(candidates, args.element_type)
    print(f'Consumed time: {time.time()-start_time:.2f}s')

    if args.hessian_type == 'hessian':
        loo = calculate_loo(model, data, candidates, args, seed, dirs['loo_model'], metric_fn, args.element_type)

        mask = torch.logical_and(is_within_2std(retrain_inf.squeeze()), is_within_2std(torch.tensor(loo)))
        plot_influence_loss(retrain_inf.squeeze()[mask], torch.tensor(loo)[mask], dirs['result'], save_name_nip, args, title=dir['fig_title'])

    elif args.hessian_type == 'GNH':
        if args.element_type == "edge_edit":
            r_total_pbrf, r_retrain_pbrf, r_perturb_pbrf = get_pbrf(args, model, data, removal_candidates, seed, dirs, 'edge_removal')
            i_total_pbrf, i_retrain_pbrf, i_perturb_pbrf = get_pbrf(args, model, data, insertion_candidates, seed, dirs, 'edge_insertion')
            
            total_pbrf = r_total_pbrf + i_total_pbrf
            retrain_pbrf = r_retrain_pbrf + i_retrain_pbrf
            perturb_pbrf = r_perturb_pbrf + i_perturb_pbrf
            r_size = len(r_total_pbrf)
        else:
            total_pbrf, retrain_pbrf, perturb_pbrf = get_pbrf(args, model, data, candidates, seed, dirs, args.element_type)
            r_size = None
        
        rename_result_dir(args, retrain_inf, retrain_pbrf, perturb_inf, perturb_pbrf, dirs)
        k=2
        mask = torch.logical_and(is_within_2std(total_inf.squeeze(),k), is_within_2std(torch.tensor(total_pbrf),k))
        plot_influence_loss(total_inf.squeeze()[mask], torch.tensor(total_pbrf)[mask], dirs['result'], save_name, args, title=dirs['fig_title'], mask=mask, r_size=r_size)
        plot_influence_loss(retrain_inf.squeeze()[mask], perturb_inf.squeeze()[mask], dirs['result'], save_name_rtpt, args, xlabel="Parameter Shift Effect", ylabel="Propagation Effect", title=dirs['fig_title'], mask=mask, r_size=r_size)