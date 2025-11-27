import sys
import torch
import numpy as np
from matplotlib import pyplot as plt
import os
from os import path as osp
import json
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator
import shutil

def get_eval_node_idxs(graph, eval_metric, seed):
    set_seed(seed)
    if eval_metric in ['feature_ablation']:
        eval_node_idxs = torch.randperm(graph.x.shape[0])[:50].tolist()
    else:
        eval_node_idxs = [i for i in range(graph.x.shape[0])]

    return eval_node_idxs


def lower_is_better(args):
    if args.eval_metric in ['mean_validation_loss']:
        return True
    elif args.eval_metric in ['feature_ablation', 'dirichlet_energy']:
        return False
    else:
        raise ValueError


def get_edge_removal_candidates(graph, num_candidates):
    # Set all edges as removal candidates
    if graph.edge_index.numel()/4 > num_candidates:
        edges = graph.edge_index.T
        sorted_edges = torch.sort(edges, dim=1)[0]
        unique_edges = torch.unique(sorted_edges, dim=0)
        random_idxs = torch.randperm(unique_edges.shape[0])[:num_candidates]
        unique_edges = unique_edges[random_idxs]
    else:
        edges = graph.edge_index.T
        sorted_edges = torch.sort(edges, dim=1)[0]
        unique_edges = torch.unique(sorted_edges, dim=0)

    return unique_edges


def get_edge_insertion_candidates(graph, num_candidates):
    # Set {num_candidates} random edges as insertion candidates
    device = "cuda" if torch.cuda.is_available() else "cpu"

    candidate_nodes = torch.arange(0, graph.num_nodes).to(device)
    all_edges = torch.combinations(candidate_nodes, r=2)
    random_idxs = torch.randperm(all_edges.shape[0])[:num_candidates]
    candidate_edges = all_edges[random_idxs].T

    if candidate_edges.numel() == 0:
        return candidate_edges
    
    existing_edges = graph.edge_index

    existing_edges = existing_edges.sort(dim=0)[0]
    candidate_edges = candidate_edges.sort(dim=0)[0]

    mask = (candidate_edges[:, :, None] == existing_edges[:, None, :]).all(dim=0).any(dim=1)
    
    edge_insertion_candidates = candidate_edges[:, ~mask]

    return edge_insertion_candidates.T


def save_config(args, filename, dirs):
    if args.json_config == 'none':
        with open(filename, 'w') as f:
            json.dump(vars(args), f, indent=4)

def get_save_id(save_dir):
    if os.path.exists(save_dir):
        file_list = os.listdir(save_dir)
        if len(file_list) == 0:
            return 0

        ids = []
        for file_name in file_list:
            if file_name == 'pbrf_checkpoints':
                continue
            ids.append(int(file_name.split('_')[-1]))

        return max(ids) + 1
    else:
        raise ValueError

def get_edge_weight(graph, edge=None, node=None):
    if edge is not None and node is not None:
        raise ValueError
    
    if edge is not None:
        edge_index = graph.edge_index
        undirected_edges = torch.stack([edge, torch.tensor([edge[1].item(), edge[0].item()]).to(edge.device)])
        edge_idx = torch.all(torch.isin(edge_index.T, undirected_edges), dim=1).nonzero().squeeze()

        return graph.edge_weight[edge_idx], edge_idx
    elif node is not None:
        tmp_graph = graph.clone()
        edges = tmp_graph.edge_index.T
        edge_weights = tmp_graph.edge_weight

        mask = (graph.edge_index == node).max(dim=0)[0]

        return None, mask
    else:
        raise ValueError

def add_gradients(grad1, grad2):
    res = []
    if grad1 is None:
        return grad2
    else:
        for grad1_elem, grad2_elem in zip(grad1, grad2):
            if grad1_elem is None and grad2_elem is None:
                res.append(None)
            else:
                res.append(grad1_elem + grad2_elem)
        return res
    
def scale_gradients(grad1, scale):
    res = []
    for grad_elem in grad1:
        if grad_elem is None:
            res.append(None)
        else:
            res.append(grad_elem * scale)

    return res


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def reshape_like_params(vec, params):
    pointer = 0
    split_tensors = []
    params_shape = tuple(p.shape for p in params)
    for dim in params_shape:
        num_param = dim.numel()
        split_tensors.append(vec[pointer: pointer + num_param].view(dim))
        pointer += num_param
    return tuple(split_tensors)


def flatten_parameters(model):
    flatten_params = []
    for p in model.parameters():
        flatten_params.append(p.view(-1))
    return torch.cat(flatten_params)

def flatten_params_like(params_like, param):
    vec = []
    for idx, p in enumerate(params_like):
        if p is None:
            vec.append(torch.zeros_like(param[idx]).view(-1))
        else:
            vec.append(p.view(-1))
    return torch.cat(vec)

def is_within_2std(x: torch.Tensor, k: float=2) -> torch.Tensor:
    """
    Check if each value in the tensor is within mean ± 2 * std.

    Parameters:
    -----------
    x : torch.Tensor
        Input tensor of any shape.

    Returns:
    --------
    torch.Tensor
        Boolean tensor indicating whether each element is within the range.
    """
    mean = x.mean()
    std = x.std(unbiased=False)  # Set to False for population std
    lower = mean - k * std
    upper = mean + k * std
    return (x >= lower) & (x <= upper)

class FixedSciFormatter(ScalarFormatter):
    def _set_format(self):
        self.format = '%1.1f'

def plot_influence_loss(influence, loo, save_dir, save_name, args, margin_rate=0.1, xlabel="Estimated Influence", ylabel="Actual Influence", title=None, mask=None, r_size=None):
    os.makedirs(save_dir,exist_ok=True)
    plt.figure(figsize=(8, 6))

    if args.element_type == 'edge_edit':
        r_num = mask[:r_size].sum().item()
        plt.scatter(influence[:r_num], loo[:r_num], color='red', alpha=0.7, s=50, marker='x', linewidths=4, label='Deletion')
        plt.scatter(influence[r_num:], loo[r_num:], color='blue', alpha=0.7, s=50, marker='o', label='Insertion')
        handles, labels = plt.gca().get_legend_handles_labels()
        #legend = plt.legend(handles, labels, fontsize=23, loc='best', framealpha=0.6, markerscale=2.0)
    else:
        #plt.scatter(influence, loo, color='red', alpha=0.7, s=50, marker='x', linewidths=4)
        plt.scatter(influence, loo, color='blue', alpha=0.7, s=50)
    
    correlation = torch.corrcoef(torch.stack((influence, torch.tensor(loo))))[0, 1]

    plt.xlabel(xlabel, fontsize=26)
    plt.ylabel(ylabel, fontsize=26)
    
    if title is not None:
        if title == "mean_validation_loss":
            plt.title("Validation Loss", fontsize=30)
        elif title == "feature_ablation":
            plt.title("Over-squashing", fontsize=30)
        elif title == "dirichlet_energy":
            plt.title("Dirichlet Energy", fontsize=30)
        elif title == "Citeseer":
            plt.title("CiteSeer", fontsize=30)
        elif title == "Pubmed":
            plt.title("PubMed", fontsize=30)
        else:
            plt.title(title, fontsize=30)

    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    ax = plt.gca()

    x_formatter = ScalarFormatter(useMathText=True)
    x_formatter.set_scientific(True)
    x_formatter.set_powerlimits((0, 0))
    x_formatter.set_useOffset(True)
    ax.xaxis.set_major_formatter(x_formatter)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.offsetText.set_fontsize(24)

    y_formatter = FixedSciFormatter(useMathText=True)
    y_formatter.set_scientific(True)
    y_formatter.set_powerlimits((0, 0))
    y_formatter.set_useOffset(True)
    ax.yaxis.set_major_formatter(y_formatter)
    ax.yaxis.offsetText.set_fontsize(24)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    # Display the plot
    plt.grid(alpha=0.3)
    min_val, max_val = min(min(influence), min(loo)), max(max(influence), max(loo))
    margin = (max_val - min_val)  * margin_rate

    ax.text(0.98, 0.02, f"Correlation: {correlation:.2f}",
        transform=ax.transAxes,  
        fontsize=22, ha='right', va='bottom',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.plot([min_val-margin, max_val+margin], [min_val-margin, max_val+margin], color='red', linestyle='--', linewidth=5)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{save_name}.png', bbox_inches='tight')
    plt.savefig(f'{save_dir}/{save_name}.pdf', bbox_inches='tight')
    plt.clf()


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, seed=12134):
    index=[i for i in range(0,data.y.shape[0])]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]
    #print(test_idx)

    data.train_mask = index_to_mask(train_idx,size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx,size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx,size=data.num_nodes)
    
    return data

def display_progress(text, current_step, last_step, enabled=True,
                     fix_zero_start=True):
    """Draws a progress indicator on the screen with the text preceeding the
    progress

    Arguments:
        test: str, text displayed to describe the task being executed
        current_step: int, current step of the iteration
        last_step: int, last possible step of the iteration
        enabled: bool, if false this function will not execute. This is
            for running silently without stdout output.
        fix_zero_start: bool, if true adds 1 to each current step so that the
            display starts at 1 instead of 0, which it would for most loops
            otherwise.
    """
    if not enabled:
        return

    # Fix display for most loops which start with 0, otherwise looks weird
    if fix_zero_start:
        current_step = current_step + 1

    term_line_len = 80
    final_chars = [':', ';', ' ', '.', ',']
    if text[-1:] not in final_chars:
        text = text + ' '
    if len(text) < term_line_len:
        bar_len = term_line_len - (len(text)
                                   + len(str(current_step))
                                   + len(str(last_step))
                                   + len("  / "))
    else:
        bar_len = 30
    filled_len = int(round(bar_len * current_step / float(last_step)))
    bar = '=' * filled_len + '.' * (bar_len - filled_len)

    bar = f"{text}[{bar:s}] {current_step:d} / {last_step:d}"
    if current_step < last_step-1:
        # Erase to end of line and print
        sys.stdout.write("\033[K" + bar + "\r")
    else:
        sys.stdout.write(bar + "\n")

    sys.stdout.flush()


def make_dirs(args):
    model_hparams = osp.join(args.model, f'linear_{args.linear}_bias_{args.bias}', args.dataset, f'layer_{args.num_layers}')
    learning_hparams = f"{args.lr}_{args.hidden_dim}_{args.epochs}_{args.weight_decay}"
    calculate_hparams = osp.join(args.element_type, f'{args.num_group_elem}edges')

    vanilla_model_dir = osp.join('checkpoints', "vanilla", model_hparams, learning_hparams)

    result_root = osp.join('results', args.hessian_type, args.eval_metric, model_hparams, calculate_hparams)
    os.makedirs(result_root, exist_ok=True)
    
    loo_model_root = osp.join('checkpoints', 'loo_checkpoints', model_hparams, calculate_hparams)
    loo_model_dir = osp.join(loo_model_root, learning_hparams)
    
    pbrf_model_root = osp.join('checkpoints', 'pbrf_checkpoints', model_hparams, calculate_hparams)
    pbrf_model_dir = osp.join(pbrf_model_root, learning_hparams, f'{args.damp}_{args.pbrf_epochs}_{args.pbrf_weight_decay}')
    
    if args.json_config != "none":
        result_id = None
        result_dir = osp.join("configs", "results", f"{args.json_config[:-5]}")
    else:
        result_id = get_save_id(result_root)
        result_dir = f'{result_root}/{result_id}'

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(loo_model_dir, exist_ok=True)
    os.makedirs(pbrf_model_dir, exist_ok=True)

    if args.fig_title == "none":
        fig_title = args.eval_metric
    else:
        fig_title = args.fig_title

    dirs = dict()
    dirs = {'vanilla': vanilla_model_dir, 
            'result': result_dir,
            'result_root': result_root,
            "result_id": result_id,
            "loo_model": loo_model_dir,
            "fig_title": fig_title,
            "pbrf_model": pbrf_model_dir
            }

    return dirs


def rename_result_dir(args, retrain_inf, retrain_pbrf, perturb_inf, perturb_pbrf, dirs):
    if args.json_config == "none":
        torch_retrain_inf = retrain_inf.clone().detach().squeeze()
        torch_retrain_pbrf = torch.tensor(retrain_pbrf)
        torch_perturb_inf = perturb_inf.clone().detach().squeeze()
        torch_perturb_pbrf = torch.tensor(perturb_pbrf)
        retrain_corr = torch.corrcoef(torch.stack((torch_retrain_inf, torch_retrain_pbrf)))[0, 1]
        perturb_corr = torch.corrcoef(torch.stack((torch_perturb_inf, torch_perturb_pbrf)))[0, 1]
        l2_error = torch.norm(torch_retrain_inf - torch_retrain_pbrf, p=2).item()
        new_save_dir = osp.join(dirs['result_root'], f'{retrain_corr:.2f}_{l2_error:.4f}_{perturb_corr:.2f}_{dirs['result_id']}')
        shutil.move(dirs['result'], new_save_dir)
        dirs['result'] = new_save_dir

    return


