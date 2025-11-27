import torch

def bregman_divergence(y, y_s, label, loss_func, berman_grad):
    """
    Bregman Divergence 계산.

    Args:
        y (torch.Tensor): 대상 점
        y_s (torch.Tensor): 기준 점
        label (torch.Tensor): Ground truth label
        loss_func (callable): Convex loss function (e.g., MSE, cross-entropy)

    Returns:
        torch.Tensor: Bregman Divergence 값
    """
    # Bregman Divergence 계산
    phi_y = loss_func(y, label)  # \phi(y)
    phi_ys = loss_func(y_s, label)  # \phi(y_s)
    divergence_term = torch.dot(berman_grad.view(-1), (y - y_s).view(-1))  # \nabla \phi(y_s) * (y - y_s)
    #phi_ys = 1
    
    return phi_y - phi_ys - divergence_term

def pbrf_loss(y_r, y_a, y, y_s, label_r, label, loss_func, lamb, theta, theta_s, berman_grad, num_trains):
    """
    PBRF(Penalized Bregman Risk Function) 계산.

    Args:
        y_r (torch.Tensor): 제거할 데이터
        y (torch.Tensor): 대상 점
        y_s (torch.Tensor): 기준 점
        label_r : 제거할 데이터의 라벨
        label (torch.Tensor): Ground truth label
        loss_func (callable): Convex loss function (e.g., MSE, cross-entropy)
        lamb (float): Regularization coefficient
        theta (torch.Tensor): Regularization parameter

    Returns:
        torch.Tensor: PBRF 값
    """
    # Bregman Divergence 계산
    divergence = bregman_divergence(y, y_s, label, loss_func, berman_grad)
    
    num_influenced_nodes = label_r.shape[0]
    remove_loss = loss_func(y_r, label_r) * num_influenced_nodes
    add_loss = loss_func(y_a, label_r) * num_influenced_nodes

    # PBRF 계산
    pbrf_value = divergence - 1/num_trains * (remove_loss - add_loss) + (lamb / 2) * torch.norm(theta - theta_s, p=2) ** 2  # Regularization term

    return pbrf_value, remove_loss, add_loss

