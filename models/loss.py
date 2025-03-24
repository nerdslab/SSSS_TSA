import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import linalg as LA
#from kmeans_pytorch import kmeans

class KLDiv(nn.Module):
    # Calculate KL-Divergence

    def forward(self, predict, target):
        assert predict.ndimension() == 2, 'Input dimension must be 2'
        target = target.detach()
        eps = 1e-70
        # KL(T||I) = \sum T(logT-logI)
        predict = eps + predict
        target = eps + target

        logI = torch.log(predict)
        logT = torch.log(target)
        logdiff = logT - logI
        TlogTdI = target * (logdiff)
        kld = TlogTdI.sum(1)
        #  criter = nn.MSELoss()
        #  kld = criter(predict,target)

        return kld


class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)

def ova_loss(out_open, label):
    assert len(out_open.size()) == 3
    assert out_open.size(1) == 2

    out_open = F.softmax(out_open, 1)
    
    label_p = torch.zeros((out_open.size(0),
                           out_open.size(2))).long().cuda()
    label_range = torch.arange(0, out_open.size(0)).long() # - 1
    label_p[label_range, label] = 1
    label_n = 1 - label_p
    open_loss_pos = torch.mean(torch.sum(-torch.log(out_open[:, 1, :]
                                                    + 1e-8) * label_p, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(out_open[:, 0, :] +
                                                1e-8) * label_n, 1)[0])
    return open_loss_pos, open_loss_neg

def open_entropy(out_open):
        assert len(out_open.size()) == 3
        assert out_open.size(1) == 2
        out_open = F.softmax(out_open, 1)
        ent_open = torch.mean(torch.mean(torch.sum(-out_open * torch.log(out_open + 1e-8), 1), 1))
        return ent_open
    
class VAT(nn.Module):
    def __init__(self, model, device):
        super(VAT, self).__init__()
        self.n_power = 1
        self.XI = 1e-6
        self.model = model
        self.epsilon = 3.5
        self.device = device

    def forward(self, X, logit):
        vat_loss = self.virtual_adversarial_loss(X, logit)
        return vat_loss

    def generate_virtual_adversarial_perturbation(self, x, logit):
        d = torch.randn_like(x, device=self.device)

        for _ in range(self.n_power):
            d = self.XI * self.get_normalized_vector(d).requires_grad_()
            logit_m = self.model(x + d)
            dist = self.kl_divergence_with_logit(logit, logit_m)
            grad = torch.autograd.grad(dist, [d])[0]
            d = grad.detach()

        return self.epsilon * self.get_normalized_vector(d)

    def kl_divergence_with_logit(self, q_logit, p_logit):
        q = F.softmax(q_logit, dim=1)
        qlogq = torch.mean(torch.sum(q * F.log_softmax(q_logit, dim=1), dim=1))
        qlogp = torch.mean(torch.sum(q * F.log_softmax(p_logit, dim=1), dim=1))
        return qlogq - qlogp

    def get_normalized_vector(self, d):
        return F.normalize(d.view(d.size(0), -1), p=2, dim=1).reshape(d.size())

    def virtual_adversarial_loss(self, x, logit):
        r_vadv = self.generate_virtual_adversarial_perturbation(x, logit)
        logit_p = logit.detach()
        logit_m = self.model(x + r_vadv)
        loss = self.kl_divergence_with_logit(logit_p, logit_m)
        return loss


class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
            torch.cuda.empty_cache()
            return loss


class CORAL(nn.Module):
    def __init__(self):
        super(CORAL, self).__init__()

    def forward(self, source, target):
        d = source.size(1)

        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm

        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt

        # frobenius norm between source and target
        loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
        loss = loss / (4 * d * d)
        return loss


### FOR DCAN #######################
def EntropyLoss(input_):
    mask = input_.ge(0.0000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = - (torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def MMD(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i + 1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)


def MMD_reg(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size_source = int(source.size()[0])
    batch_size_target = int(target.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size_source):
        s1, s2 = i, (i + 1) % batch_size_source
        t1, t2 = s1 + batch_size_target, s2 + batch_size_target
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size_source + batch_size_target)


### FOR HoMM #######################
class HoMM_loss(nn.Module):
    def __init__(self):
        super(HoMM_loss, self).__init__()

    def forward(self, xs, xt):
        xs = xs - torch.mean(xs, axis=0)
        xt = xt - torch.mean(xt, axis=0)
        xs = torch.unsqueeze(xs, axis=-1)
        xs = torch.unsqueeze(xs, axis=-1)
        xt = torch.unsqueeze(xt, axis=-1)
        xt = torch.unsqueeze(xt, axis=-1)
        xs_1 = xs.permute(0, 2, 1, 3)
        xs_2 = xs.permute(0, 2, 3, 1)
        xt_1 = xt.permute(0, 2, 1, 3)
        xt_2 = xt.permute(0, 2, 3, 1)
        HR_Xs = xs * xs_1 * xs_2  # dim: b*L*L*L
        HR_Xs = torch.mean(HR_Xs, axis=0)  # dim: L*L*L
        HR_Xt = xt * xt_1 * xt_2
        HR_Xt = torch.mean(HR_Xt, axis=0)
        return torch.mean((HR_Xs - HR_Xt) ** 2)


### FOR DSAN #######################
class LMMD_loss(nn.Module):
    def __init__(self, device, class_num=3, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(LMMD_loss, self).__init__()
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type
        self.device = device

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def get_loss(self, source, target, s_label, t_label):
        batch_size = source.size()[0]
        weight_ss, weight_tt, weight_st = self.cal_weight(
            s_label, t_label, batch_size=batch_size, class_num=self.class_num)
        weight_ss = torch.from_numpy(weight_ss).to(self.device)
        weight_tt = torch.from_numpy(weight_tt).to(self.device)
        weight_st = torch.from_numpy(weight_st).to(self.device)

        kernels = self.guassian_kernel(source, target,
                                       kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        loss = torch.Tensor([0]).to(self.device)
        if torch.sum(torch.isnan(sum(kernels))):
            return loss
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]

        loss += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
        return loss

    def convert_to_onehot(self, sca_label, class_num=31):
        return np.eye(class_num)[sca_label]

    def cal_weight(self, s_label, t_label, batch_size=32, class_num=4):
        batch_size = s_label.size()[0]
        s_sca_label = s_label.cpu().data.numpy()
        s_vec_label = self.convert_to_onehot(s_sca_label, class_num=self.class_num)
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum

        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        t_vec_label = t_label.cpu().data.numpy()
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        index = list(set(s_sca_label) & set(t_sca_label))
        mask_arr = np.zeros((batch_size, class_num))
        mask_arr[:, index] = 1
        t_vec_label = t_vec_label * mask_arr
        s_vec_label = s_vec_label * mask_arr

        weight_ss = np.matmul(s_vec_label, s_vec_label.T)
        weight_tt = np.matmul(t_vec_label, t_vec_label.T)
        weight_st = np.matmul(s_vec_label, t_vec_label.T)

        length = len(index)
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')


# Adapted from https://github.com/gpeyre/SinkhornAutoDiff

class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().to('cuda')
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().to('cuda')

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            # print(mu.device, self.M(C,u,v).device)
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
       
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps, max_iter, reduction='none'):

        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.count = 0
    def forward(self, x, y):
        self.count = 0
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().to('cuda')
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().to('cuda')

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            # print(mu.device, self.M(C,u,v).device)
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()
            self.count += 1
            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"

        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u10


class SinkhornDistance_custom(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance_custom, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y,C):
        # The Sinkhorn algorithm takes as input three variables :
        #C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().to('cuda')
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().to('cuda')

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            # print(mu.device, self.M(C,u,v).device)
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"

        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u10

class LOT(nn.Module):
    def __init__(self,n_src_ancs,n_trgt_ancs,eps,eps_z,intensity_vector,device,floyditer=30,norm=2,tolratio=1e-3,random_state = None):
        super(LOT, self).__init__()
        self.epsilon = eps
        self.epsilon_z = eps_z
        self.device = device
        self.n_source_anchors, self.n_target_anchors = n_src_ancs, n_trgt_ancs

        self.intensity = intensity_vector
        self.niter = floyditer
        self.tolratio = tolratio
        self.p = norm

        self.random_state = random_state

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    def update_transport_plans(self, Kx, Kz, Ky, niter=100, tol=1e-20, epsilon=0, clip_val=1e8, epsilon1=0):

        dimx = Kx.shape[0]
        dimy = Ky.shape[1]
        dimz1, dimz2 = Kz.shape

        mu = (1 / dimx * torch.ones([dimx, 1])).to(self.device)
        nu = 1 / dimy * torch.ones([dimy, 1]).to(self.device)

        epsilon1 = torch.zeros(mu.shape).to(self.device)
        epsilon = torch.zeros((dimz1,1)).to(self.device)
        clip_val1 = clip_val*torch.ones(mu.shape).to(self.device)
        clip_val2 = clip_val * torch.ones((dimz1,1)).to(self.device)
        ax = torch.ones([dimx, 1]).to(self.device)
        bx = torch.ones([dimz1, 1]).to(self.device)
        ay = torch.ones([dimz2, 1]).to(self.device)
        by = torch.ones([dimy, 1]).to(self.device)
        az = torch.ones([dimz1, 1]).to(self.device)
        bz = torch.ones([dimz2, 1]).to(self.device)
        wxz = torch.ones([dimz1, 1]).to(self.device)
        wzy = torch.ones([dimz2, 1]).to(self.device)
        for i in range(1, niter + 1):

            ax = torch.exp(
                torch.minimum(torch.log(torch.maximum(mu, epsilon1)) - torch.log(torch.maximum(Kx.matmul(bx), epsilon1)), clip_val1))
            err1x = LA.norm(bx * Kx.T.matmul(ax) - wxz, ord=1)

            by = torch.exp(
                torch.minimum(torch.log(torch.maximum(nu, epsilon1)) - torch.log(torch.maximum(Ky.T.matmul(ay), epsilon1)), clip_val1))
            err2y = LA.norm(ay * (Ky.matmul(by)) - wzy, ord=1)

            wxz = ((az * (Kz.matmul(bz))) * (bx * (Kx.T.matmul(ax)))) ** (1 / 2)
            bx = torch.exp(
                torch.minimum(torch.log(torch.maximum(wxz, epsilon)) - torch.log(torch.maximum(Kx.T.matmul(ax), epsilon)), clip_val2))
            err2x = LA.norm(ax * (Kx.matmul(bx)) - mu, ord=1)

            az = torch.exp(
                torch.minimum(torch.log(torch.maximum(wxz, epsilon)) - torch.log(torch.maximum(Kz.matmul(bz), epsilon)), clip_val2))
            err1z = LA.norm(bz * Kz.T.matmul(az) - wzy, ord=1)
            wzy = ((ay * (Ky.matmul(by))) * (bz * (Kz.T.matmul(az)))) ** (1 / 2)
            bz = torch.exp(
                torch.minimum(torch.log(torch.maximum(wzy, epsilon)) - torch.log(torch.maximum(Kz.T.matmul(az), epsilon)), clip_val2))
            err2z = LA.norm(az * (Kz.matmul(bz)) - wxz, ord=1)

            ay = torch.exp(
                torch.minimum(torch.log(torch.maximum(wzy, epsilon)) - torch.log(torch.maximum(Ky.matmul(by), epsilon)), clip_val2))
            err1y = LA.norm(by * Ky.T.matmul(ay) - nu, ord=1)
            if max(err1x, err2x, err1z, err2z, err1y, err2y) < tol:
                break

        Px = torch.diagflat(ax).matmul(Kx.matmul(torch.diagflat(bx)))
        Pz = torch.diagflat(az).matmul(Kz.matmul(torch.diagflat(bz)))
        Py = torch.diagflat(ay).matmul(Ky.matmul(torch.diagflat(by)))
        const = 0
        z1 = Px.T.matmul(torch.ones([dimx, 1]).to(self.device)) + const
        z2 = Py.matmul(torch.ones([dimy, 1]).to(self.device)) + const
        P = torch.matmul(Px / z1.T, torch.matmul(Pz, Py / z2))
        return Px, Py, Pz, P

    def update_anchors(self, Px, Py, Pz, source, target):
        'source'
        n = source.shape[0]
        m = target.shape[0]
        Px = self.intensity[0] * Px
        Pz = self.intensity[1] * Pz
        Py = self.intensity[2] * Py

        temp = torch.concat((torch.diagflat(Px.T.matmul(torch.ones([n, 1]).to(self.device)) +
                                           Pz.matmul(torch.ones([self.n_target_anchors, 1]).to(self.device))), -Pz), axis=1)
        temp1 = torch.concat((-Pz.T, torch.diagflat(Py.matmul(torch.ones([m, 1]).to(self.device)) +
                                                   Pz.T.matmul(torch.ones([self.n_source_anchors, 1]).to(self.device)))), axis=1)
        temp = torch.concat((temp, temp1), axis=0)
        sol = torch.concat((source.T.matmul(Px), target.T.matmul(Py.T)), axis=1).matmul(LA.inv(temp))
        Cx = sol[:, 0:self.n_source_anchors].T
        Cy = sol[:, self.n_source_anchors:self.n_source_anchors + self.n_target_anchors].T
        return Cx, Cy

    def transport(self, source, target):
        n = source.shape[0]
        m = target.shape[0]
        Cx_lot = self.Px_.T.dot(source) / (self.Px_.T.dot(np.ones([n, 1])) + 10 ** -20)
        Cy_lot = self.Py_.dot(target) / (self.Py_.dot(np.ones([m, 1])) + 10 ** -20)
        transported = source + np.dot(
            np.dot(
                self.Px_ / np.sum(self.Px_, axis=1).reshape([n, 1]),
                self.Pz_ / np.sum(self.Pz_, axis=1).reshape([self.n_source_anchors, 1])
            ),
            Cy_lot) - np.dot(self.Px_ / np.sum(self.Px_, axis=1).reshape([n, 1]), Cx_lot)
        return transported

    def robust_transport(self, source, target, threshold=0.8, decay=0):
        n = source.shape[0]
        m = target.shape[0]
        Cx_lot = self.Px_.T.dot(source) / (self.Px_.T.dot(np.ones([n, 1])) + 10 ** -20)
        Cy_lot = self.Py_.dot(target) / (self.Py_.dot(np.ones([m, 1])) + 10 ** -20)

        maxPz = np.max(self.Pz_, axis=1)
        Pz_robust = self.Pz_.copy()

        for i in range(0, self.n_source_anchors):
            for j in range(0, self.n_target_anchors):
                if self.Pz_[i, j] < maxPz[i] * threshold:
                    Pz_robust[i, j] = self.Pz_[i, j] * decay
        Pz_robust = Pz_robust / np.sum(Pz_robust, axis=1).reshape([self.n_source_anchors, 1]) * \
                    np.sum(self.Pz_, axis=1).reshape([self.n_source_anchors, 1])

        transported = source + np.dot(
            np.dot(
                self.Px_ / np.sum(self.Px_, axis=1).reshape([n, 1]),
                Pz_robust / np.sum(Pz_robust, axis=1).reshape([self.n_source_anchors, 1])
            ), Cy_lot) - np.dot(self.Px_ / np.sum(self.Px_, axis=1).reshape([n, 1]), Cx_lot)
        return transported

    def forward(self, source, target):
        # centroid initialized by K-means
        Cx = compute_kmeans_centroids(source, n_clusters=self.n_source_anchors, device =self.device).to(self.device)#.to(self.device)
        Cy = compute_kmeans_centroids(target, n_clusters=self.n_target_anchors, device =self.device).to(self.device)
        C = self._cost_matrix(source,target , p =self.p)
        # Px, Py initialized by K-means and one-sided OT
        n = source.shape[0]
        m = target.shape[0]
        #mu = 1 / n * torch.ones([n, 1]).to(self.device)
        #nu = 1 / m * torch.ones([m, 1]).to(self.device)
        #cost_xy = compute_cost_matrix(source, target, p=self.p)
        P = (torch.zeros([n, m]) + 1 / n / m).to(self.device)
        
        converrlist = np.zeros(self.niter) + np.inf
        for t in range(0, self.niter):

            # compute cost matrices
            cost_x = self._cost_matrix(source, Cx)
            cost_z = self._cost_matrix(Cx, Cy, p=self.p)
            cost_y = self._cost_matrix(Cy, target, p=self.p)
            Kx = torch.exp(-self.intensity[0] * cost_x / self.epsilon)
            Kz = torch.exp(-self.intensity[1] * cost_z / self.epsilon_z)
            Ky = torch.exp(-self.intensity[2] * cost_y / self.epsilon)

            Pt1 = P
            Px, Py, Pz, P = self.update_transport_plans(Kx, Kz, Ky)  # update trans. plan

            # check for convergence
            converr = LA.norm(P - Pt1) / LA.norm(Pt1)
            converrlist[t] = converr
            if converr < self.tolratio:
                break

            # update anchors
            if t < self.niter - 1:
                Cx, Cy = self.update_anchors(Px, Py, Pz, source, target)

        self.Cx, self.Cy = Cx, Cy
        cost = torch.sum(P * C, dim=(-2, -1))
        self.Px_, self.Py_, self.Pz_, self.P_ = Px, Py, Pz, P
        return cost,Cx,Cy


def compute_kmeans_centroids(X,n_clusters,device):
    _, cluster_centers = kmeans(
    X=X, num_clusters=n_clusters, distance='euclidean', device=device)
    return cluster_centers


def compute_cost_matrix(source, target, p=2):
    cost_matrix = np.sum(np.power(source.reshape([source.shape[0], 1, source.shape[1]]) -
                                  target.reshape([1, target.shape[0], target.shape[1]]),
                                  p), axis=-1)
    return cost_matrix

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive #divide by 0 positive)
        mean_log_prob_pos = (mask * log_prob).sum(1) / torch.max(mask.sum(1),torch.ones(len(mask)).to(device))

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        "get mask for similar points (diagonal and off diagonal, negative (rest of them left)"
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)

        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)
        'Broadcasting to get a 2N x @n similarioty matrix'
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature


        'get off diagonal terms to get the similarity points'
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        '''Reshape into logits where N = 28*batch with no of classes or equivalent = 1 + neg Each sample has one sim and 
        rest negative'''

        logits = torch.cat((positive_samples, negative_samples), dim=1)
        # SIMCLR

        'get 0 labels for all classes'
        labels = torch.from_numpy(np.array([0] * N)).reshape(-1).to(positive_samples.device).long()  # .float()


        'Use cross entropy loss '
        loss = self.criterion(logits, labels)
        loss /= N

        return loss