import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
#from .RAINCOAT import CNN
import torch.nn.functional as F
from models.models import classifier, ReverseLayerF, Discriminator, RandomLayer, Discriminator_CDAN, \
    codats_classifier, AdvSKM_Disc, SepReps_with_multihead,SepReps,AttnCombMultihead,SepReps_with_multihead_with_freq,\
    simple_average_feed_forward, SepReps_with_sum,CLUDA_NN
from models.loss import MMD_loss, CORAL, ConditionalEntropyLoss, VAT, LMMD_loss, HoMM_loss
from models.augmentations import jitter, scaling, permutation
from torch.optim import SGD
from models.loss import SinkhornDistance,LOT,SupConLoss,SimCLR_Loss
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
from CLUDA_main.utils.augmentations import Augmenter, concat_mask
def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs

        #self.cross_entropy = nn.CrossEntropyLoss(weight=torch.tensor([0.1,8]).float().to(0))
        #when no imbalanced
        self.cross_entropy = nn.CrossEntropyLoss()

    def update(self, *args, **kwargs):
        raise NotImplementedError


class Lower_Upper_bounds(Algorithm):
    """
    Lower bound: train on source and test on target.
    Upper bound: train on target and test on target.
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(Lower_Upper_bounds, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )
        self.hparams = hparams

    def update(self, src_x, src_y):
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        src_cls_loss = self.cross_entropy(src_pred, src_y)

        loss = src_cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Src_cls_loss': src_cls_loss.item()}


class DANCE(Algorithm):
    """
    Universal Domain Adaptation through Self-Supervision
    https://arxiv.org/abs/2002.07953
    Original code: https://github.com/VisionLearningGroup/DANCE
    """

    class LinearAverage(nn.Module):
        def __init__(self, inputSize, outputSize, T=0.05, momentum=0.0):
            super().__init__()
            self.nLem = outputSize
            self.momentum = momentum
            self.register_buffer('params', torch.tensor([T, momentum]));
            self.register_buffer('memory', torch.zeros(outputSize, inputSize))
            self.flag = 0
            self.T = T
            # self.memory =  self.memory.cuda()
        def forward(self, x, y):
            out = torch.mm(x, self.memory.t())/self.T
            return out

        def update_weight(self, features, index):
            if not self.flag:
                weight_pos = self.memory.index_select(0, index.data.view(-1)).resize_as_(features)
                weight_pos.mul_(0.0)
                weight_pos.add_(torch.mul(features.data, 1.0))

                w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
                updated_weight = weight_pos.div(w_norm)
                self.memory.index_copy_(0, index, updated_weight)
                self.flag = 1
            else:
                weight_pos = self.memory.index_select(0, index.data.view(-1)).resize_as_(features)
                weight_pos.mul_(self.momentum)
                weight_pos.add_(torch.mul(features.data, 1 - self.momentum))

                w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
                updated_weight = weight_pos.div(w_norm)
                self.memory.index_copy_(0, index, updated_weight)

            self.memory = F.normalize(self.memory)#.cuda()


        def set_weight(self, features, index):
            self.memory.index_copy_(0, index, features)


    @staticmethod
    def entropy(p):
        p = F.softmax(p,dim=-1)
        return -torch.mean(torch.sum(p * torch.log(p+1e-5), 1))

    @staticmethod
    def entropy_margin(p, value, margin=0.2, weight=None):

        def hinge(input, margin=0.2):
            return torch.clamp(input, min=margin)

        p = F.softmax(p, -1)
        return -torch.mean(hinge(torch.abs(-torch.sum(p * torch.log(p+1e-5), 1)-value), margin))


    def __init__(self, backbone_fe, configs, hparams, device, trg_train_size):
        super().__init__(configs)
        
        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)

        self.optimizer = torch.optim.SGD(
            list(self.feature_extractor.parameters()) + list(self.classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
            momentum=hparams["momentum"],
            nesterov=True,
        )

        self.lemniscate = self.LinearAverage(configs.features_len * configs.final_out_channels, trg_train_size, hparams["temp"])
        self.device = device
        self.hparams = hparams

    def update(self, src_x, src_y, trg_x, trg_index, step, epoch, len_dataloader):
        total_steps = self.hparams["num_epochs"] + 1 / len_dataloader
        current_step = step + epoch * len_dataloader

        # TODO: weight norm on feature extractor?

        src_feat = self.feature_extractor(src_x)
        src_logits = self.classifier(src_feat)
        src_loss = F.cross_entropy(src_logits, src_y)

        trg_feat = self.feature_extractor(trg_x)
        trg_logits = self.classifier(trg_feat)
        trg_feat = F.normalize(trg_feat)

        # calculate mini-batch x memory similarity
        feat_mat = self.lemniscate(trg_feat, trg_index)

        # do not use memory features present in mini-batch
        feat_mat[:, trg_index] = -1 / self.hparams["temp"]

        # calculate mini-batch x mini-batch similarity
        feat_mat2 = torch.matmul(trg_feat, trg_feat.t()) / self.hparams["temp"]

        mask = torch.eye(feat_mat2.shape[0], feat_mat2.shape[0]).bool().to(self.device)
    
        feat_mat2.masked_fill_(mask, -1 / self.hparams["temp"])

        loss_nc = self.hparams["eta"] * self.entropy(torch.cat([trg_logits, feat_mat, feat_mat2], 1))

        loss_ent = self.hparams["eta"] * self.entropy_margin(trg_logits, self.hparams["thr"], self.hparams["margin"])

        loss = src_loss + loss_nc + loss_ent

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.lemniscate.update_weight(trg_feat, trg_index)

        return {'total_loss': loss.item(), 'src_loss': src_loss.item(), 'loss_nc': loss_nc.item(), 'loss_ent': loss_nc.item()}


class OVANet(Algorithm):
    """
    OVANet https://arxiv.org/pdf/2104.03344v3.pdf
    Based on PyTorch implementation: https://github.com/VisionLearningGroup/OVANet
    """
    def __init__(self, backbone_fe, configs, hparams, device):
        super().__init__(configs)
        
        self.device = device
        self.hparams = hparams
        self.criterion = nn.CrossEntropyLoss()
        
        self.feature_extractor = backbone_fe(configs) # G
        self.classifier1 = classifier(configs) # C1
        
        configs2 = configs
        configs2.num_classes = configs.num_classes * 2
        
        self.classifier2 = classifier(configs2) # C2
        
        self.feature_extractor.to(device)
        self.classifier1.to(device)
        self.classifier2.to(device)
        
        self.opt_g = SGD(self.feature_extractor.parameters(), momentum=self.hparams['sgd_momentum'],
                         lr = self.hparams['learning_rate'], weight_decay=0.0005, nesterov=True)
        self.opt_c = SGD(list(self.classifier1.parameters()) + list(self.classifier2.parameters()), lr=1.0,
                           momentum=self.hparams['sgd_momentum'], weight_decay=0.0005,
                           nesterov=True)
        
        param_lr_g = []
        for param_group in self.opt_g.param_groups:
            param_lr_g.append(param_group["lr"])
        param_lr_c = []
        for param_group in self.opt_c.param_groups:
            param_lr_c.append(param_group["lr"])
        
        self.param_lr_g = param_lr_g
        self.param_lr_c = param_lr_c

    
    @staticmethod
    def _inv_lr_scheduler(param_lr, optimizer, iter_num, gamma=10,
                     power=0.75, init_lr=0.001,weight_decay=0.0005,
                     max_iter=10000):
        #10000
        """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
        #max_iter = 10000
        gamma = 10.0
        lr = init_lr * (1 + gamma * min(1.0, iter_num / max_iter)) ** (-power)
        i=0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_lr[i]
            i+=1
        return lr

    def update(self, src_x, src_y, trg_x, step, epoch, len_train_source, len_train_target):
        
        # Applying classifier network => replacing G, C2 in paper
        self.feature_extractor.train()
        self.classifier1.train()
        self.classifier2.train()
        
        self._inv_lr_scheduler(self.param_lr_g, self.opt_g, step,
                         init_lr=self.hparams['learning_rate'],
                         max_iter=self.hparams['min_step'])
        self._inv_lr_scheduler(self.param_lr_c, self.opt_c, step,
                         init_lr=self.hparams['learning_rate'],
                         max_iter=self.hparams['min_step'])
        
        self.opt_g.zero_grad()
        self.opt_c.zero_grad()
        
#         self.classifier2.weight_norm()
        
        ## Source loss calculation
        out_s = self.classifier1(self.feature_extractor(src_x))
        out_open = self.classifier2(self.feature_extractor(src_x))

        ## source classification loss
        loss_s = self.criterion(out_s, src_y)
        ## open set loss for source
        out_open = out_open.view(out_s.size(0), 2, -1)
        open_loss_pos, open_loss_neg = ova_loss(out_open, src_y)
        ## b x 2 x C
        loss_open = 0.5 * (open_loss_pos + open_loss_neg)
        ## open set loss for target
        all = loss_s + loss_open
        
        # OEM - Open Entropy Minimization
        no_adapt = False
        if not no_adapt: # TODO: Figure out if this needs to be altered
            out_open_t = self.classifier2(self.feature_extractor(trg_x))
            out_open_t = out_open_t.view(trg_x.size(0), 2, -1)

            ent_open = open_entropy(out_open_t)
            all += self.hparams['multi'] * ent_open
        
        all.backward()
        
        self.opt_g.step()
        self.opt_c.step()
        self.opt_g.zero_grad()
        self.opt_c.zero_grad()

        return {'src_loss': loss_s.item(),
                'open_loss': loss_open.item(), 
                'open_src_pos_loss': open_loss_pos.item(),
                'open_src_neg_loss': open_loss_neg.item(),
                'open_trg_loss': ent_open.item()
               }

class AdaMatch(Algorithm):
    """
    AdaMatch https://arxiv.org/abs/2106.04732
    Based on PyTorch implementation: https://github.com/zysymu/AdaMatch-pytorch
    """
    def __init__(self, backbone_fe, configs, hparams, device):
        super().__init__(configs)
        
        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)

        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + list(self.classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.device = device
        self.hparams = hparams

    @staticmethod
    def _enable_batchnorm_tracking(model):
        """start tracking running stats for batch norm"""
        def fn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = True

        model.apply(fn)
        
    @staticmethod
    def _disable_batchnorm_tracking(model):
        """stop tracking running stats for batch norm"""
        def fn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = False

        model.apply(fn)
        
    @staticmethod
    def _compute_src_loss(logits_weak, logits_strong, labels):
        loss_function = nn.CrossEntropyLoss()
        weak_loss = loss_function(logits_weak, labels)
        strong_loss = loss_function(logits_strong, labels)
        return (weak_loss + strong_loss) / 2

    @staticmethod
    def _compute_trg_loss(pseudolabels, logits_strong, mask):
        loss_function = nn.CrossEntropyLoss(reduction="none")
        pseudolabels = pseudolabels.detach()
        loss = loss_function(logits_strong, pseudolabels)
        return (loss * mask).mean()
    
    def augment_weak(self, x):
        return scaling(x, self.hparams["jitter_scale_ratio"])

    def augment_strong(self, x):
        return jitter(permutation(x, max_segments=self.hparams["max_segments"]), self.hparams["jitter_ratio"])
    
    def update(self, src_x, src_y, trg_x, step, epoch, len_dataloader):

        total_steps = self.hparams["num_epochs"] + 1 / len_dataloader
        current_step = step + epoch * len_dataloader
    
        src_x_weak = self.augment_weak(src_x)
        src_x_strong = self.augment_strong(src_x)

        trg_x_weak = self.augment_weak(trg_x)
        trg_x_strong = self.augment_strong(trg_x)

        x_combined = torch.cat([src_x_weak, src_x_strong, trg_x_weak, trg_x_strong], dim=0)
        src_x_combined = torch.cat([src_x_weak, src_x_strong], dim=0)

        src_total = src_x_combined.shape[0]

        logits_combined = self.classifier(self.feature_extractor(x_combined))
        logits_source_p = logits_combined[:src_total]

        self._disable_batchnorm_tracking(self.feature_extractor)
        self._disable_batchnorm_tracking(self.classifier)
        logits_source_pp = self.classifier(self.feature_extractor(src_x_combined))
        self._enable_batchnorm_tracking(self.feature_extractor)
        self._enable_batchnorm_tracking(self.classifier)

        # random logit interpolation
        lambd = torch.rand_like(logits_source_p)
        final_logits_src = (lambd * logits_source_p) + ((1 - lambd) * logits_source_pp)

        # distribution alignment
        # softmax for logits of weakly augmented source timeseries
        logits_src_weak = final_logits_src[:src_x_weak.shape[0]]
        pseudolabels_src = F.softmax(logits_src_weak, dim=1)

        # softmax for logits of weakly augmented target timeseries
        logits_trg = logits_combined[src_total:]
        logits_trg_weak = logits_trg[:trg_x_weak.shape[0]]
        pseudolabels_trg = F.softmax(logits_trg_weak, dim=1)


        # align target label distribution to source label distribution
        expectation_ratio = (1e-6 + torch.mean(pseudolabels_src)) / (1e-6 + torch.mean(pseudolabels_trg))
        # l2 norm
        final_pseudolabels = F.normalize((pseudolabels_trg * expectation_ratio), p=2, dim=1)

        # relative confidence tresholding
        row_wise_max, _ = torch.max(pseudolabels_src, dim=1)
        final_sum = torch.mean(row_wise_max)

        # relative confidence threshold
        c_tau = self.hparams['tau'] * final_sum

        max_values, _ = torch.max(final_pseudolabels, dim=1)
        mask = (max_values >= c_tau).float()

        src_loss = self._compute_src_loss(logits_src_weak, final_logits_src[src_x_weak.shape[0]:], src_y)

        final_pseudolabels = torch.max(final_pseudolabels, 1)[1]
        trg_loss = self._compute_trg_loss(final_pseudolabels, logits_trg[trg_x_weak.shape[0]:], mask)

        pi = torch.tensor(np.pi, dtype=torch.float).to(self.device)
        mu = 0.5 - torch.cos(torch.minimum(pi, (2 * pi * current_step) / total_steps)) / 2
        loss = src_loss + (mu * trg_loss)
        

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'src_loss': src_loss.item(), 'Domain_loss': trg_loss.item(), "mu": mu.item(), "current_step": current_step}


class Deep_Coral(Algorithm):
    """
    Deep Coral: https://arxiv.org/abs/1607.01719
    """
    def __init__(self, backbone_fe, configs, hparams, device):
        super(Deep_Coral, self).__init__(configs)

        self.coral = CORAL()

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, src_x, src_y, trg_x):
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        src_cls_loss = self.cross_entropy(src_pred, src_y)

        trg_feat = self.feature_extractor(trg_x)

        coral_loss = self.coral(src_feat, trg_feat)

        loss = self.hparams["coral_wt"] * coral_loss + \
               self.hparams["src_cls_loss_wt"] * src_cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Domain_loss': coral_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


class MMDA(Algorithm):
    """
    MMDA: https://arxiv.org/abs/1901.00282
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(MMDA, self).__init__(configs)

        self.mmd = MMD_loss()
        self.coral = CORAL()
        self.cond_ent = ConditionalEntropyLoss()

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, src_x, src_y, trg_x):
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        src_cls_loss = self.cross_entropy(src_pred, src_y)

        trg_feat = self.feature_extractor(trg_x)

        coral_loss = self.coral(src_feat, trg_feat)
        mmd_loss = self.mmd(src_feat, trg_feat)
        cond_ent_loss = self.cond_ent(trg_feat)

        loss = self.hparams["coral_wt"] * coral_loss + \
               self.hparams["mmd_wt"] * mmd_loss + \
               self.hparams["cond_ent_wt"] * cond_ent_loss + \
               self.hparams["src_cls_loss_wt"] * src_cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Coral_loss': coral_loss.item(), 'MMD_loss': mmd_loss.item(),
                'cond_ent_wt': cond_ent_loss.item(), 'Src_cls_loss': src_cls_loss.item()}

class Supervised(Algorithm):
    """
    MMDA: https://arxiv.org/abs/1901.00282
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(Supervised, self).__init__(configs)



        self.feature_extractor = backbone_fe(configs)
        #self.feature_extractor = simple_average_feed_forward(configs)
        #backbone_fe(configs)


        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, src_x, src_y, trg_x):
        src_feat = self.feature_extractor(src_x)

        src_pred = self.classifier(src_feat)

        src_cls_loss = self.cross_entropy(src_pred, src_y)

        #trg_feat = self.feature_extractor(trg_x)

        #coral_loss = self.coral(src_feat, trg_feat)
        #mmd_loss = self.mmd(src_feat, trg_feat)
        #cond_ent_loss = self.cond_ent(trg_feat)

        loss = self.hparams["src_cls_loss_wt"]* src_cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Src_cls_loss': src_cls_loss.item(),'Domain_loss':0}


class Supervised_trg(Algorithm):
    """
    MMDA: https://arxiv.org/abs/1901.00282
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(Supervised_trg, self).__init__(configs)



        self.feature_extractor = backbone_fe(configs)
        #self.feature_extractor = simple_average_feed_forward(configs)
        #backbone_fe(configs)


        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, src_x, src_y, trg_x,trg_y,step, epoch, len_dataloader):
        trg_feat = self.feature_extractor(trg_x)

        trg_pred = self.classifier(trg_feat)

        src_cls_loss = self.cross_entropy(trg_pred, trg_y)

        #trg_feat = self.feature_extractor(trg_x)

        #coral_loss = self.coral(src_feat, trg_feat)
        #mmd_loss = self.mmd(src_feat, trg_feat)
        #cond_ent_loss = self.cond_ent(trg_feat)

        loss = self.hparams["src_cls_loss_wt"]* src_cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Src_cls_loss': src_cls_loss.item(),'Domain_loss':0}

class SepReps(Algorithm):
    """
    MMDA: https://arxiv.org/abs/1901.00282
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(SepReps, self).__init__(configs)

        self.feature_extractor =nn.ModuleList([])
        self.classifier = nn.ModuleList([])
        self.domain_classifier_list = nn.ModuleList([])
        self.optimizer_list = []
        self.optimizer_disc_list = []
        configs.input_channels=1
        for k in range(0,4):
            self.feature_extractor.append(backbone_fe(configs))
            self.classifier.append(classifier(configs))
            self.domain_classifier_list.append(Discriminator(configs))

            self.optimizer_list.append(torch.optim.Adam(
                list(self.feature_extractor[k].parameters()) + list(self.classifier[k].parameters()),
                lr=1 * hparams["learning_rate"],
                weight_decay=hparams["weight_decay"], betas=(0.5, 0.99))
            )
            self.optimizer_disc_list.append(torch.optim.Adam(
                list(self.domain_classifier_list[k].parameters()),
                lr=hparams["learning_rate"],
                weight_decay=hparams["weight_decay"], betas=(0.5, 0.99))
            )

        self.hparams = hparams
        self.feature_extractor.to(device)
        self.domain_classifier_list.to(device)
        self.classifier.to(device)
        self.device = device
    def update(self, src_x, src_y, trg_x,step,epoch,len_dataloader):
        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        total_loss = 0

        for k in range(0,4):
            # zero grad
            src_x_k = src_x[:,k,:].unsqueeze(1)
            trg_x_k = trg_x[:,k,:].unsqueeze(1)
            self.optimizer_list[k].zero_grad()
            self.optimizer_disc_list[k].zero_grad()

            domain_label_src = torch.ones(len(src_x_k)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x_k)).to(self.device)

            src_feat = self.feature_extractor[k](src_x_k)
            src_pred = self.classifier[k](src_feat)

            trg_feat = self.feature_extractor[k](trg_x_k)

            # Task classification  Loss
            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            # Domain classification loss
            # source
            src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
            src_domain_pred = self.domain_classifier_list[k](src_feat_reversed)
            src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

            # target
            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
            trg_domain_pred = self.domain_classifier_list[k](trg_feat_reversed)
            trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

            # Total domain loss
            domain_loss = src_domain_loss + trg_domain_loss

            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
                   self.hparams["domain_loss_wt"] * domain_loss

            loss.backward()
            self.optimizer_list[k].step()
            self.optimizer_disc_list[k].step()
            total_loss = loss.item()

        return {'Total_loss': loss.item(), 'Src_cls_loss': src_cls_loss.item(),'Domain_loss':domain_loss.item()}

    def eval_update(self, src, trg):
        joint_loaders = enumerate(zip(src, trg))
        len_dataloader = min(len(src), len(trg))
        src_loss_list = []
        dom_loss_list = []
        tloss_list = []
        with torch.no_grad():
            for step, ((src_x, src_y), (trg_x, trg_y)) in joint_loaders:
                src_x, src_y, trg_x, trg_y = src_x.float().to(self.device), src_y.long().to(self.device), \
                                             trg_x.float().to(self.device), trg_y.to(self.device)
                for k in range(0, 3):
                    # zero grad
                    src_x_k = src_x[:, k, :].unsqueeze(1)
                    trg_x_k = trg_x[:, k, :].unsqueeze(1)


                    domain_label_src = torch.ones(len(src_x_k)).to(self.device)
                    domain_label_trg = torch.zeros(len(trg_x_k)).to(self.device)

                    src_feat = self.feature_extractor[k](src_x_k)
                    src_pred = self.classifier[k](src_feat)

                    trg_feat = self.feature_extractor[k](trg_x_k)

                    # Task classification  Loss
                    src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

                    # Domain classification loss
                    # source

                    src_domain_pred = self.domain_classifier_list[k](src_feat)
                    src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

                    # target

                    trg_domain_pred = self.domain_classifier_list[k](trg_feat)
                    trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

                    # Total domain loss
                    domain_loss = src_domain_loss + trg_domain_loss

                    loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
                           self.hparams["domain_loss_wt"] * domain_loss


                    total_loss = loss.item()

                return {'Total_loss': loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                        'Domain_loss': domain_loss.item()}
    def eval(self,  trg_data,fpath=None,cpath=None,final=False):
        if final == True:
            self.feature_extractor.load_state_dict(torch.load(fpath))
            self.classifier.load_state_dict(torch.load(cpath))
        self.feature_extractor.eval()
        self.classifier.eval()

        total_loss_ = []

        trg_pred_labels = np.array([])
        trg_true_labels = np.array([])

        preds0 = np.asarray([])
        preds1 = np.asarray([])
        preds2 = np.asarray([])
        with torch.no_grad():
            for data, labels in trg_data:
                data = data.float().to(self.device)
                # data[:,:,0]=0
                # data[:, :, 2] = 0
                # data[:, :, 1] = 0
                preds_t = 0
                for k in range(0,3):
                    trg_x_k = data[:,k,:].unsqueeze(1)
                    features_k = self.feature_extractor[k](trg_x_k)
                    predictions_k = self.classifier[k](features_k)
                    preds_t = predictions_k+preds_t
                    if k ==0:
                        preds0 = np.concatenate((preds0, predictions_k.detach().cpu().numpy()),axis=0) if len(preds0) else  predictions_k.detach().cpu().numpy()
                    elif k == 1:
                        preds1 = np.concatenate((preds1, predictions_k.detach().cpu().numpy()),axis=0) if len(preds1) else  predictions_k.detach().cpu().numpy()
                    elif k ==2:
                        preds2 = np.concatenate((preds2, predictions_k.detach().cpu().numpy()),axis=0) if len(preds2) else  predictions_k.detach().cpu().numpy()
                pred = preds_t.detach().argmax(dim=1)  # get the index of the max log-probability

                trg_pred_labels = np.append(trg_pred_labels, pred.cpu().numpy())
                trg_true_labels = np.append(trg_true_labels, labels.data.cpu().numpy())
        accuracy = accuracy_score(trg_true_labels, trg_pred_labels)
        f1 = f1_score(trg_true_labels, trg_pred_labels, pos_label=None, average="macro")
        cm = confusion_matrix(trg_true_labels, trg_pred_labels, normalize=None)
        return accuracy * 100, f1, cm

class SinkDiv_Alignment(Algorithm):
    """
    MMDA: https://arxiv.org/abs/1901.00282
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(SinkDiv_Alignment, self).__init__(configs)


        self.sink = SinkhornDistance(eps=1e-3, max_iter=1000, reduction='sum')
        self.feature_extractor = backbone_fe(configs)
        #self.feature_extractor = simple_average_feed_forward(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters())+list(self.classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams
        self.device = device

    def update(self, src_x, src_y, trg_x):

        self.optimizer.zero_grad()
        src_feat = self.feature_extractor(src_x)
        trg_feat = self.feature_extractor(trg_x)




        #coral_loss = self.coral(src_feat, trg_feat)
        domain_loss = self.sink(src_feat, trg_feat)[0]
        #cond_ent_loss = self.cond_ent(trg_feat)
        loss_domain = domain_loss
        loss_domain.backward(retain_graph=True)

        src_pred = self.classifier(src_feat)

        src_cls_loss = self.cross_entropy(src_pred, src_y)
        loss_sup =src_cls_loss # self.hparams["src_cls_loss_wt"]* src_cls_loss #1*self.hparams['sinkdiv_loss_wt']*domain_loss



        loss_sup.backward(retain_graph=True)
        self.optimizer.step()
        #domain_loss.detach()
        return {'Total_loss': loss_sup.item()+domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(),'Domain_loss':domain_loss.item()}


    def eval_update(self, src,trg):
        joint_loaders = enumerate(zip(src, trg))
        len_dataloader = min(len(src), len(trg))
        src_loss_list = []
        dom_loss_list = []
        tloss_list = []
        with torch.no_grad():
            for step, ((src_x, src_y), (trg_x, trg_y)) in joint_loaders:
                src_x, src_y, trg_x, trg_y = src_x.float().to(self.device), src_y.long().to(self.device), \
                                             trg_x.float().to(self.device), trg_y.to(self.device)
                src_feat = self.feature_extractor(src_x)
                src_pred = self.classifier(src_feat)

                src_cls_loss = self.cross_entropy(src_pred, src_y)

                trg_feat = self.feature_extractor(trg_x)

                # coral_loss = self.coral(src_feat, trg_feat)
                domain_loss = self.sink(src_feat, trg_feat)[0]
                # cond_ent_loss = self.cond_ent(trg_feat)

                loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams['sinkdiv_loss_wt'] * domain_loss
                dom_loss_list.append(domain_loss.item())
                src_loss_list.append(src_cls_loss.item())
                tloss_list.append(loss.item())

        # self.ema.update()
        return {'Total_loss': np.mean(tloss_list), 'Src_cls_loss':np.mean(src_loss_list), 'Domain_loss': np.mean(dom_loss_list)}
class DANN(Algorithm):
    """
    DANN: https://arxiv.org/abs/1505.07818
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(DANN, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        #self.feature_extractor = simple_average_feed_forward(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier).to('cuda')

        self.domain_classifier = Discriminator(configs)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=1*hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr= hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )
        self.hparams = hparams
        self.device = device
        #self.ema = EMA2(self.network, 0.9)
        #self.ema.register()
        
    def update(self, src_x, src_y, trg_x, step, epoch, len_dataloader,val=False):
        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # zero grad
        self.optimizer.zero_grad()
        self.optimizer_disc.zero_grad()

        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        trg_feat = self.feature_extractor(trg_x)

        # Task classification  Loss
        src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

        # Domain classification loss
        # source
        src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
        src_domain_pred = self.domain_classifier(src_feat_reversed)
        src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

        # target
        trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
        trg_domain_pred = self.domain_classifier(trg_feat_reversed)
        trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

        # Total domain loss
        domain_loss = src_domain_loss + trg_domain_loss

        loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
               self.hparams["domain_loss_wt"] * domain_loss

        loss.backward()
        self.optimizer.step()
        self.optimizer_disc.step()
        # self.ema.update()
        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}

    def eval_update(self, src,trg):
        joint_loaders = enumerate(zip(src, trg))
        len_dataloader = min(len(src), len(trg))
        src_loss_list = []
        dom_loss_list = []
        tloss_list = []
        with torch.no_grad():
            for step, ((src_x, src_y), (trg_x, trg_y)) in joint_loaders:
                src_x, src_y, trg_x, trg_y = src_x.float().to(self.device), src_y.long().to(self.device), \
                                             trg_x.float().to(self.device), trg_y.to(self.device)
                domain_label_src = torch.ones(len(src_x)).to(self.device)
                domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

                src_feat = self.feature_extractor(src_x)
                src_pred = self.classifier(src_feat)

                trg_feat = self.feature_extractor(trg_x)

                # Task classification  Loss
                src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

                # Domain classification loss
                # source

                src_domain_pred = self.domain_classifier(src_feat)
                src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

                # target

                trg_domain_pred = self.domain_classifier(trg_feat)
                trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

                # Total domain loss
                domain_loss = src_domain_loss + trg_domain_loss

                loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
                       self.hparams["domain_loss_wt"] * domain_loss
                dom_loss_list.append(domain_loss.item())
                src_loss_list.append(src_cls_loss.item())
                tloss_list.append(loss.item())

        # self.ema.update()
        return {'Total_loss': np.mean(tloss_list), 'Src_cls_loss':np.mean(src_loss_list), 'Domain_loss': np.mean(dom_loss_list)}
class CDAN(Algorithm):
    """
    CDAN: https://arxiv.org/abs/1705.10667
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(CDAN, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.domain_classifier = Discriminator_CDAN(configs)
        self.random_layer = RandomLayer([configs.features_len * configs.final_out_channels, configs.num_classes],
                                        configs.features_len * configs.final_out_channels)

        # optimizers
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        # hparams
        self.hparams = hparams
        self.criterion_cond = ConditionalEntropyLoss().to(device)
        self.device = device

    def update(self, src_x, src_y, trg_x):
        # prepare true domain labels
        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

        # source features and predictions
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        # target features and predictions
        trg_feat = self.feature_extractor(trg_x)
        trg_pred = self.classifier(trg_feat)

        # concatenate features and predictions
        feat_concat = torch.cat((src_feat, trg_feat), dim=0)
        pred_concat = torch.cat((src_pred, trg_pred), dim=0)

        # Domain classification loss
        feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1)).detach()
        disc_prediction = self.domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)))
        disc_loss = self.cross_entropy(disc_prediction, domain_label_concat)

        # update Domain classification
        self.optimizer_disc.zero_grad()
        disc_loss.backward()
        self.optimizer_disc.step()

        # prepare fake domain labels for training the feature extractor
        domain_label_src = torch.zeros(len(src_x)).long().to(self.device)
        domain_label_trg = torch.ones(len(trg_x)).long().to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

        # Repeat predictions after updating discriminator
        feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1))
        disc_prediction = self.domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)))
        # loss of domain discriminator according to fake labels

        domain_loss = self.cross_entropy(disc_prediction, domain_label_concat)

        # Task classification  Loss
        src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

        # conditional entropy loss.
        loss_trg_cent = self.criterion_cond(trg_pred)

        # total loss
        loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["domain_loss_wt"] * domain_loss + \
               self.hparams["cond_ent_wt"] * loss_trg_cent

        # update feature extractor
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                'cond_ent_loss': loss_trg_cent.item()}


class DIRT(Algorithm):
    """
    DIRT-T: https://arxiv.org/abs/1802.08735
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(DIRT, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.domain_classifier = Discriminator(configs)

        # optimizers
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        # hparams
        self.hparams = hparams

        # criterion
        self.criterion_cond = ConditionalEntropyLoss().to(device)
        self.vat_loss = VAT(self.network, device).to(device)

        # device for further usage
        self.device = device

    def update(self, src_x, src_y, trg_x):
        # prepare true domain labels
        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        # target features and predictions
        trg_feat = self.feature_extractor(trg_x)
        trg_pred = self.classifier(trg_feat)

        # concatenate features and predictions
        feat_concat = torch.cat((src_feat, trg_feat), dim=0)

        # Domain classification loss
        disc_prediction = self.domain_classifier(feat_concat.detach())
        disc_loss = self.cross_entropy(disc_prediction, domain_label_concat)

        # update Domain classification
        self.optimizer_disc.zero_grad()
        disc_loss.backward()
        self.optimizer_disc.step()

        # prepare fake domain labels for training the feature extractor
        domain_label_src = torch.zeros(len(src_x)).long().to(self.device)
        domain_label_trg = torch.ones(len(trg_x)).long().to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

        # Repeat predictions after updating discriminator
        disc_prediction = self.domain_classifier(feat_concat)

        # loss of domain discriminator according to fake labels
        domain_loss = self.cross_entropy(disc_prediction, domain_label_concat)

        # Task classification  Loss
        src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

        # conditional entropy loss.
        loss_trg_cent = self.criterion_cond(trg_pred)

        # Virual advariarial training loss
        loss_src_vat = self.vat_loss(src_x, src_pred)
        loss_trg_vat = self.vat_loss(trg_x, trg_pred)
        total_vat = loss_src_vat + loss_trg_vat
        # total loss
        loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["domain_loss_wt"] * domain_loss + \
               self.hparams["cond_ent_wt"] * loss_trg_cent + self.hparams["vat_loss_wt"] * total_vat

        # # update exponential moving average
        # self.ema(self.network)

        # update feature extractor
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                'cond_ent_loss': loss_trg_cent.item()}


class DSAN(Algorithm):
    """
    DSAN: https://ieeexplore.ieee.org/document/9085896
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(DSAN, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams
        self.device = device
        self.loss_LMMD = LMMD_loss(device=device, class_num=configs.num_classes).to(device)

    def update(self, src_x, src_y, trg_x):
        # extract source features
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        # extract target features
        trg_feat = self.feature_extractor(trg_x)
        trg_pred = self.classifier(trg_feat)

        # calculate lmmd loss
        domain_loss = self.loss_LMMD.get_loss(src_feat, trg_feat, src_y, torch.nn.functional.softmax(trg_pred, dim=1))

        # calculate source classification loss
        src_cls_loss = self.cross_entropy(src_pred, src_y)

        # calculate the total loss
        loss = self.hparams["domain_loss_wt"] * domain_loss + \
               self.hparams["src_cls_loss_wt"] * src_cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'LMMD_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


class HoMM(Algorithm):
    """
    HoMM: https://arxiv.org/pdf/1912.11976.pdf
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(HoMM, self).__init__(configs)

        self.coral = CORAL()

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams
        self.device = device
        self.HoMM_loss = HoMM_loss()

    def update(self, src_x, src_y, trg_x):
        # extract source features
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        # extract target features
        trg_feat = self.feature_extractor(trg_x)
        trg_pred = self.classifier(trg_feat)

        # calculate source classification loss
        src_cls_loss = self.cross_entropy(src_pred, src_y)

        # calculate lmmd loss
        domain_loss = self.HoMM_loss(src_feat, trg_feat)

        # calculate the total loss
        loss = self.hparams["domain_loss_wt"] * domain_loss + \
               self.hparams["src_cls_loss_wt"] * src_cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'HoMM_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


class DDC(Algorithm):
    """
    DDC: https://arxiv.org/abs/1412.3474
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(DDC, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams
        self.device = device
        self.mmd_loss = MMD_loss()

    def update(self, src_x, src_y, trg_x):
        # extract source features
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        # extract target features
        trg_feat = self.feature_extractor(trg_x)

        # calculate source classification loss
        src_cls_loss = self.cross_entropy(src_pred, src_y)

        # calculate mmd loss
        domain_loss = self.mmd_loss(src_feat, trg_feat)

        # calculate the total loss
        loss = self.hparams["domain_loss_wt"] * domain_loss + \
               self.hparams["src_cls_loss_wt"] * src_cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'MMD_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


class CoDATS(Algorithm):
    """
    CoDATS: https://arxiv.org/pdf/2005.10996.pdf
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(CoDATS, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        # we replace the original classifier with codats the classifier
        # remember to use same name of self.classifier, as we use it for the model evaluation
        self.classifier = codats_classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.domain_classifier = Discriminator(configs)

        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + list(self.classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )
        self.hparams = hparams
        self.device = device

    def update(self, src_x, src_y, trg_x, step, epoch, len_dataloader):
        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # zero grad
        self.optimizer.zero_grad()
        self.optimizer_disc.zero_grad()

        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        trg_feat = self.feature_extractor(trg_x)

        # Task classification  Loss
        src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

        # Domain classification loss
        # source
        src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
        src_domain_pred = self.domain_classifier(src_feat_reversed)
        src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

        # target
        trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
        trg_domain_pred = self.domain_classifier(trg_feat_reversed)
        trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

        # Total domain loss
        domain_loss = src_domain_loss + trg_domain_loss

        loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
               self.hparams["domain_loss_wt"] * domain_loss

        loss.backward()
        self.optimizer.step()
        self.optimizer_disc.step()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


class AdvSKM(Algorithm):
    """
    AdvSKM: https://www.ijcai.org/proceedings/2021/0378.pdf
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(AdvSKM, self).__init__(configs)
        self.AdvSKM_embedder = AdvSKM_Disc(configs).to(device)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_disc = torch.optim.Adam(
            self.AdvSKM_embedder.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.hparams = hparams
        self.device = device
        self.mmd_loss = MMD_loss()

    def update(self, src_x, src_y, trg_x):
        # extract source features
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        # extract target features
        trg_feat = self.feature_extractor(trg_x)

        source_embedding_disc = self.AdvSKM_embedder(src_feat.detach())
        target_embedding_disc = self.AdvSKM_embedder(trg_feat.detach())
        mmd_loss = - self.mmd_loss(source_embedding_disc, target_embedding_disc)
        mmd_loss.requires_grad = True

        # update discriminator
        self.optimizer_disc.zero_grad()
        mmd_loss.backward()
        self.optimizer_disc.step()

        # calculate source classification loss
        src_cls_loss = self.cross_entropy(src_pred, src_y)

        # domain loss.
        source_embedding_disc = self.AdvSKM_embedder(src_feat)
        target_embedding_disc = self.AdvSKM_embedder(trg_feat)

        mmd_loss_adv = self.mmd_loss(source_embedding_disc, target_embedding_disc)
        mmd_loss_adv.requires_grad = True

        # calculate the total loss
        loss = self.hparams["domain_loss_wt"] * mmd_loss_adv + \
               self.hparams["src_cls_loss_wt"] * src_cls_loss

        # update optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Domain_loss': mmd_loss_adv.item(), 'Src_cls_loss': src_cls_loss.item()}

class SepRepTranAlignEnd(Algorithm):
    """
    Separate representations for each chanell. Multihead attention to combine them. Loss applied on combined term
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(SepRepTranAlignEnd, self).__init__(configs)

        true_final_out_channels = configs.true_final_out_channels
        configs.final_out_channels = true_final_out_channels
        self.true_input_channel = configs.input_channels
        self.feature_extractor = SepReps_with_multihead(configs,backbone_fe)
        configs.input_channels = self.true_input_channel
        configs.final_out_channels = true_final_out_channels*3
        self.classifier = classifier(configs)
        self.domain_classifier = Discriminator(configs)

        self.classifier_list = nn.ModuleList([])
        #for k in range(0,)
        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters())  +\
       list(self.classifier.parameters()),
            lr=1 * hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99))

        self.optimizer_disc = torch.optim.Adam(
            list(self.domain_classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99))

        self.hparams = hparams
        #self.feature_extractor.to(device)
        #self.domain_classifier.to(device)
        #self.classifier.to(device)
        #self.multihead.to(device)
        #self.multihead_ff.to(device)
        self.device = device
        self.dist = SinkhornDistance(eps=1e-4,max_iter=1000)


    def update(self, src_x, src_y, trg_x,step,epoch,len_dataloader):
        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        total_loss = 0
        chnl_output_src = []
        chnl_output_trg = []

        self.optimizer.zero_grad()
        self.optimizer_disc.zero_grad()



        src_feat_all = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat_all)
        src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)
        trg_feat_all = self.feature_extractor(trg_x)
        #src_feat_reversed = ReverseLayerF.apply(src_feat_all, alpha)

        #src_domain_pred = self.domain_classifier(src_feat_reversed)
        #src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

        # target
        #trg_feat_reversed = ReverseLayerF.apply(trg_feat_all , alpha)
        #trg_domain_pred = self.domain_classifier(trg_feat_reversed)
        #trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

        # Total domain loss
        #domain_loss = src_domain_loss + trg_domain_loss
        domain_loss = self.dist(src_feat_all,trg_feat_all)[0]
        loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
                   self.hparams["domain_loss_wt"] * domain_loss

        loss.backward()
        self.optimizer.step()
        self.optimizer_disc.step()
        total_loss = loss.item()

        return {'Total_loss': total_loss,'Src_cls_loss':src_cls_loss.item(),'Domain_loss':domain_loss.item()}

    def eval_update(self, src,trg):
        joint_loaders = enumerate(zip(src, trg))
        len_dataloader = min(len(src), len(trg))
        src_loss_list = []
        dom_loss_list = []
        tloss_list = []
        with torch.no_grad():
            for step, ((src_x, src_y), (trg_x, trg_y)) in joint_loaders:
                src_x, src_y, trg_x, trg_y = src_x.float().to(self.device), src_y.long().to(self.device), \
                                             trg_x.float().to(self.device), trg_y.to(self.device)

                self.optimizer.zero_grad()
                self.optimizer_disc.zero_grad()
                domain_label_src = torch.ones(len(src_x)).to(self.device)
                domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

                src_feat_all = self.feature_extractor(src_x)
                src_pred = self.classifier(src_feat_all)
                src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)
                trg_feat_all = self.feature_extractor(trg_x)
                # src_feat_reversed = ReverseLayerF.apply(src_feat_all, alpha)

                # src_domain_pred = self.domain_classifier(src_feat_reversed)
                # src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

                # target
                # trg_feat_reversed = ReverseLayerF.apply(trg_feat_all , alpha)
                # trg_domain_pred = self.domain_classifier(trg_feat_reversed)
                # trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

                # Total domain loss
                # domain_loss = src_domain_loss + trg_domain_loss
                domain_loss = self.dist(src_feat_all, trg_feat_all)[0]
                loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
                       self.hparams["domain_loss_wt"] * domain_loss
                dom_loss_list.append(domain_loss.item())
                src_loss_list.append(src_cls_loss.item())
                tloss_list.append(loss.item())

        # self.ema.update()
        return {'Total_loss': np.mean(tloss_list), 'Src_cls_loss':np.mean(src_loss_list), 'Domain_loss': np.mean(dom_loss_list)}

class SepAligThenAttn(Algorithm):
    """
    Separate representations for each chanell. Multihead attention to combine them. Loss applied on combined term. Uses DANN
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(SepAligThenAttn, self).__init__(configs)
        true_final_out_channels = configs.true_final_out_channels
        self.true_input_channel = configs.input_channels
        configs.final_out_channels = true_final_out_channels
        self.feature_extractor = SepReps_with_multihead(configs,backbone_fe)


        self.classifier_list_ind = nn.ModuleList([])
        self.domain_classifier_list_ind = nn.ModuleList([])
        for k in range(0,self.true_input_channel):
            self.classifier_list_ind.append(classifier(configs))

        for k in range(0,self.true_input_channel ):
            self.domain_classifier_list_ind.append(Discriminator(configs))
            #for k in range(0,)
        self.optimizer_ind = torch.optim.Adam(
            list(self.feature_extractor.backbone_nets.parameters())  +\
       list(self.classifier_list_ind.parameters()),
            lr=1 * hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99))

        self.optimizer_disc_ind = torch.optim.Adam(
            list(self.domain_classifier_list_ind.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99))

        configs.input_channels = self.true_input_channel
        configs.final_out_channels = true_final_out_channels* self.true_input_channel

        self.classifier = classifier(configs)
        self.domain_classifier = Discriminator(configs)
        #self.optimizer_comb = torch.optim.Adam(list(self.feature_extractor.multihead_attention.parameters())+
        #                                       list(self.classifier.parameters()),lr=1 * hparams["learning_rate"],
        #    weight_decay=hparams["weight_decay"], betas=(0.5, 0.99))


        self.optimizer_comb = torch.optim.Adam(list(self.feature_extractor.backbone_nets.parameters()) + \
                                               list(self.classifier_list_ind.parameters()) + list(
            self.feature_extractor.multihead_attention.parameters()) +
                                               list(self.classifier.parameters()), lr=1 * hparams["learning_rate"],
                                               weight_decay=hparams["weight_decay"], betas=(0.5, 0.99))
        self.optimizer_disc_comb = torch.optim.Adam(list(self.domain_classifier_list_ind.parameters())+list(self.domain_classifier.parameters())
                                               ,lr=1 * hparams["learning_rate"], weight_decay=hparams["weight_decay"], betas=(0.5, 0.99))
        self.hparams = hparams
        self.feature_extractor.to(device)
        self.domain_classifier.to(device)
        self.classifier.to(device)
        self.domain_classifier_list_ind.to(device)
        self.classifier_list_ind.to(device)
        #self.multihead.to(device)
        #self.multihead_ff.to(device)
        self.device = device



    def update(self, src_x, src_y, trg_x,step,epoch,len_dataloader):
        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        total_loss = 0
        chnl_output_src = []
        chnl_output_trg = []

        #self.optimizer_disc_ind.zero_grad()
        #self.optimizer_ind.zero_grad()
        self.optimizer_comb.zero_grad()
        self.optimizer_disc_comb.zero_grad()

        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)


        src_reps_list_chnl = self.feature_extractor.fetch_individual_reps(src_x)

        clfr_src_list = 0
        for k in range(0,self.true_input_channel):
            clfr_k_pred = self.classifier_list_ind[k](src_reps_list_chnl[k])
            clfr_src_list = clfr_src_list + (self.cross_entropy( clfr_k_pred.squeeze(), src_y))

        dom_clfr_src_list = 0
        for k in range(0, self.true_input_channel):
            src_rep_reversed_k = ReverseLayerF.apply(src_reps_list_chnl[k], alpha)
            dom_clfr_k_pred_src = self.domain_classifier_list_ind[k](src_rep_reversed_k)
            dom_clfr_src_list = dom_clfr_src_list  + (self.cross_entropy(dom_clfr_k_pred_src, domain_label_src.long()))

        trg_reps_list_chnl = self.feature_extractor.fetch_individual_reps(trg_x)

        dom_clfr_trg_list = 0

        for k in range(0, self.true_input_channel):
            trg_rep_reversed_k = ReverseLayerF.apply(trg_reps_list_chnl[k], alpha)
            dom_clfr_k_pred_trg = self.domain_classifier_list_ind[k](trg_rep_reversed_k)
            dom_clfr_trg_list = dom_clfr_src_list  + (self.cross_entropy(dom_clfr_k_pred_trg.squeeze(), domain_label_trg.long()))

        domain_loss_ind = torch.sum(dom_clfr_src_list) + torch.sum(dom_clfr_trg_list)
        loss_ind = self.hparams["src_cls_loss_wt"] * torch.sum(clfr_src_list) + \
               self.hparams["domain_loss_wt"] * domain_loss_ind
        #loss_ind.backward()
        #self.optimizer_ind.step()
        #self.optimizer_disc_ind.step()
        #loss_ind_total = loss_ind.item()

        #self.optimizer_comb.zero_grad()
        #self.optimizer_disc_comb.zero_grad()
        #self.optimizer_disc_ind.zero_grad()
        #self.optimizer_ind.zero_grad()



        comb_reps_src,src_attn = self.feature_extractor.combine_ind_through_attn(src_reps_list_chnl)
        clfr_pred_comb = self.classifier(comb_reps_src)
        loss_sup_src = self.cross_entropy(clfr_pred_comb.squeeze(),src_y)

        comb_src_rev = ReverseLayerF.apply(comb_reps_src, alpha)
        dom_clfr_src_com = self.domain_classifier(comb_src_rev)
        dom_clfr_src_loss = self.cross_entropy(dom_clfr_src_com.squeeze(),domain_label_src.long())

        comb_reps_trg,trg_attn = self.feature_extractor.combine_ind_through_attn(trg_reps_list_chnl)
        dom_clfr_trg_com = self.domain_classifier(comb_reps_trg)
        dom_clfr_trg_loss = self.cross_entropy(dom_clfr_trg_com.squeeze(), domain_label_trg.long())


        domain_loss = dom_clfr_src_loss + dom_clfr_trg_loss





        loss_comb = self.hparams["src_cls_loss_wt"] * loss_sup_src + \
                   self.hparams["domain_loss_wt"] * domain_loss+1*loss_ind

        loss_comb.backward()
        self.optimizer_comb.step()
        self.optimizer_disc_comb.step()
        loss_comb_total = loss_comb.item()

        #reps_src_per_c = self.feature_extractor.fetch_individual_reps(src_x)
        #reps_trg_per_c = self.feature_extractor.fetch_individual_reps(trg_x)

       # for k in range(0,3):







    # self.ema.update()
        return {'Total_loss': loss_comb_total,'Src_cls_loss':loss_sup_src.item(),'Domain_loss':domain_loss.item()}


    def get_ind_scores(self,x):
        self.feature_extractor.eval()
        self.classifier_list_ind.eval()

        pred_prob_list =[]
        pred_list =[]
        with torch.no_grad():
            src_reps_list_chnl = self.feature_extractor.fetch_individual_reps(x)
            for k in range(0, self.true_input_channel):
                pred_prob_list.append(self.classifier_list_ind[k](src_reps_list_chnl[k]).detach().cpu())
                pred_list.append(self.classifier_list_ind[k](src_reps_list_chnl[k].detach().cpu()).argmax(dim=1))
        return pred_prob_list,pred_list


    def eval_update(self, src,trg):
        joint_loaders = enumerate(zip(src, trg))
        len_dataloader = min(len(src), len(trg))
        src_loss_list = []
        dom_loss_list = []
        tloss_list = []
        with torch.no_grad():
            for step, ((src_x, src_y), (trg_x, trg_y)) in joint_loaders:
                src_x, src_y, trg_x, trg_y = src_x.float().to(self.device), src_y.long().to(self.device), \
                                             trg_x.float().to(self.device), trg_y.to(self.device)


                domain_label_src = torch.ones(len(src_x)).to(self.device)
                domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

                src_reps_list_chnl = self.feature_extractor.fetch_individual_reps(src_x)

                clfr_src_list = 0
                for k in range(0, self.true_input_channel):
                    clfr_k_pred = self.classifier_list_ind[k](src_reps_list_chnl[k])
                    clfr_src_list = clfr_src_list + (self.cross_entropy(clfr_k_pred.squeeze(), src_y))

                dom_clfr_src_list = 0
                for k in range(0, self.true_input_channel):

                    dom_clfr_k_pred_src = self.domain_classifier_list_ind[k](src_reps_list_chnl[k])
                    dom_clfr_src_list = dom_clfr_src_list + (
                        self.cross_entropy(dom_clfr_k_pred_src, domain_label_src.long()))

                trg_reps_list_chnl = self.feature_extractor.fetch_individual_reps(trg_x)

                dom_clfr_trg_list = 0

                for k in range(0, self.true_input_channel):

                    dom_clfr_k_pred_trg = self.domain_classifier_list_ind[k](trg_reps_list_chnl[k])
                    dom_clfr_trg_list = dom_clfr_src_list + (
                        self.cross_entropy(dom_clfr_k_pred_trg.squeeze(), domain_label_trg.long()))

                domain_loss_ind = torch.sum(dom_clfr_src_list) + torch.sum(dom_clfr_trg_list)
                loss_ind = self.hparams["src_cls_loss_wt"] * torch.sum(clfr_src_list) + \
                           self.hparams["domain_loss_wt"] * domain_loss_ind


                comb_reps_src = self.feature_extractor.combine_ind_through_attn(src_reps_list_chnl)
                clfr_pred_comb = self.classifier(comb_reps_src)
                loss_sup_src = self.cross_entropy(clfr_pred_comb.squeeze(), src_y)


                dom_clfr_src_com = self.domain_classifier(comb_reps_src)
                dom_clfr_src_loss = self.cross_entropy(dom_clfr_src_com.squeeze(), domain_label_src.long())

                comb_reps_trg = self.feature_extractor.combine_ind_through_attn(trg_reps_list_chnl)
                dom_clfr_trg_com = self.domain_classifier(comb_reps_trg)
                dom_clfr_trg_loss = self.cross_entropy(dom_clfr_trg_com.squeeze(), domain_label_trg.long())

                domain_loss = dom_clfr_src_loss + dom_clfr_trg_loss

                loss_comb = self.hparams["src_cls_loss_wt"] * loss_sup_src + \
                            self.hparams["domain_loss_wt"] * domain_loss

                dom_loss_list.append(domain_loss.item())
                src_loss_list.append(loss_sup_src.item())
                tloss_list.append(loss_comb.item())

        # self.ema.update()
        return {'Total_loss': np.mean(tloss_list), 'Src_cls_loss':np.mean(src_loss_list), 'Domain_loss': np.mean(dom_loss_list)}



class SepAligThenAttnSink(Algorithm):
    """
    Separate representations for each chanell. Multihead attention to combine them. Loss applied on combined term
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(SepAligThenAttnSink, self).__init__(configs)
        true_final_out_channels = configs.true_final_out_channels
        configs.final_out_channels = true_final_out_channels
        self.true_input_channel = configs.input_channels
        self.feature_extractor = SepReps_with_multihead(configs,backbone_fe)


        self.classifier_list_ind = nn.ModuleList([])
        self.domain_classifier_list_ind = nn.ModuleList([])
        for k in range(0,self.true_input_channel):
            self.classifier_list_ind.append(classifier(configs))


        self.optimizer_ind = torch.optim.Adam(
            list(self.feature_extractor.backbone_nets.parameters())  +\
       list(self.classifier_list_ind.parameters()),
            lr=1 * hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99))



        configs.input_channels = self.true_input_channel
        configs.final_out_channels = true_final_out_channels * self.true_input_channel

        self.classifier = classifier(configs)

        self.optimizer_comb = torch.optim.Adam(list(self.feature_extractor.backbone_nets.parameters())  +\
       list(self.classifier_list_ind.parameters()) +list(self.feature_extractor.multihead_attention.parameters())+
                                               list(self.classifier.parameters()),lr=1 * hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99))

        self.hparams = hparams
        self.feature_extractor.to(device)

        self.classifier.to(device)

        self.classifier_list_ind.to(device)
        #self.multihead.to(device)
        #self.multihead_ff.to(device)
        self.device = device

        self.sink = SinkhornDistance(eps=1e-3, max_iter=1000, reduction='sum')
        #self.sink = SinkhornDistance(eps=1e-1, max_iter=1000, reduction='sum')

    def update(self, src_x, src_y, trg_x,trg_y,step,epoch,len_dataloader):


        #self.optimizer_ind.zero_grad()
        self.optimizer_comb.zero_grad()





        src_reps_list_chnl = self.feature_extractor.fetch_individual_reps(src_x)

        clfr_src_list = 0
        for k in range(0,self.true_input_channel):
            clfr_k_pred = self.classifier_list_ind[k](src_reps_list_chnl[k])
            clfr_src_list = clfr_src_list + (self.cross_entropy( clfr_k_pred.squeeze(), src_y))


        trg_reps_list_chnl = self.feature_extractor.fetch_individual_reps(trg_x)
        align_loss = 0
        for k in range(0, self.true_input_channel):
            #align_loss_k = self.sink(src_reps_list_chnl[k], trg_reps_list_chnl[k])[0]
            align_loss_k =  + self.sink(src_reps_list_chnl[k], trg_reps_list_chnl[k])[0]
            #clfr_k_pred = self.classifier_list_ind[k](src_reps_list_chnl[k])

            #idx_src = torch.where(src_y == 3)[0]
            #idx_src = torch.where(trg_y == 3)[0]
            #map = self.sink(src_reps_list_chnl[k][idx_src, :], trg_reps_list_chnl[k])[1].detach().cpu()
            #P = torch.cdist(src_x[idx_src,k,:],trg_x[:,k,:]).detach().cpu()
            #torch.argmax(clfr_k_pred, dim=-1)[idx_src]
            #clfr_loss = self.cross_entropy( clfr_k_pred.squeeze(), src_y)
            #chnl_loss = clfr_loss +align_loss_k
            #chnl_loss.backward(retain_graph=True)
            align_loss = align_loss + align_loss_k





        domain_loss_ind = 1*torch.sum(clfr_src_list ) + torch.sum(align_loss  )
        loss_ind = self.hparams["src_cls_loss_wt"] * torch.sum(clfr_src_list) + \
              1* self.hparams["domain_loss_wt"] * domain_loss_ind
        #loss_ind.backward()
        #self.optimizer_ind.step()

        #loss_ind_total = loss_ind.item()

        #self.optimizer_comb.zero_grad()

        #self.optimizer_ind.zero_grad()



        comb_reps_src,src_comb_attn = self.feature_extractor.combine_ind_through_attn(src_reps_list_chnl)
        clfr_pred_comb = self.classifier(comb_reps_src)
        loss_sup_src = self.cross_entropy(clfr_pred_comb.squeeze(),src_y)



        comb_reps_trg,trg_comb_attn  = self.feature_extractor.combine_ind_through_attn(trg_reps_list_chnl)


        domain_loss = self.sink(comb_reps_src,comb_reps_trg)[0]# +self.sink(comb_reps_src,comb_reps_trg)[0]

        if epoch> 0:

            loss_comb = 1*(self.hparams["src_cls_loss_wt"] * loss_sup_src + \
                       self.hparams["domain_loss_wt"]*1 * domain_loss) + 1*loss_ind
        else:
            loss_comb = 0

        loss_comb.backward()
        self.optimizer_comb.step()

        loss_comb_total = loss_comb.item()

        #reps_src_per_c = self.feature_extractor.fetch_individual_reps(src_x)
        #reps_trg_per_c = self.feature_extractor.fetch_individual_reps(trg_x)

       # for k in range(0,3):


    # self.ema.update()
        return {'Total_loss': loss_comb_total,'Src_cls_loss':loss_sup_src.item(),'Domain_loss':domain_loss.item()}


    def get_ind_scores(self,x):
        #self.feature_extractor.eval()
        #self.classifier_list_ind.eval()

        pred_prob_list =[]
        pred_list =[]
        with torch.no_grad():
            src_reps_list_chnl = self.feature_extractor.fetch_individual_reps(x)
            for k in range(0, self.true_input_channel):
                pred_prob_list.append(self.classifier_list_ind[k](src_reps_list_chnl[k]).detach().cpu())
                pred_list.append(self.classifier_list_ind[k](src_reps_list_chnl[k]).argmax(dim=1).detach().cpu())
        return pred_prob_list,pred_list


    def eval_update(self, src,trg):
        joint_loaders = enumerate(zip(src, trg))
        len_dataloader = min(len(src), len(trg))
        src_loss_list = []
        dom_loss_list = []
        tloss_list = []
        with torch.no_grad():
            for step, ((src_x, src_y), (trg_x, trg_y)) in joint_loaders:
                src_x, src_y, trg_x, trg_y = src_x.float().to(self.device), src_y.long().to(self.device), \
                                             trg_x.float().to(self.device), trg_y.to(self.device)



                domain_label_src = torch.ones(len(src_x)).to(self.device)
                domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

                src_reps_list_chnl = self.feature_extractor.fetch_individual_reps(src_x)

                clfr_src_list = 0
                for k in range(0, self.true_input_channel):
                    clfr_k_pred = self.classifier_list_ind[k](src_reps_list_chnl[k])
                    clfr_src_list = clfr_src_list + (self.cross_entropy(clfr_k_pred.squeeze(), src_y))
                trg_reps_list_chnl = self.feature_extractor.fetch_individual_reps(trg_x)
                align_loss = 0
                for k in range(0, self.true_input_channel):
                    align_loss_k = self.sink(src_reps_list_chnl[k], trg_reps_list_chnl[k])

                    align_loss = align_loss + align_loss_k[0]

                domain_loss_ind = torch.sum(clfr_src_list) + torch.sum(align_loss)
                loss_ind = self.hparams["src_cls_loss_wt"] * torch.sum(clfr_src_list) + \
                           self.hparams["domain_loss_wt"] * domain_loss_ind


                comb_reps_src = self.feature_extractor.combine_ind_through_attn(src_reps_list_chnl)
                clfr_pred_comb = self.classifier(comb_reps_src)
                loss_sup_src = self.cross_entropy(clfr_pred_comb.squeeze(), src_y)

                comb_reps_trg = self.feature_extractor.combine_ind_through_attn(trg_reps_list_chnl)

                domain_loss = self.sink(comb_reps_src, comb_reps_trg)[0]

                loss_comb = self.hparams["src_cls_loss_wt"] * loss_sup_src + \
                            self.hparams["domain_loss_wt"] * domain_loss + 0.3*loss_ind

                dom_loss_list.append(domain_loss.item())
                src_loss_list.append(loss_sup_src.item())
                tloss_list.append(loss_comb.item())

        # self.ema.update()
        return {'Total_loss': np.mean(tloss_list), 'Src_cls_loss':np.mean(src_loss_list), 'Domain_loss': np.mean(dom_loss_list)}

class NoSepAligThenAttnSink(Algorithm):
    """
    Separate representations for each chanell. Multihead attention to combine them. Loss applied on combined term
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(NoSepAligThenAttnSink, self).__init__(configs)
        true_final_out_channels = configs.true_final_out_channels
        configs.final_out_channels = true_final_out_channels
        self.true_input_channel = configs.input_channels
        self.feature_extractor = SepReps_with_multihead(configs,backbone_fe)


        self.classifier_list_ind = nn.ModuleList([])
        self.domain_classifier_list_ind = nn.ModuleList([])
        for k in range(0,self.true_input_channel):
            self.classifier_list_ind.append(classifier(configs))


        self.optimizer_ind = torch.optim.Adam(
            list(self.feature_extractor.backbone_nets.parameters())  +\
       list(self.classifier_list_ind.parameters()),
            lr=1 * hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99))



        configs.input_channels = self.true_input_channel
        configs.final_out_channels = true_final_out_channels * self.true_input_channel

        self.classifier = classifier(configs)

        self.optimizer_comb = torch.optim.Adam(list(self.feature_extractor.backbone_nets.parameters())  +\
       list(self.classifier_list_ind.parameters()) +list(self.feature_extractor.multihead_attention.parameters())+
                                               list(self.classifier.parameters()),lr=1 * hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99))

        self.hparams = hparams
        self.feature_extractor.to(device)

        self.classifier.to(device)

        self.classifier_list_ind.to(device)
        #self.multihead.to(device)
        #self.multihead_ff.to(device)
        self.device = device

        self.sink = SinkhornDistance(eps=1e-3, max_iter=1000, reduction='sum')
        #self.sink = SinkhornDistance(eps=1e-1, max_iter=1000, reduction='sum')

    def update(self, src_x, src_y, trg_x,trg_y,step,epoch,len_dataloader):


        #self.optimizer_ind.zero_grad()
        self.optimizer_comb.zero_grad()





        src_reps_list_chnl = self.feature_extractor.fetch_individual_reps(src_x)

        clfr_src_list = 0
        for k in range(0,self.true_input_channel):
            clfr_k_pred = self.classifier_list_ind[k](src_reps_list_chnl[k])
            clfr_src_list = clfr_src_list + (self.cross_entropy( clfr_k_pred.squeeze(), src_y))


        trg_reps_list_chnl = self.feature_extractor.fetch_individual_reps(trg_x)
        align_loss = 0
        for k in range(0, self.true_input_channel):
            #align_loss_k = self.sink(src_reps_list_chnl[k], trg_reps_list_chnl[k])[0]
            align_loss_k =  + self.sink(src_reps_list_chnl[k], trg_reps_list_chnl[k])[0]
            #clfr_k_pred = self.classifier_list_ind[k](src_reps_list_chnl[k])

            #idx_src = torch.where(src_y == 3)[0]
            #idx_src = torch.where(trg_y == 3)[0]
            #map = self.sink(src_reps_list_chnl[k][idx_src, :], trg_reps_list_chnl[k])[1].detach().cpu()
            #P = torch.cdist(src_x[idx_src,k,:],trg_x[:,k,:]).detach().cpu()
            #torch.argmax(clfr_k_pred, dim=-1)[idx_src]
            #clfr_loss = self.cross_entropy( clfr_k_pred.squeeze(), src_y)
            #chnl_loss = clfr_loss +align_loss_k
            #chnl_loss.backward(retain_graph=True)
            align_loss = align_loss + align_loss_k






        #loss_ind.backward()
        #self.optimizer_ind.step()

        #loss_ind_total = loss_ind.item()

        #self.optimizer_comb.zero_grad()

        #self.optimizer_ind.zero_grad()



        comb_reps_src,src_comb_attn = self.feature_extractor.combine_ind_through_attn(src_reps_list_chnl)
        clfr_pred_comb = self.classifier(comb_reps_src)
        loss_sup_src = self.cross_entropy(clfr_pred_comb.squeeze(),src_y)



        comb_reps_trg,trg_comb_attn  = self.feature_extractor.combine_ind_through_attn(trg_reps_list_chnl)


        domain_loss = self.sink(comb_reps_src,comb_reps_trg)[0]# +self.sink(comb_reps_src,comb_reps_trg)[0]

        if epoch> 0:

            loss_comb = 1*(self.hparams["src_cls_loss_wt"] * loss_sup_src + \
                       self.hparams["domain_loss_wt"]*1 * domain_loss)
        else:
            loss_comb = 0

        loss_comb.backward()
        self.optimizer_comb.step()

        loss_comb_total = loss_comb.item()

        #reps_src_per_c = self.feature_extractor.fetch_individual_reps(src_x)
        #reps_trg_per_c = self.feature_extractor.fetch_individual_reps(trg_x)

       # for k in range(0,3):


    # self.ema.update()
        return {'Total_loss': loss_comb_total,'Src_cls_loss':loss_sup_src.item(),'Domain_loss':domain_loss.item()}


    def get_ind_scores(self,x):
        #self.feature_extractor.eval()
        #self.classifier_list_ind.eval()

        pred_prob_list =[]
        pred_list =[]
        with torch.no_grad():
            src_reps_list_chnl = self.feature_extractor.fetch_individual_reps(x)
            for k in range(0, self.true_input_channel):
                pred_prob_list.append(self.classifier_list_ind[k](src_reps_list_chnl[k]).detach().cpu())
                pred_list.append(self.classifier_list_ind[k](src_reps_list_chnl[k]).argmax(dim=1).detach().cpu())
        return pred_prob_list,pred_list


    def eval_update(self, src,trg):
        joint_loaders = enumerate(zip(src, trg))
        len_dataloader = min(len(src), len(trg))
        src_loss_list = []
        dom_loss_list = []
        tloss_list = []
        with torch.no_grad():
            for step, ((src_x, src_y), (trg_x, trg_y)) in joint_loaders:
                src_x, src_y, trg_x, trg_y = src_x.float().to(self.device), src_y.long().to(self.device), \
                                             trg_x.float().to(self.device), trg_y.to(self.device)



                domain_label_src = torch.ones(len(src_x)).to(self.device)
                domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

                src_reps_list_chnl = self.feature_extractor.fetch_individual_reps(src_x)

                clfr_src_list = 0
                for k in range(0, self.true_input_channel):
                    clfr_k_pred = self.classifier_list_ind[k](src_reps_list_chnl[k])
                    clfr_src_list = clfr_src_list + (self.cross_entropy(clfr_k_pred.squeeze(), src_y))
                trg_reps_list_chnl = self.feature_extractor.fetch_individual_reps(trg_x)
                align_loss = 0
                for k in range(0, self.true_input_channel):
                    align_loss_k = self.sink(src_reps_list_chnl[k], trg_reps_list_chnl[k])

                    align_loss = align_loss + align_loss_k[0]

                domain_loss_ind = torch.sum(clfr_src_list) + torch.sum(align_loss)
                loss_ind = self.hparams["src_cls_loss_wt"] * torch.sum(clfr_src_list) + \
                           self.hparams["domain_loss_wt"] * domain_loss_ind


                comb_reps_src = self.feature_extractor.combine_ind_through_attn(src_reps_list_chnl)
                clfr_pred_comb = self.classifier(comb_reps_src)
                loss_sup_src = self.cross_entropy(clfr_pred_comb.squeeze(), src_y)

                comb_reps_trg = self.feature_extractor.combine_ind_through_attn(trg_reps_list_chnl)

                domain_loss = self.sink(comb_reps_src, comb_reps_trg)[0]

                loss_comb = self.hparams["src_cls_loss_wt"] * loss_sup_src + \
                            self.hparams["domain_loss_wt"] * domain_loss + 0.3*loss_ind

                dom_loss_list.append(domain_loss.item())
                src_loss_list.append(loss_sup_src.item())
                tloss_list.append(loss_comb.item())

        # self.ema.update()
        return {'Total_loss': np.mean(tloss_list), 'Src_cls_loss':np.mean(src_loss_list), 'Domain_loss': np.mean(dom_loss_list)}
class SepAligThenNoAttnSink(Algorithm):
    """
    Separate representations for each chanell. Multihead attention to combine them. Loss applied on combined term
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(SepAligThenNoAttnSink, self).__init__(configs)
        true_final_out_channels = configs.true_final_out_channels
        configs.final_out_channels = true_final_out_channels
        self.true_input_channel = configs.input_channels
        self.feature_extractor = SepReps_with_multihead(configs,backbone_fe)


        self.classifier_list_ind = nn.ModuleList([])
        self.domain_classifier_list_ind = nn.ModuleList([])
        for k in range(0,self.true_input_channel):
            self.classifier_list_ind.append(classifier(configs))


        self.optimizer_ind = torch.optim.Adam(
            list(self.feature_extractor.backbone_nets.parameters())  +\
       list(self.classifier_list_ind.parameters()),
            lr=1 * hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99))



        configs.input_channels = self.true_input_channel
        configs.final_out_channels = true_final_out_channels * self.true_input_channel

        self.classifier = classifier(configs)

        self.optimizer_comb = torch.optim.Adam(list(self.feature_extractor.backbone_nets.parameters())  +\
       list(self.classifier_list_ind.parameters()) +list(self.feature_extractor.multihead_attention.parameters())+
                                               list(self.classifier.parameters()),lr=1 * hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99))

        self.hparams = hparams
        self.feature_extractor.to(device)

        self.classifier.to(device)

        self.classifier_list_ind.to(device)
        #self.multihead.to(device)
        #self.multihead_ff.to(device)
        self.device = device

        self.sink = SinkhornDistance(eps=1e-3, max_iter=1000, reduction='sum')
        #self.sink = SinkhornDistance(eps=1e-1, max_iter=1000, reduction='sum')

    def update(self, src_x, src_y, trg_x,trg_y,step,epoch,len_dataloader):


        #self.optimizer_ind.zero_grad()
        self.optimizer_comb.zero_grad()





        src_reps_list_chnl = self.feature_extractor.fetch_individual_reps(src_x)

        clfr_src_list = 0
        for k in range(0,self.true_input_channel):
            clfr_k_pred = self.classifier_list_ind[k](src_reps_list_chnl[k])
            clfr_src_list = clfr_src_list + (self.cross_entropy( clfr_k_pred.squeeze(), src_y))


        trg_reps_list_chnl = self.feature_extractor.fetch_individual_reps(trg_x)
        align_loss = 0
        for k in range(0, self.true_input_channel):
            #align_loss_k = self.sink(src_reps_list_chnl[k], trg_reps_list_chnl[k])[0]
            align_loss_k =  + self.sink(src_reps_list_chnl[k], trg_reps_list_chnl[k])[0]
            #clfr_k_pred = self.classifier_list_ind[k](src_reps_list_chnl[k])

            #idx_src = torch.where(src_y == 3)[0]
            #idx_src = torch.where(trg_y == 3)[0]
            #map = self.sink(src_reps_list_chnl[k][idx_src, :], trg_reps_list_chnl[k])[1].detach().cpu()
            #P = torch.cdist(src_x[idx_src,k,:],trg_x[:,k,:]).detach().cpu()
            #torch.argmax(clfr_k_pred, dim=-1)[idx_src]
            #clfr_loss = self.cross_entropy( clfr_k_pred.squeeze(), src_y)
            #chnl_loss = clfr_loss +align_loss_k
            #chnl_loss.backward(retain_graph=True)
            align_loss = align_loss + align_loss_k





        domain_loss_ind = 1*torch.sum(clfr_src_list ) + torch.sum(align_loss  )
        loss_ind = self.hparams["src_cls_loss_wt"] * torch.sum(clfr_src_list) + \
               self.hparams["domain_loss_wt"] * domain_loss_ind
        #loss_ind.backward()
        #self.optimizer_ind.step()

        #loss_ind_total = loss_ind.item()

        #self.optimizer_comb.zero_grad()

        #self.optimizer_ind.zero_grad()



        comb_reps_src = torch.hstack(src_reps_list_chnl)
        clfr_pred_comb = self.classifier(comb_reps_src)
        loss_sup_src = self.cross_entropy(clfr_pred_comb.squeeze(),src_y)



        comb_reps_trg = torch.hstack(trg_reps_list_chnl)


        domain_loss = self.sink(comb_reps_src,comb_reps_trg)[0]# +self.sink(comb_reps_src,comb_reps_trg)[0]

        if epoch> 0:

            loss_comb = 1*(self.hparams["src_cls_loss_wt"] * loss_sup_src + \
                       self.hparams["domain_loss_wt"]*1 * domain_loss) + 1*loss_ind
        else:
            loss_comb = 0

        loss_comb.backward()
        self.optimizer_comb.step()

        loss_comb_total = loss_comb.item()

        #reps_src_per_c = self.feature_extractor.fetch_individual_reps(src_x)
        #reps_trg_per_c = self.feature_extractor.fetch_individual_reps(trg_x)

       # for k in range(0,3):


    # self.ema.update()
        return {'Total_loss': loss_comb_total,'Src_cls_loss':loss_sup_src.item(),'Domain_loss':domain_loss.item()}


    def get_ind_scores(self,x):
        #self.feature_extractor.eval()
        #self.classifier_list_ind.eval()

        pred_prob_list =[]
        pred_list =[]
        with torch.no_grad():
            src_reps_list_chnl = self.feature_extractor.fetch_individual_reps(x)
            for k in range(0, self.true_input_channel):
                pred_prob_list.append(self.classifier_list_ind[k](src_reps_list_chnl[k]).detach().cpu())
                pred_list.append(self.classifier_list_ind[k](src_reps_list_chnl[k]).argmax(dim=1).detach().cpu())
        return pred_prob_list,pred_list


    def eval_update(self, src,trg):
        joint_loaders = enumerate(zip(src, trg))
        len_dataloader = min(len(src), len(trg))
        src_loss_list = []
        dom_loss_list = []
        tloss_list = []
        with torch.no_grad():
            for step, ((src_x, src_y), (trg_x, trg_y)) in joint_loaders:
                src_x, src_y, trg_x, trg_y = src_x.float().to(self.device), src_y.long().to(self.device), \
                                             trg_x.float().to(self.device), trg_y.to(self.device)



                domain_label_src = torch.ones(len(src_x)).to(self.device)
                domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

                src_reps_list_chnl = self.feature_extractor.fetch_individual_reps(src_x)

                clfr_src_list = 0
                for k in range(0, self.true_input_channel):
                    clfr_k_pred = self.classifier_list_ind[k](src_reps_list_chnl[k])
                    clfr_src_list = clfr_src_list + (self.cross_entropy(clfr_k_pred.squeeze(), src_y))
                trg_reps_list_chnl = self.feature_extractor.fetch_individual_reps(trg_x)
                align_loss = 0
                for k in range(0, self.true_input_channel):
                    align_loss_k = self.sink(src_reps_list_chnl[k], trg_reps_list_chnl[k])

                    align_loss = align_loss + align_loss_k[0]

                domain_loss_ind = torch.sum(clfr_src_list) + torch.sum(align_loss)
                loss_ind = self.hparams["src_cls_loss_wt"] * torch.sum(clfr_src_list) + \
                           self.hparams["domain_loss_wt"] * domain_loss_ind


                comb_reps_src = self.feature_extractor.combine_ind_through_attn(src_reps_list_chnl)
                clfr_pred_comb = self.classifier(comb_reps_src)
                loss_sup_src = self.cross_entropy(clfr_pred_comb.squeeze(), src_y)

                comb_reps_trg = self.feature_extractor.combine_ind_through_attn(trg_reps_list_chnl)

                domain_loss = self.sink(comb_reps_src, comb_reps_trg)[0]

                loss_comb = self.hparams["src_cls_loss_wt"] * loss_sup_src + \
                            self.hparams["domain_loss_wt"] * domain_loss + 0.3*loss_ind

                dom_loss_list.append(domain_loss.item())
                src_loss_list.append(loss_sup_src.item())
                tloss_list.append(loss_comb.item())

        # self.ema.update()
        return {'Total_loss': np.mean(tloss_list), 'Src_cls_loss':np.mean(src_loss_list), 'Domain_loss': np.mean(dom_loss_list)}
class SepAligThenSum(Algorithm):
    """
    Separate representations for each chanell. Multihead attention to combine them. Loss applied on combined term
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(SepAligThenSum, self).__init__(configs)
        true_final_out_channels = configs.true_final_out_channels
        configs.final_out_channels = true_final_out_channels
        self.true_input_channel = configs.input_channels
        self.feature_extractor = SepReps_with_sum(configs,backbone_fe)


        self.classifier_list_ind = nn.ModuleList([])
        self.domain_classifier_list_ind = nn.ModuleList([])
        for k in range(0,self.true_input_channel):
            self.classifier_list_ind.append(classifier(configs))


        self.optimizer_ind = torch.optim.Adam(
            list(self.feature_extractor.backbone_nets.parameters())  +\
       list(self.classifier_list_ind.parameters()),
            lr=1 * hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99))



        configs.input_channels = self.true_input_channel
        configs.final_out_channels = true_final_out_channels

        self.classifier = classifier(configs)

        self.optimizer_comb = torch.optim.Adam(list(self.feature_extractor.backbone_nets.parameters())  +\
       list(self.classifier_list_ind.parameters()) +
                                               list(self.classifier.parameters()),lr=1 * hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99))

        self.hparams = hparams
        self.feature_extractor.to(device)

        self.classifier.to(device)

        self.classifier_list_ind.to(device)
        #self.multihead.to(device)
        #self.multihead_ff.to(device)
        self.device = device

        self.sink = SinkhornDistance(eps=1e-3, max_iter=1000, reduction='sum')
        #self.sink = SinkhornDistance(eps=1e-1, max_iter=1000, reduction='sum')

    def update(self, src_x, src_y, trg_x,trg_y,step,epoch,len_dataloader):


        #self.optimizer_ind.zero_grad()
        self.optimizer_comb.zero_grad()
        self.feature_extractor.train()
        self.classifier_list_ind.train()
        self.classifier.train()




        src_reps_list_chnl = self.feature_extractor.fetch_individual_reps(src_x)

        clfr_src_list = 0
        for k in range(0,self.true_input_channel):
            clfr_k_pred = self.classifier_list_ind[k](src_reps_list_chnl[k])
            clfr_src_list = clfr_src_list + (self.cross_entropy( clfr_k_pred.squeeze(), src_y))


        trg_reps_list_chnl = self.feature_extractor.fetch_individual_reps(trg_x)
        align_loss = 0
        for k in range(0, self.true_input_channel):
            #align_loss_k = self.sink(src_reps_list_chnl[k], trg_reps_list_chnl[k])[0]
            align_loss_k = self.sink( trg_reps_list_chnl[k],src_reps_list_chnl[k])[0] + self.sink(src_reps_list_chnl[k], trg_reps_list_chnl[k])[0]
            #clfr_k_pred = self.classifier_list_ind[k](src_reps_list_chnl[k])

            #idx_src = torch.where(src_y == 3)[0]
            #idx_src = torch.where(trg_y == 3)[0]
            #map = self.sink(src_reps_list_chnl[k][idx_src, :], trg_reps_list_chnl[k])[1].detach().cpu()
            #P = torch.cdist(src_x[idx_src,k,:],trg_x[:,k,:]).detach().cpu()
            #torch.argmax(clfr_k_pred, dim=-1)[idx_src]
            #clfr_loss = self.cross_entropy( clfr_k_pred.squeeze(), src_y)
            #chnl_loss = clfr_loss +align_loss_k
            #chnl_loss.backward(retain_graph=True)
            align_loss = align_loss + align_loss_k





        domain_loss_ind = 1*torch.sum(clfr_src_list ) + torch.sum(align_loss  )
        loss_ind = self.hparams["src_cls_loss_wt"] * torch.sum(clfr_src_list) + \
               self.hparams["domain_loss_wt"] * domain_loss_ind

        sum_reps_src = 0
        sum_reps_trgt = 0
        for k in range(0,self.true_input_channel):
            sum_reps_src = src_reps_list_chnl[k] + sum_reps_src
            sum_reps_trgt = trg_reps_list_chnl[k] + sum_reps_trgt

        domain_loss = self.sink(sum_reps_src,sum_reps_trgt)[0]

        predicts_src = self.classifier(sum_reps_src)


        loss_sup_src = self.cross_entropy(predicts_src.squeeze(),src_y)



        if epoch> 0:

            loss_comb = 1*(self.hparams["src_cls_loss_wt"] * loss_sup_src + \
                       self.hparams["domain_loss_wt"]*4 * domain_loss) + 1*loss_ind
        else:
            loss_comb = 0

        loss_comb.backward()
        self.optimizer_comb.step()

        loss_comb_total = loss_comb.item()

        #reps_src_per_c = self.feature_extractor.fetch_individual_reps(src_x)
        #reps_trg_per_c = self.feature_extractor.fetch_individual_reps(trg_x)

       # for k in range(0,3):


    # self.ema.update()
        return {'Total_loss': loss_comb_total,'Src_cls_loss':loss_sup_src.item(),'Domain_loss':domain_loss.item()}



        # self.ema.update()
        return {'Total_loss': np.mean(tloss_list), 'Src_cls_loss':np.mean(src_loss_list), 'Domain_loss': np.mean(dom_loss_list)}
class SepAligThenAttnSinkFreq(Algorithm):
    """
    Separate representations for each chanell. Multihead attention to combine them. Loss applied on combined term
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(SepAligThenAttnSinkFreq, self).__init__(configs)
        true_final_out_channels = configs.true_final_out_channels
        configs.final_out_channels = true_final_out_channels
        self.true_input_channel = configs.input_channels
        self.feature_extractor = SepReps_with_multihead_with_freq(configs,backbone_fe)


        self.classifier_list_ind = nn.ModuleList([])

        for k in range(0,2*self.true_input_channel):
            self.classifier_list_ind.append(classifier(configs))


        self.optimizer_ind = torch.optim.Adam(
            list(self.feature_extractor.backbone_nets.parameters())  +\
       list(self.classifier_list_ind.parameters()),
            lr=1 * hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99))



        configs.input_channels = self.true_input_channel
        configs.final_out_channels = true_final_out_channels * 3*2

        self.classifier = classifier(configs)

        self.optimizer_comb = torch.optim.Adam(list(self.feature_extractor.backbone_nets.parameters())  +\
       list(self.classifier_list_ind.parameters()) +list(self.feature_extractor.multihead_attention.parameters())+
                                               list(self.classifier.parameters()),lr=1 * hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99))

        self.hparams = hparams
        self.feature_extractor.to(device)

        self.classifier.to(device)

        self.classifier_list_ind.to(device)
        #self.multihead.to(device)
        #self.multihead_ff.to(device)
        self.device = device
        self.sink = SinkhornDistance(eps=1e-2, max_iter=30, reduction='sum')


    def update(self, src_x, src_y, trg_x,step,epoch,len_dataloader):


        #self.optimizer_ind.zero_grad()
        self.optimizer_comb.zero_grad()




        src_reps_list_chnl = self.feature_extractor.fetch_individual_reps(src_x)

        clfr_src_list = 0
        for k in range(0,2*self.true_input_channel):
            clfr_k_pred = self.classifier_list_ind[k](src_reps_list_chnl[k])
            clfr_src_list = clfr_src_list + (self.cross_entropy( clfr_k_pred.squeeze(), src_y))
        trg_reps_list_chnl = self.feature_extractor.fetch_individual_reps(trg_x)
        align_loss = 0
        for k in range(0, 2*self.true_input_channel):
            align_loss_k = self.sink(src_reps_list_chnl[k], trg_reps_list_chnl[k])

            align_loss = align_loss + align_loss_k[0]





        domain_loss_ind = torch.sum(clfr_src_list ) + torch.sum(align_loss  )
        loss_ind = self.hparams["src_cls_loss_wt"] * torch.sum(clfr_src_list) + \
               self.hparams["domain_loss_wt"] * domain_loss_ind
        #loss_ind.backward()
        #self.optimizer_ind.step()

        #loss_ind_total = loss_ind.item()

        #self.optimizer_comb.zero_grad()

        #self.optimizer_ind.zero_grad()



        comb_reps_src = self.feature_extractor.combine_ind_through_attn(src_reps_list_chnl)
        clfr_pred_comb = self.classifier(comb_reps_src)
        loss_sup_src = self.cross_entropy(clfr_pred_comb.squeeze(),src_y)



        comb_reps_trg = self.feature_extractor.combine_ind_through_attn(trg_reps_list_chnl)



        domain_loss = self.sink(comb_reps_src,comb_reps_trg)[0]

        if epoch> 30:

            loss_comb = self.hparams["src_cls_loss_wt"] * loss_sup_src + \
                       self.hparams["domain_loss_wt"] * domain_loss  +1*loss_ind
        else:
            loss_comb = 1*loss_ind

        loss_comb.backward()
        self.optimizer_comb.step()

        loss_comb_total = loss_comb.item()

        #reps_src_per_c = self.feature_extractor.fetch_individual_reps(src_x)
        #reps_trg_per_c = self.feature_extractor.fetch_individual_reps(trg_x)

       # for k in range(0,3):


    # self.ema.update()
        return {'Total_loss': loss_comb_total,'Src_cls_loss':loss_sup_src.item(),'Domain_loss':domain_loss.item()}


    def get_ind_scores(self,x):
        self.feature_extractor.eval()
        self.classifier_list_ind.eval()

        pred_prob_list =[]
        pred_list =[]
        with torch.no_grad():
            src_reps_list_chnl = self.feature_extractor.fetch_individual_reps(x)
            for k in range(0, self.true_input_channel):
                pred_prob_list.append(self.classifier_list_ind[k](src_reps_list_chnl[k]).detach().cpu())
                pred_list.append(self.classifier_list_ind[k](src_reps_list_chnl[k]).argmax(dim=1).detach().cpu())
            for k in range(0, self.true_input_channel):
                pred_prob_list.append(self.classifier_list_ind[k+3](src_reps_list_chnl[k+3]).detach().cpu())
                pred_list.append(self.classifier_list_ind[k+3](src_reps_list_chnl[k+3]).argmax(dim=1).detach().cpu())
        return pred_prob_list,pred_list


    def eval_update(self, src,trg):
        joint_loaders = enumerate(zip(src, trg))
        len_dataloader = min(len(src), len(trg))
        src_loss_list = []
        dom_loss_list = []
        tloss_list = []
        with torch.no_grad():
            for step, ((src_x, src_y), (trg_x, trg_y)) in joint_loaders:
                src_x, src_y, trg_x, trg_y = src_x.float().to(self.device), src_y.long().to(self.device), \
                                             trg_x.float().to(self.device), trg_y.to(self.device)



                domain_label_src = torch.ones(len(src_x)).to(self.device)
                domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

                src_reps_list_chnl = self.feature_extractor.fetch_individual_reps(src_x)

                clfr_src_list = 0
                for k in range(0, self.true_input_channel):
                    clfr_k_pred = self.classifier_list_ind[k](src_reps_list_chnl[k])
                    clfr_src_list = clfr_src_list + (self.cross_entropy(clfr_k_pred.squeeze(), src_y))
                trg_reps_list_chnl = self.feature_extractor.fetch_individual_reps(trg_x)
                align_loss = 0
                for k in range(0, self.true_input_channel):
                    align_loss_k = self.sink(src_reps_list_chnl[k], trg_reps_list_chnl[k])

                    align_loss = align_loss + align_loss_k[0]

                domain_loss_ind = torch.sum(clfr_src_list) + torch.sum(align_loss)
                loss_ind = self.hparams["src_cls_loss_wt"] * torch.sum(clfr_src_list) + \
                           self.hparams["domain_loss_wt"] * domain_loss_ind


                comb_reps_src = self.feature_extractor.combine_ind_through_attn(src_reps_list_chnl)
                clfr_pred_comb = self.classifier(comb_reps_src)
                loss_sup_src = self.cross_entropy(clfr_pred_comb.squeeze(), src_y)

                comb_reps_trg = self.feature_extractor.combine_ind_through_attn(trg_reps_list_chnl)

                domain_loss = self.sink(comb_reps_src, comb_reps_trg)[0]

                loss_comb = self.hparams["src_cls_loss_wt"] * loss_sup_src + \
                            self.hparams["domain_loss_wt"] * domain_loss + 0.3*loss_ind

                dom_loss_list.append(domain_loss.item())
                src_loss_list.append(loss_sup_src.item())
                tloss_list.append(loss_comb.item())

        # self.ema.update()
        return {'Total_loss': np.mean(tloss_list), 'Src_cls_loss':np.mean(src_loss_list), 'Domain_loss': np.mean(dom_loss_list)}


# CLUDA Algorithm Implementation
class CLUDA(Algorithm):

    def __init__(self, backbone_fe, configs, hparams,device):

        super(CLUDA, self).__init__(configs)
        self.use_mask = configs.use_mask
        self.hparams = hparams
        self.device = device
        self.input_channels_dim = configs.input_channels
        self.input_static_dim = configs.input_static_dim

        # different from other algorithms, we import entire model at onces. (i.e. no separate feature extractor or classifier)
        self.feature_extractor = CLUDA_NN(configs,backbone_fe)

        self.classifier = classifier(configs)

        self.augmenter = None
        self.concat_mask = concat_mask

        self.criterion_CL = nn.CrossEntropyLoss()

        self.cuda()

        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + list(self.classifier.parameters()),
            lr=1 * hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )



    def update(self, src_x, src_y, trg_x,step,epoch,len_dataloader, **kwargs):
        # For Augmenter, Cutout length is calculated relative to the sequence length
        # If there is only one channel, there will be no spatial dropout
        if self.augmenter is None:
            self.get_augmenter(src_x)



        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1


        # Go through augmentations first


        seq_q_src, seq_mask_q_src = self.augmenter(src_x, torch.ones(src_x.shape).to(self.device))
        seq_k_src, seq_mask_k_src = self.augmenter(src_x, torch.ones(src_x.shape).to(self.device))

        seq_q_trg, seq_mask_q_trg = self.augmenter(trg_x, torch.ones(trg_x.shape).to(self.device))
        seq_k_trg, seq_mask_k_trg = self.augmenter(trg_x, torch.ones(trg_x.shape).to(self.device))

        # Concat mask if use_mask = True
        seq_q_src = self.concat_mask(seq_q_src, seq_mask_q_src, self.use_mask)
        seq_k_src = self.concat_mask(seq_k_src, seq_mask_k_src, self.use_mask)
        seq_q_trg = self.concat_mask(seq_q_trg, seq_mask_q_trg, self.use_mask)
        seq_k_trg = self.concat_mask(seq_k_trg, seq_mask_k_trg, self.use_mask)

        # compute output

        #Understand this
        #sample_batched_src.get('static')
        #sample_batched_trg.get('static')
        #output_s, target_s, output_t, target_t, output_ts, target_ts, output_disc, target_disc, pred_s = self.feature_extractor.update(
        #    seq_q_src, seq_k_src, sample_batched_src.get('static'), seq_q_trg, seq_k_trg,
        #    sample_batched_trg.get('static'), alpha)

        output_s, target_s, output_t, target_t, output_ts, target_ts, output_disc, target_disc, q_s = self.feature_extractor.update(
            seq_q_src, seq_k_src, None, seq_q_trg, seq_k_trg,
            None, alpha)

        # Compute all losses
        loss_s = self.criterion_CL(output_s, target_s)
        loss_t = self.criterion_CL(output_t, target_t)
        loss_ts = self.criterion_CL(output_ts, target_ts)
        loss_disc = F.binary_cross_entropy(output_disc, target_disc)

        pred_s = self.classifier(q_s)
        # Task classification  Loss
        src_cls_loss = self.criterion_CL(pred_s, src_y)

        loss = 1* loss_s + 1 * loss_t + \
               1 * loss_ts + 1 * loss_disc + 1 * src_cls_loss


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Domain_loss': loss_disc.item(), 'Src_cls_loss': src_cls_loss.item()}

    def return_metrics(self):
        return [self.losses_s, self.top1_s, self.losses_t, self.top1_t, self.losses_ts, self.top1_ts,
                self.losses_disc, self.top1_disc, self.losses_pred, self.score_pred, self.losses]





    # We need to overwrite below functions for CLUDA
    def predict_trg(self, sample_batched):

        seq_t = self.concat_mask(sample_batched['sequence'], sample_batched['sequence_mask'], self.use_mask)
        y_pred_trg = self.model.predict(seq_t, sample_batched.get('static'), is_target=True)

        self.pred_meter_val_trg.update(sample_batched['label'], y_pred_trg, id_patient=sample_batched.get('patient_id'),
                                       stay_hour=sample_batched.get('stay_hour'))

    def predict_src(self, sample_batched):

        seq_s = self.concat_mask(sample_batched['sequence'], sample_batched['sequence_mask'], self.use_mask)
        y_pred_src = self.model.predict(seq_s, sample_batched.get('static'), is_target=False)

        self.pred_meter_val_src.update(sample_batched['label'], y_pred_src, id_patient=sample_batched.get('patient_id'),
                                       stay_hour=sample_batched.get('stay_hour'))

    def get_embedding(self, sample_batched):

        seq = self.concat_mask(sample_batched['sequence'], sample_batched['sequence_mask'], self.use_mask)
        feat = self.model.get_encoding(seq)

        return feat

    def get_augmenter(self, sample_batched):

        #seq_len = sample_batched['sequence'].shape[1]
        #num_channel = sample_batched['sequence'].shape[2]


        seq_len = sample_batched.shape[1]
        num_channel = sample_batched.shape[2]
        cutout_len = math.floor(seq_len / 12)
        if self.input_channels_dim != 1:
            self.augmenter = Augmenter(cutout_length=cutout_len)
        # IF THERE IS ONLY ONE CHANNEL, WE NEED TO MAKE SURE THAT CUTOUT AND CROPOUT APPLIED (i.e. their probs are 1)
        # for extremely long sequences (such as SSC with 3000 time steps)
        # apply the cutout in multiple places, in return, reduce history crop
        elif self.input_channels_dim == 1 and seq_len > 1000:
            self.augmenter = Augmenter(cutout_length=cutout_len, cutout_prob=1, crop_min_history=0.25, crop_prob=1,
                                       dropout_prob=0.0)
            # we apply cutout 3 times in a row.
            self.augmenter.augmentations = [self.augmenter.history_cutout, self.augmenter.history_cutout,
                                            self.augmenter.history_cutout,
                                            self.augmenter.history_crop, self.augmenter.gaussian_noise,
                                            self.augmenter.spatial_dropout]
        # if there is only one channel but not long, we just need to make sure that we don't drop this only channel
        else:
            self.augmenter = Augmenter(cutout_length=cutout_len, dropout_prob=0.0)
