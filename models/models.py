import torch
from torch import nn
import math
from torch.autograd import Function
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import torch.fft as FFT
from models.multihead_attention import MultiHeadAttention as Multihead2

from CLUDA_main.utils.nearest_neighbor import NN, sim_matrix
from CLUDA_main.utils.mlp import MLP
import time

def get_backbone_class(backbone_name):
    """Return the algorithm class with the given name."""
    if backbone_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(backbone_name))
    return globals()[backbone_name]


##################################################
##########  BACKBONE NETWORKS  ###################
##################################################

##########  ( #############################
class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels , kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels , configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        x_flat = x.reshape(x.shape[0], -1)

        return F.normalize(x_flat)

    def get_batch_norm_stats_init(self):
        layer1_bnorm = self.conv_block1[1]
        layer2_bnorm = self.conv_block2[1]
        layer3_bnorm = self.conv_block3[1]


        return [[layer1_bnorm.running_mean,layer1_bnorm.running_var],[layer2_bnorm.running_mean,layer2_bnorm.running_var],[layer3_bnorm.running_mean,layer3_bnorm.running_var]]

    def get_batch_norm_stats_layer1(self,x):
        z = self.conv_block1[0](x)
        z2 = self.conv_block1[1](z)
        a = self.conv_block1[1]
        return [a.running_mean,a.running_var]

    def get_batch_norm_stats_layer2(self,x):
        x_1 = self.conv_block1(x)
        z = self.conv_block2[0](x_1 )
        z2 = self.conv_block2[1](z)
        a = self.conv_block2[1]
        return [a.running_mean,a.running_var]

    def get_batch_norm_stats_layer3(self,x):
        x_1 = self.conv_block1(x)
        x2 = self.conv_block2(x_1)
        z = self.conv_block3[0](x2)
        z2 = self.conv_block3[1](z)
        a = self.conv_block3[1]
        return [a.running_mean,a.running_var]


    def set_1st_layer_batchnorm(self,mean,var):
        a = self.conv_block1[1]
        a.running_mean = mean
        a.running_var = var

    def set_2nd_layer_batchnorm(self,mean,var):
        a = self.conv_block2[1]
        a.running_mean = mean
        a.running_var = var

    def set_3rd_layer_batchnorm(self,mean,var):
        a = self.conv_block3[1]
        a.running_mean = mean
        a.running_var = var
class classifier2(nn.Module):
    def __init__(self, configs,emb_dims):
        super(classifier2, self).__init__()
        #model_output_dim = configs.final_out_channels
        self.logits = nn.Linear(emb_dims, configs.num_classes, bias=False)
        self.tmp= 0.1

    def forward(self, x):
        predictions = self.logits(x)/self.tmp
        return predictions

class classifier(nn.Module):
    def __init__(self, configs):
        super(classifier, self).__init__()
        model_output_dim =   configs.final_out_channels*configs.features_len #192 #configs.final_out_channels

        self.ff = nn.Sequential(nn.Linear(model_output_dim,model_output_dim, bias=False),nn.ReLU())

        self.logits = nn.Linear(model_output_dim, configs.num_classes, bias=False)
        self.tmp= 1

    def forward(self, x):
        #x = self.ff(x)
        predictions = self.logits(x)/self.tmp
        return predictions


class ResClassifier_MME(nn.Module):
    def __init__(self, configs):
        super(ResClassifier_MME, self).__init__()
        self.norm = True
        self.tmp = 0.02
        num_classes = configs.num_classes
        input_size = configs.out_dim
   
        self.fc = nn.Linear(input_size, num_classes, bias=False)
            
    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False):
        if return_feat:
            return x
        x = self.fc(x)/self.tmp
        return x

    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))
        
    def weights_init(self):
        self.fc.weight.data.normal_(0.0, 0.1)

########## TCN #############################
torch.backends.cudnn.benchmark = True  # might be required to fasten TCN


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TCN(nn.Module):
    def __init__(self, configs):
        super(TCN, self).__init__()

        in_channels0 = configs.input_channels
        out_channels0 = configs.tcn_layers[1]
        kernel_size = configs.tcn_kernel_size
        stride = 1
        dilation0 = 1
        padding0 = (kernel_size - 1) * dilation0

        self.net0 = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels0, out_channels0, kernel_size, stride=stride, padding=padding0,
                                  dilation=dilation0)),
            nn.ReLU(),
            weight_norm(nn.Conv1d(out_channels0, out_channels0, kernel_size, stride=stride, padding=padding0,
                                  dilation=dilation0)),
            nn.ReLU(),
        )

        self.downsample0 = nn.Conv1d(in_channels0, out_channels0, 1) if in_channels0 != out_channels0 else None
        self.relu = nn.ReLU()

        in_channels1 = configs.tcn_layers[0]
        out_channels1 = configs.tcn_layers[1]
        dilation1 = 2
        padding1 = (kernel_size - 1) * dilation1
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels0, out_channels1, kernel_size, stride=stride, padding=padding1, dilation=dilation1),
            nn.ReLU(),
            nn.Conv1d(out_channels1, out_channels1, kernel_size, stride=stride, padding=padding1, dilation=dilation1),
            nn.ReLU(),
        )
        self.downsample1 = nn.Conv1d(out_channels1, out_channels1, 1) if in_channels1 != out_channels1 else None

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels0, out_channels0, kernel_size=kernel_size, stride=stride, bias=False, padding=padding0,
                      dilation=dilation0),
            Chomp1d(padding0),
            nn.BatchNorm1d(out_channels0),
            nn.ReLU(),

            nn.Conv1d(out_channels0, out_channels0, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding0, dilation=dilation0),
            Chomp1d(padding0),
            nn.BatchNorm1d(out_channels0),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(out_channels0, out_channels1, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding1, dilation=dilation1),
            Chomp1d(padding1),
            nn.BatchNorm1d(out_channels1),
            nn.ReLU(),

            nn.Conv1d(out_channels1, out_channels1, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding1, dilation=dilation1),
            Chomp1d(padding1),
            nn.BatchNorm1d(out_channels1),
            nn.ReLU(),
        )

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        x0 = self.conv_block1(inputs)
        res0 = inputs if self.downsample0 is None else self.downsample0(inputs)
        out_0 = self.relu(x0 + res0)

        x1 = self.conv_block2(out_0)
        res1 = out_0 if self.downsample1 is None else self.downsample1(out_0)
        out_1 = self.relu(x1 + res1)

        out = out_1[:, :, -1]
        return out


######## RESNET ##############################################
class RESNET18(nn.Module):
    def __init__(self, configs):
        layers = [2, 2, 2, 2]
        block = BasicBlock
        self.inplanes = configs.input_channels
        super(RESNET18, self).__init__()
        self.layer1 = self._make_layer(block, configs.mid_channels, layers[0], stride=configs.stride)
        self.layer2 = self._make_layer(block, configs.mid_channels * 2, layers[1], stride=1)
        self.layer3 = self._make_layer(block, configs.final_out_channels, layers[2], stride=1)
        self.layer4 = self._make_layer(block, configs.final_out_channels, layers[3], stride=1)

        self.avgpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.adaptive_pool(x)

        x_flat = x.reshape(x.shape[0], -1)
        return x_flat


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out


##################################################
##########  OTHER NETWORKS  ######################
##################################################

class codats_classifier(nn.Module):
    def __init__(self, configs):
        super(codats_classifier, self).__init__()
        model_output_dim = configs.features_len
        self.hidden_dim = configs.hidden_dim
        self.logits = nn.Sequential(
            nn.Linear(model_output_dim * configs.final_out_channels, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, configs.num_classes))

    def forward(self, x_in):
        predictions = self.logits(x_in)
        return predictions


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, configs):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(configs.features_len * configs.final_out_channels , configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, 2)
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out


#### Codes required by DANN ##############
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, fl=128):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        self.bn_freq = nn.BatchNorm1d(
            64 )
        self.avg = nn.Conv1d(1, 1, kernel_size=3,
                             stride=1, bias=False, padding=(3 // 2))
    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)

        #why the cosine?
        #x = torch.cos(x)
        x_ft = torch.fft.rfft(x, norm='ortho')
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        r = out_ft[:, :, :self.modes1].abs()
        #p = out_ft[:, :, :self.modes1].angle()
        f= r
        ef = F.relu(self.bn_freq(self.avg(f).squeeze()))
        return F.normalize((ef))


class simple_average_feed_forward(nn.Module):
    def __init__(self,configs):
        super(simple_average_feed_forward,self).__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool1d(64)
        self.linear = nn.Linear(64*configs.input_channels,64)
        self.Relu = nn.ReLU()
        self.linear2 = nn.Linear(64, 64)
        self.net = nn.Sequential(self.linear,self.Relu,self.linear2)
    def forward(self,x):
        x = self.adaptive_pool(x).reshape(x.shape[0],-1)

        x = self.net(x)
        return x

class SepReps_with_multihead(nn.Module):
    'code to get separate reps and combine through a multi attention head across channels'
    def __init__(self, configs,backbonenet):
        super(SepReps_with_multihead, self).__init__()
        self.no_channels = configs.input_channels
        self.backbone_nets =  nn.ModuleList([])

        for k in range(0,self.no_channels):
            configs.input_channels = 1
            self.backbone_nets.append(backbonenet(configs))
            #self.backbone_nets.append(simple_average_feed_forward(configs))

        #self.multihead_attention  = nn.MultiheadAttention( configs.final_out_channels ,num_heads=1,batch_first=True,bias=False)
        self.multihead_attention =  Multihead2( configs.final_out_channels ,head_num=1,bias=False,temp=configs.temp)
        self.ff_attent = nn.Sequential(nn.Linear(configs.final_out_channels*self.no_channels,configs.final_out_channels*self.no_channels)
                                       ,nn.ReLU(),nn.Linear(configs.final_out_channels*self.no_channels,configs.final_out_channels*self.no_channels))
        self.ff_attent2 = nn.Sequential(
            nn.Linear(configs.final_out_channels * self.no_channels, configs.final_out_channels * self.no_channels)
            , nn.ReLU(),
            nn.Linear( configs.final_out_channels * self.no_channels,configs.final_out_channels * self.no_channels))
    def forward(self,x):
        rep_list =[]
        for k in range(0,self.no_channels):
            x_k = x[:,k,:].unsqueeze(1)
            t_1= time.time()
            rep_list.append(F.normalize(self.backbone_nets[k](x_k),dim=1))
            t_2 = time.time()
            #print("here")
        rep_all = torch.stack(rep_list,dim=1)
        rep_comb,_ = self.multihead_attention(rep_all,rep_all,rep_all)

        #rep_comb2 = rep_comb+rep_all
        rep_comb = rep_comb.reshape(rep_comb.shape[0],-1)


        #rep_comb
        #rep_comb = self.ff_attent(rep_comb)
        #rep_comb3 = rep_comb2 + self.ff_attent2(rep_comb2)
        rep_list = []
        return rep_comb

    def fetch_att_weights(self,x):
        rep_list = []
        for k in range(0, self.no_channels):
            x_k = x[:, k, :].unsqueeze(1)
            #rep_list.append(self.backbone_nets[k](x_k))
            t_1 = time.time()
            rep_list.append(F.normalize(self.backbone_nets[k](x_k), dim=1))
            t_2 = time.time()
        rep_all = torch.stack(rep_list, dim=1)
        rep_comb, rep_attn = self.multihead_attention(rep_all, rep_all, rep_all)
        rep_comb = rep_comb.reshape(rep_comb.shape[0], -1)
        rep_comb = self.ff_attent(rep_comb)
        return rep_attn


    def fetch_individual_reps(self,x):
        rep_list = []
        for k in range(0, self.no_channels):
            x_k = x[:, k, :].unsqueeze(1)
            s_1 = time.time()
            rep_list.append(F.normalize(self.backbone_nets[k](x_k),dim=1))
            s_2 = time.time()
            #rep_list.append(x_k[:,:,0:64].squeeze(1))
        rep_all = torch.stack(rep_list, dim=1)
        rep_comb, rep_attn = self.multihead_attention(rep_all, rep_all, rep_all)
        #rep_comb = rep_comb.reshape(rep_comb.shape[0], -1)
        #rep_comb = self.ff_attent(rep_comb)
        return rep_list

    def combine_ind_through_attn(self,rep_list):
        'takes in a list of representations'

        rep_all = torch.stack(rep_list, dim=1).detach()
        rep_comb, rep_attn = self.multihead_attention(rep_all, rep_all, rep_all)
        rep_comb = rep_comb.reshape(rep_comb.shape[0], -1)
        #rep_comb = self.ff_attent(rep_comb)
        return rep_comb,rep_attn

class SepRepsComEnc_with_multihead(nn.Module):
    'code to get separate reps and combine through a multi attention head across channels'
    def __init__(self, configs,backbonenet):
        super(SepRepsComEnc_with_multihead, self).__init__()
        self.no_channels = configs.input_channels
        self.backbone_nets =  nn.ModuleList([])

        #for k in range(0,self.no_channels):
        #    configs.input_channels = 1
        #    self.backbone_nets.append(backbonenet(configs))
        #Only one channel
        for k in range(0, 1):
                configs.input_channels = 1
                self.backbone_nets.append(backbonenet(configs))
            #self.backbone_nets.append(simple_average_feed_forward(configs))

        #self.multihead_attention  = nn.MultiheadAttention( configs.final_out_channels ,num_heads=1,batch_first=True,bias=False)
        self.multihead_attention =  Multihead2( configs.final_out_channels ,head_num=1,bias=False,temp=configs.temp)
        self.ff_attent = nn.Sequential(nn.Linear(configs.final_out_channels*self.no_channels,configs.final_out_channels*self.no_channels)
                                       ,nn.ReLU(),nn.Linear(configs.final_out_channels*self.no_channels,configs.final_out_channels*self.no_channels))
        self.ff_attent2 = nn.Sequential(
            nn.Linear(configs.final_out_channels * self.no_channels, configs.final_out_channels * self.no_channels)
            , nn.ReLU(),
            nn.Linear( configs.final_out_channels * self.no_channels,configs.final_out_channels * self.no_channels))
    def forward(self,x):
        rep_list =[]
        for k in range(0,self.no_channels):
            x_k = x[:,k,:].unsqueeze(1)
            rep_list.append(F.normalize(self.backbone_nets[0](x_k),dim=1))
        rep_all = torch.stack(rep_list,dim=1)
        rep_comb,_ = self.multihead_attention(rep_all,rep_all,rep_all)

        #rep_comb2 = rep_comb+rep_all
        rep_comb = rep_comb.reshape(rep_comb.shape[0],-1)


        #rep_comb
        #rep_comb = self.ff_attent(rep_comb)
        #rep_comb3 = rep_comb2 + self.ff_attent2(rep_comb2)
        rep_list = []
        return rep_comb

    def fetch_att_weights(self,x):
        rep_list = []
        for k in range(0, self.no_channels):
            x_k = x[:, k, :].unsqueeze(1)
            #rep_list.append(self.backbone_nets[k](x_k))
            rep_list.append(F.normalize(self.backbone_nets[0](x_k), dim=1))
        rep_all = torch.stack(rep_list, dim=1)
        rep_comb, rep_attn = self.multihead_attention(rep_all, rep_all, rep_all)
        rep_comb = rep_comb.reshape(rep_comb.shape[0], -1)
        rep_comb = self.ff_attent(rep_comb)
        return rep_attn


    def fetch_individual_reps(self,x):
        rep_list = []
        for k in range(0, self.no_channels):
            x_k = x[:, k, :].unsqueeze(1)

            rep_list.append(F.normalize(self.backbone_nets[0](x_k),dim=1))
            #rep_list.append(x_k[:,:,0:64].squeeze(1))
        rep_all = torch.stack(rep_list, dim=1)
        rep_comb, rep_attn = self.multihead_attention(rep_all, rep_all, rep_all)
        #rep_comb = rep_comb.reshape(rep_comb.shape[0], -1)
        #rep_comb = self.ff_attent(rep_comb)
        return rep_list

    def combine_ind_through_attn(self,rep_list):
        'takes in a list of representations'

        rep_all = torch.stack(rep_list, dim=1).detach()
        rep_comb, rep_attn = self.multihead_attention(rep_all, rep_all, rep_all)
        rep_comb = rep_comb.reshape(rep_comb.shape[0], -1)
        #rep_comb = self.ff_attent(rep_comb)
        return rep_comb,rep_attn

class SepReps_with_sum(nn.Module):
    'code to get separate reps and combine through summing them'
    def __init__(self, configs,backbonenet):
        super(SepReps_with_sum, self).__init__()
        self.no_channels = configs.input_channels
        self.backbone_nets =  nn.ModuleList([])

        for k in range(0,self.no_channels):
            configs.input_channels = 1
            self.backbone_nets.append(backbonenet(configs))
            #self.backbone_nets.append(simple_average_feed_forward(configs))

    def forward(self,x):
        rep_sum = 0
        for k in range(0,self.no_channels):
            x_k = x[:,k,:].unsqueeze(1)
            rep_sum = rep_sum + F.normalize(self.backbone_nets[k](x_k),dim=-1)

        return rep_sum


    def fetch_individual_reps(self,x):
        rep_list = []
        for k in range(0, self.no_channels):
            x_k = x[:, k, :].unsqueeze(1)

            rep_list.append(F.normalize(self.backbone_nets[k](x_k),dim=-1))
            #rep_list.append(x_k[:,:,0:64].squeeze(1))
        rep_all = torch.stack(rep_list, dim=1)

        return rep_list

class SepReps_with_multihead_with_freq(nn.Module):
    'code to get separate reps and combine through a multi attention head across channels'
    def __init__(self, configs,backbonenet):
        super(SepReps_with_multihead_with_freq, self).__init__()
        self.no_channels = configs.input_channels
        self.backbone_nets =  nn.ModuleList([])

        for k in range(0,self.no_channels):
            configs.input_channels = 1
            self.backbone_nets.append(backbonenet(configs))
        for k in range(0,self.no_channels):
            self.backbone_nets.append(SpectralConv1d(in_channels=configs.input_channels,out_channels=configs.input_channels,modes1=64))
        #self.multihead_attention  = nn.MultiheadAttention( configs.final_out_channels ,num_heads=1,batch_first=True)
        self.multihead_attention = Multihead2(configs.final_out_channels, head_num=1, bias=False, temp=configs.temp)
        self.ff_attent = nn.Sequential(nn.Linear(configs.final_out_channels*self.no_channels,configs.final_out_channels*self.no_channels)
                                       ,nn.ReLU(),nn.Linear(configs.final_out_channels*self.no_channels,configs.final_out_channels*self.no_channels))
        self.ff_attent2 = nn.Sequential(
            nn.Linear(configs.final_out_channels * self.no_channels, configs.final_out_channels * self.no_channels)
            , nn.ReLU(),
            nn.Linear( configs.final_out_channels * self.no_channels,configs.final_out_channels * self.no_channels))
    def forward(self,x):
        rep_list =[]
        for k in range(0,self.no_channels):
            x_k = x[:,k,:].unsqueeze(1)
            rep_list.append(self.backbone_nets[k](x_k))
        for k in range(0,self.no_channels):
            x_k = x[:,k,:].unsqueeze(1)
            rep_list.append(self.backbone_nets[k+self.no_channels](x_k))
        rep_all = torch.stack(rep_list,dim=1)
        rep_comb,_ = self.multihead_attention(rep_all,rep_all,rep_all)

        #rep_comb2 = rep_comb+rep_all
        rep_comb = rep_comb.reshape(rep_comb.shape[0],-1)


        #rep_comb
        #rep_comb2 = rep_comb + self.ff_attent(rep_comb)
        #rep_comb3 = rep_comb2 + self.ff_attent2(rep_comb2)
        rep_list = []
        return rep_comb

    def fetch_att_weights(self,x):
        rep_list = []
        for k in range(0, self.no_channels):
            x_k = x[:, k, :].unsqueeze(1)
            rep_list.append(self.backbone_nets[k](x_k))
        for k in range(0,self.no_channels):
            x_k = x[:,k,:].unsqueeze(1)
            rep_list.append(self.backbone_nets[k+self.no_channels](x_k))
        rep_all = torch.stack(rep_list, dim=1)
        rep_comb, rep_attn = self.multihead_attention(rep_all, rep_all, rep_all)
        rep_comb = rep_comb.reshape(rep_comb.shape[0], -1)
        #rep_comb = self.ff_attent(rep_comb)
        return rep_attn


    def fetch_individual_reps(self,x):
        rep_list = []
        for k in range(0, self.no_channels):
            x_k = x[:, k, :].unsqueeze(1)
            rep_list.append(self.backbone_nets[k](x_k))
        for k in range(0,self.no_channels):
            x_k = x[:,k,:].unsqueeze(1)
            rep_list.append(self.backbone_nets[k+self.no_channels](x_k))
        rep_all = torch.stack(rep_list, dim=1)
        rep_comb, rep_attn = self.multihead_attention(rep_all, rep_all, rep_all)
        #rep_comb = rep_comb.reshape(rep_comb.shape[0], -1)
        #rep_comb = self.ff_attent(rep_comb)
        return rep_list

    def combine_ind_through_attn(self,rep_list):
        'takes in a list of representations'

        rep_all = torch.stack(rep_list, dim=1).detach()
        rep_comb, rep_attn = self.multihead_attention(rep_all, rep_all, rep_all)
        rep_comb = rep_comb.reshape(rep_comb.shape[0], -1)
        #rep_comb = self.ff_attent(rep_comb)
        return rep_comb


class SepReps(nn.Module):
    'code to get separate reps'
    def __init__(self, configs,backbonenet):
        super(SepReps, self).__init__()
        self.no_channels = configs.input_channels
        self.backbone_nets =  nn.ModuleList([])
        for k in range(0,self.no_channels):
            configs.input_channels = 1
            self.backbone_nets.append(backbonenet(configs))
        #self.multihead_attention  = nn.MultiheadAttention( configs.final_out_channels ,num_heads=32,batch_first=True)

        #self.ff_attent = nn.Sequential(nn.Linear(configs.final_out_channels*self.no_channels,configs.final_out_channels*self.no_channels)
        #                               ,nn.ReLU())
    def forward(self,x):
        rep_list =[]
        for k in range(0,self.no_channels):
            x_k = x[:,k,:].unsqueeze(1)
            rep_list.append(self.backbone_nets[k](x_k))
        return rep_list

class AttnCombMultihead(nn.Module):
    'Class that takes in separate representations in a list and tries to align them'
    def __init__(self, configs,nhead=32):
        self.multihead_attention = nn.MultiheadAttention(configs.final_out_channels, num_heads=nhead, batch_first=True)
        self.ff_attent = nn.Sequential(
            nn.Linear(configs.final_out_channels * self.no_channels, configs.final_out_channels * self.no_channels)
            , nn.ReLU())

    def forward(self, rep_list):
        'takes in a list of representations'
        rep_all = torch.stack(rep_list, dim=1)
        rep_comb, rep_attn = self.multihead_attention(rep_all, rep_all, rep_all)

        return rep_comb

    def fetch_att_weights(self, rep_list):
        rep_all = torch.stack(rep_list, dim=1)
        _, rep_attn = self.multihead_attention(rep_all, rep_all, rep_all)
        # rep_comb = rep_comb.reshape(rep_comb.shape[0], -1)
        # rep_comb = self.ff_attent(rep_comb)
        return rep_attn.detach().cpu()
#### Codes required by CDAN ##############
class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]


class Discriminator_CDAN(nn.Module):
    """Discriminator model for CDAN ."""

    def __init__(self, configs):
        """Init discriminator."""
        super(Discriminator_CDAN, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(configs.features_len * configs.final_out_channels * configs.num_classes, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, 2)
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out


#### Codes required by AdvSKM ##############
class Cosine_act(nn.Module):
    def __init__(self):
        super(Cosine_act, self).__init__()

    def forward(self, input):
        return torch.cos(input)


cos_act = Cosine_act()

class AdvSKM_Disc(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, configs):
        """Init discriminator."""
        super(AdvSKM_Disc, self).__init__()

        self.input_dim = configs.features_len * configs.final_out_channels
        self.hid_dim = configs.DSKN_disc_hid
        self.branch_1 = nn.Sequential(
            nn.Linear(self.input_dim, self.hid_dim),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.BatchNorm1d(self.hid_dim),
            cos_act,
            nn.Linear(self.hid_dim, self.hid_dim // 2),
            nn.Linear(self.hid_dim // 2, self.hid_dim // 2),
            nn.BatchNorm1d(self.hid_dim // 2),
            cos_act
        )
        self.branch_2 = nn.Sequential(
            nn.Linear(configs.features_len * configs.final_out_channels, configs.disc_hid_dim),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim),
            nn.BatchNorm1d(configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim // 2),
            nn.Linear(configs.disc_hid_dim // 2, configs.disc_hid_dim // 2),
            nn.BatchNorm1d(configs.disc_hid_dim // 2),
            nn.ReLU())

    def forward(self, input):
        """Forward the discriminator."""
        out_cos = self.branch_1(input)
        out_rel = self.branch_2(input)
        total_out = torch.cat((out_cos, out_rel), dim=1)
        return total_out

    
#### Codes for attention ############## 
class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """前向传播.

        Args:
            q: Queries张量，形状为[B, L_q, D_q]
            k: Keys张量，形状为[B, L_k, D_k]
            v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
            scale: 缩放因子，一个浮点标量
            attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
            上下文张量和attetention张量
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        # if attn_mask:
        #     # 给需要mask的地方设置一个负无穷
        #     attention = attention.masked_fill_(attn_mask, -np.inf)
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和V做点积
        context = torch.bmm(attention, v)
        return context, attention

#### Codes for Simclr ############## 
class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out

class Projection(nn.Module):
    """
    Creates projection head
    Args:
    n_in (int): Number of input features
    n_hidden (int): Number of hidden features
    n_out (int): Number of output features
    use_bn (bool): Whether to use batch norm
    """
    def __init__(self, n_in: int, n_hidden: int, n_out: int,
               use_bn: bool = True):
        super().__init__()

        # No point in using bias if we've batch norm
        self.lin1 = nn.Linear(n_in, n_hidden, bias=not use_bn)
        self.bn = nn.BatchNorm1d(n_hidden) if use_bn else nn.Identity()
        self.relu = nn.ReLU()
        # No bias for the final linear layer
        self.lin2 = nn.Linear(n_hidden, n_out, bias=False)

    def forward(self, x):
        x = self.lin1(x)
        # x = self.bn(x)
        x = self.relu(x)
        return x


class SimCLRModel(nn.Module):
    def __init__(self, encoder: nn.Module, projection_n_in: int = 128,
               projection_n_hidden: int = 128, projection_n_out: int = 128,
               projection_use_bn: bool = True):
        super().__init__()

        self.encoder = encoder
        self.projection = Projection(projection_n_in, projection_n_hidden,
                                     projection_n_out, projection_use_bn)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return h, z


class CLUDA_NN(nn.Module):

    def __init__(self, configs,backbonenet):

        self.output_dim = configs.final_out_channels

        mlp_hidden_dim = configs.mlp_hidden_dim
        num_neighbors = configs.num_neighbors
        K = configs.K
        m = configs.m
        T = configs.T

        use_batch_norm = configs.use_batch_norm
        super(CLUDA_NN, self).__init__()

        self.sigmoid = nn.Sigmoid()

        self.K = K
        self.m = m
        self.T = T
        self.num_neighbors = num_neighbors
        num_static = configs.input_static_dim
        # encoders
        # num_classes is the output fc dimension
        #self.encoder_q = TemporalConvNet(num_inputs=num_inputs, num_channels=num_channels, kernel_size=kernel_size,
        #                                 stride=stride, dilation_factor=dilation_factor, dropout=dropout)
        #self.encoder_k = TemporalConvNet(num_inputs=num_inputs, num_channels=num_channels, kernel_size=kernel_size,
        #                                 stride=stride, dilation_factor=dilation_factor, dropout=dropout)

        self.encoder_q = backbonenet(configs)
        self.encoder_k = backbonenet(configs)

        # projector for query
        self.projector = MLP(input_dim=configs.final_out_channels , hidden_dim=mlp_hidden_dim,
                             output_dim=self.output_dim, use_batch_norm=use_batch_norm)

        # Classifier trained by source query
        self.predictor = MLP(input_dim=configs.final_out_channels+ num_static, hidden_dim=mlp_hidden_dim,
                             output_dim=self.output_dim, use_batch_norm=use_batch_norm)

        # Discriminator
        self.discriminator = MLP(input_dim=configs.final_out_channels, hidden_dim=mlp_hidden_dim,
                                 output_dim=1, use_batch_norm=use_batch_norm)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue_s", torch.randn(self.output_dim, K))
        self.queue_s = nn.functional.normalize(self.queue_s, dim=0)

        self.register_buffer("queue_t", torch.randn(self.output_dim, K))
        self.queue_t = nn.functional.normalize(self.queue_t, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        # No update during evaluation
        if self.training:
            # Update the encoder
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_s, keys_t):
        # No update during evaluation
        if self.training:
            # gather keys before updating queue
            batch_size = keys_s.shape[0]

            ptr = int(self.queue_ptr)
            # For now, ignore below assertion
            # assert self.K % batch_size == 0  # for simplicity

            # replace the keys at ptr (dequeue and enqueue)
            try:
                self.queue_s[:, ptr:ptr + batch_size] = keys_s[0:batch_size].T
                self.queue_t[:, ptr:ptr + batch_size] = keys_t[0:batch_size].T

                ptr = (ptr + batch_size) % self.K  # move pointer

                self.queue_ptr[0] = ptr
            except RuntimeError:
                pass


    def forward(self,sequence_q_s):
        q_s = self.encoder_q(sequence_q_s) #[:, :, -1]
        q_s = nn.functional.normalize(q_s, dim=1)
        return q_s

    def update(self, sequence_q_s, sequence_k_s, static_s, sequence_q_t, sequence_k_t, static_t, alpha):
        """
        Input:
            sequence_q: a batch of query sequences
            sequence_k: a batch of key sequences
            static: a batch of static features
        Output:
            logits, targets
        """

        # SOURCE DATASET query computations

        # compute query features
        # Input is in the shape N*L*C whereas TCN expects N*C*L
        # Since the tcn_out has the shape N*C_out*L, we will get the last timestep of the output
        #q_s = self.encoder_q(sequence_q_s.transpose(1, 2))[:, :, -1]  # queries: NxC
        q_s = self.encoder_q(sequence_q_s)
        q_s = nn.functional.normalize(q_s, dim=1)
        # Project the query
        p_q_s = self.projector(q_s, None)  # queries: NxC
        p_q_s = nn.functional.normalize(p_q_s, dim=1)

        # TARGET DATASET query computations

        # compute query features
        # Input is in the shape N*L*C whereas TCN expects N*C*L
        # Since the tcn_out has the shape N*C_out*L, we will get the last timestep of the output
        q_t = self.encoder_q(sequence_q_t) # queries: NxC
        q_t = nn.functional.normalize(q_t, dim=1)
        # Project the query
        p_q_t = self.projector(q_t, None)  # queries: NxC
        p_q_t = nn.functional.normalize(p_q_t, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoders

            # SOURCE DATASET key computations

            # Input is in the shape N*L*C whereas TCN expects N*C*L
            # Since the tcn_out has the shape N*C_out*L, we will get the last timestep of the output
            k_s = self.encoder_k(sequence_k_s) # queries: NxC
            k_s = nn.functional.normalize(k_s, dim=1)

            # TARGET DATASET key computations

            # Input is in the shape N*L*C whereas TCN expects N*C*L
            # Since the tcn_out has the shape N*C_out*L, we will get the last timestep of the output
            k_t = self.encoder_k(sequence_k_t) # queries: NxC
            k_t = nn.functional.normalize(k_t, dim=1)

        # SOURCE DATASET contrastive loss
        # Calculate the logits of the given batch: NxN
        l_batch_s = torch.mm(p_q_s, k_s.transpose(0, 1))
        # Calculate the logits of the queue: NxK
        l_queue_s = torch.mm(p_q_s, self.queue_s.clone().detach())

        # logits Nx(N+K)
        logits_s = torch.cat([l_batch_s, l_queue_s], dim=1)

        # apply temperature
        logits_s /= self.T

        # labels
        labels_s = torch.arange(p_q_s.shape[0], dtype=torch.long).to(device=p_q_s.device)

        # TARGET DATASET contrastive loss
        # Calculate the logits of the given batch: NxN
        l_batch_t = torch.mm(p_q_t, k_t.transpose(0, 1))
        # Calculate the logits of the queue: NxK
        l_queue_t = torch.mm(p_q_t, self.queue_t.clone().detach())

        # logits Nx(N+K)
        logits_t = torch.cat([l_batch_t, l_queue_t], dim=1)

        # apply temperature
        logits_t /= self.T

        # labels
        labels_t = torch.arange(p_q_t.shape[0], dtype=torch.long).to(device=p_q_t.device)

        # TARGET-SOURCE Contrastive loss:
        # We want the target query (not its projection!) to get closer to its key's NN in source query.

        _, indices_nn = NN(k_t, q_s.clone().detach(), num_neighbors=self.num_neighbors, return_indices=True)

        # logits for NNs: NxN
        logits_ts = torch.mm(q_t, q_s.transpose(0, 1).clone().detach())

        # apply temperature
        logits_ts /= self.T

        # labels
        labels_ts = indices_nn.squeeze(1).to(device=q_t.device)

        # DOMAIN DISCRIMINATION Loss

        domain_label_s = torch.ones((len(q_s), 1)).to(device=q_s.device)
        domain_label_t = torch.zeros((len(q_t), 1)).to(device=q_t.device)

        labels_domain = torch.cat([domain_label_s, domain_label_t], dim=0)

        q_s_reversed = ReverseLayerF.apply(q_s, alpha)
        q_t_reversed = ReverseLayerF.apply(q_t, alpha)

        q_reversed = torch.cat([q_s_reversed, q_t_reversed], dim=0)
        pred_domain = self.discriminator(q_reversed, None)

        # SOURCE Prediction task
        #y_s = self.predictor(q_s, static_s)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_s, k_t)

        return logits_s, labels_s, logits_t, labels_t, logits_ts, labels_ts, pred_domain, labels_domain, q_s

    def get_encoding(self, sequence, is_target=True):
        # compute the encoding of a sequence (i.e. before projection layer)
        # Input is in the shape N*L*C whereas TCN expects N*C*L
        # Since the tcn_out has the shape N*C_out*L, we will get the last timestep of the output

        # We will use the encoder from a given domain (either source or target)

        q = self.encoder_q(sequence)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        return q

    def predict(self, sequence, static, is_target=True):
        # Get the encoding of a sequence from a given domain
        q = self.get_encoding(sequence, is_target=is_target)

        # Make the prediction based on the encoding
        y = self.predictor(q, static)

        return y


class CNN_ATTN(nn.Module):
    def __init__(self, configs):
        super(CNN_ATTN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)
        self.attn_network = attn_network(configs)
        self.sparse_max = Sparsemax(dim=-1)
        self.feat_len = configs.features_len

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        x_flat = x.reshape(x.shape[0], -1)
        attentive_feat = self.calculate_attentive_feat(x_flat)
        return attentive_feat

    def self_attention(self, Q, K, scale=True, sparse=True, k=3):

        attention_weight = torch.bmm(Q.view(Q.shape[0], self.feat_len, -1), K.view(K.shape[0], -1, self.feat_len))

        attention_weight = torch.mean(attention_weight, dim=2, keepdim=True)

        if scale:
            d_k = torch.tensor(K.shape[-1]).float()
            attention_weight = attention_weight / torch.sqrt(d_k)
        if sparse:
            attention_weight_sparse = self.sparse_max(torch.reshape(attention_weight, [-1, self.feat_len]))
            attention_weight = torch.reshape(attention_weight_sparse, [-1, attention_weight.shape[1],
                                                                       attention_weight.shape[2]])
        else:
            attention_weight = self.softmax(attention_weight)

        return attention_weight

    def attention_fn(self, Q, K, scaled=False, sparse=True, k=1):

        attention_weight = torch.matmul(F.normalize(Q, p=2, dim=-1),
                                        F.normalize(K, p=2, dim=-1).view(K.shape[0], K.shape[1], -1, self.feat_len))

        if scaled:
            d_k = torch.tensor(K.shape[-1]).float()
            attention_weight = attention_weight / torch.sqrt(d_k)
            attention_weight = k * torch.log(torch.tensor(self.feat_len, dtype=torch.float32)) * attention_weight

        if sparse:
            attention_weight_sparse = self.sparse_max(torch.reshape(attention_weight, [-1, self.feat_len]))

            attention_weight = torch.reshape(attention_weight_sparse, attention_weight.shape)
        else:
            attention_weight = self.softmax(attention_weight)

        return attention_weight

    def calculate_attentive_feat(self, candidate_representation_xi):
        Q_xi, K_xi, V_xi = self.attn_network(candidate_representation_xi)
        intra_attention_weight_xi = self.self_attention(Q=Q_xi, K=K_xi, sparse=True)
        Z_i = torch.bmm(intra_attention_weight_xi.view(intra_attention_weight_xi.shape[0], 1, -1),
                        V_xi.view(V_xi.shape[0], self.feat_len, -1))
        final_feature = F.normalize(Z_i, dim=-1).view(Z_i.shape[0],-1)

        return final_feature

class attn_network(nn.Module):
    def __init__(self, configs):
        super(attn_network, self).__init__()

        self.h_dim = configs.features_len * configs.final_out_channels
        self.self_attn_Q = nn.Sequential(nn.Linear(in_features=self.h_dim, out_features=self.h_dim),
                                         nn.ELU()
                                         )
        self.self_attn_K = nn.Sequential(nn.Linear(in_features=self.h_dim, out_features=self.h_dim),
                                         nn.LeakyReLU()
                                         )
        self.self_attn_V = nn.Sequential(nn.Linear(in_features=self.h_dim, out_features=self.h_dim),
                                         nn.LeakyReLU()
                                         )

    def forward(self, x):
        Q = self.self_attn_Q(x)
        K = self.self_attn_K(x)
        V = self.self_attn_V(x)

        return Q, K, V


    # Sparse max
class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation

        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=input.device, dtype=input.dtype).view(1,
                                                                                                                     -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input
