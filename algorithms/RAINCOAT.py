import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.loss import SinkhornDistance
from pytorch_metric_learning import losses
from models.models import ResClassifier_MME, classifier


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()

    def update(self, *args, **kwargs):
        raise NotImplementedError


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

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x = torch.cos(x)
        x_ft = torch.fft.rfft(x, norm='ortho')
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        r = out_ft[:, :, :self.modes1].abs()
        p = out_ft[:, :, :self.modes1].angle()
        return torch.concat([r, p], -1), out_ft


class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()
        self.width = configs.input_channels
        self.channel = configs.input_channels
        self.fl = configs.sequence_len
        self.fc0 = nn.Linear(self.channel, self.width)  # input channel is 2: (a(x), x)
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        x_flat = x.reshape(x.shape[0], -1)
        return x_flat


class tf_encoder(nn.Module):
    def __init__(self, configs):
        super(tf_encoder, self).__init__()
        self.modes1 = configs.fourier_modes  # Number of low-frequency modes to keep
        self.width = configs.input_channels
        self.length = configs.sequence_len
        self.freq_feature = SpectralConv1d(self.width, self.width, self.modes1,
                                           self.length)  # Frequency Feature Encoder
        self.bn_freq = nn.BatchNorm1d(
            configs.fourier_modes * 2)  # It doubles because frequency features contain both amplitude and phase
        self.cnn = CNN(configs).to('cuda')  # Time Feature Encoder
        self.avg = nn.Conv1d(self.width, 1, kernel_size=3,
                             stride=configs.stride, bias=False, padding=(3 // 2))

    def forward(self, x):
        ef, out_ft = self.freq_feature(x)
        ef = F.relu(self.bn_freq(self.avg(ef).squeeze()))

        et = self.cnn(x)
        f = torch.concat([ef, et], -1)
        return F.normalize(f), out_ft


class tf_decoder(nn.Module):
    def __init__(self, configs):
        super(tf_decoder, self).__init__()
        self.input_channels, self.sequence_len = configs.input_channels, configs.sequence_len
        self.bn1 = nn.BatchNorm1d(self.input_channels, self.sequence_len)
        self.bn2 = nn.BatchNorm1d(self.input_channels, self.sequence_len)
        self.convT = torch.nn.ConvTranspose1d(configs.final_out_channels, self.sequence_len, self.input_channels,
                                              stride=1)
        #self.da_type =
        self.modes = configs.fourier_modes

    def forward(self, f, out_ft):
        #x_low = self.bn1(
        #    torch.fft.irfft(out_ft, n=128))

        #equal to length size
        x_low = self.bn1(
            torch.fft.irfft(out_ft, n=self.sequence_len))
        # reconstruct  time series by using low frequency frequency features
        et = f[:, self.modes * 2:]
        x_high = F.relu(self.bn2(self.convT(et.unsqueeze(2)).permute(0, 2,
                                                                     1)))  # reconstruct time series by using time features for high frequency patterns.
        return x_low + x_high


class RAINCOAT(Algorithm):
    def __init__(self, configs, hparams, device):
        super(RAINCOAT, self).__init__(configs)
        self.feature_extractor = tf_encoder(configs).to(device)
        self.decoder = tf_decoder(configs).to(device)
        true_final_out_channels = configs.final_out_channels
        #configs.final_out_channels = 84 #256
        configs.final_out_channels = 128 #for HhAR
        configs.final_out_channels = 256 #for HCI
        self.classifier = classifier(configs).to(device)
        configs.final_out_channels = true_final_out_channels
        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + \
            list(self.decoder.parameters()) + \
            list(self.classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.coptimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + list(self.decoder.parameters()),
            lr=1 * hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.hparams = hparams
        self.recons = nn.L1Loss(reduction='sum').to(device)
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        self.loss_func = losses.ContrastiveLoss(pos_margin=0.5)
        self.sink = SinkhornDistance(eps=1e-3, max_iter=1000, reduction='sum')

    def update(self, src_x, src_y, trg_x):
        self.optimizer.zero_grad()
        # Encode both source and target features via our time-frequency feature encoder
        #src_feat, out_s = self.feature_extractor(src_x)
        #trg_feat, out_t = self.feature_extractor(trg_x)

        src_feat, out_s = self.feature_extractor(src_x)
        trg_feat, out_t = self.feature_extractor(trg_x)
        # Decode extracted features to time series
        src_recon = self.decoder(src_feat, out_s)
        trg_recon = self.decoder(trg_feat, out_t)
        # Compute reconstruction loss
        recons = 1e-4 * (self.recons(src_recon, src_x) + self.recons(trg_recon, trg_x))
        recons.backward(retain_graph=True)
        # Compute alignment loss
        dr, _, _ = self.sink(src_feat, trg_feat)
        sink_loss = dr
        sink_loss.backward(retain_graph=True)
        # Compute classification loss
        src_pred = self.classifier(src_feat)
        loss_cls = self.cross_entropy(src_pred, src_y)
        loss_cls.backward(retain_graph=True)
        self.optimizer.step()
        return {'Total_loss': loss_cls.item(), 'Domain_loss': sink_loss.item(),'Src_cls_loss':loss_cls.item()}

    def correct(self, src_x, src_y, trg_x):
        self.coptimizer.zero_grad()
        src_feat, out_s = self.feature_extractor(src_x)
        trg_feat, out_t = self.feature_extractor(trg_x)
        src_recon = self.decoder(src_feat, out_s)
        trg_recon = self.decoder(trg_feat, out_t)
        recons = 1e-4 * (self.recons(trg_recon, trg_x) + self.recons(src_recon, src_x))
        recons.backward()
        self.coptimizer.step()
        return {'recon': recons.item()}