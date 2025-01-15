import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.loss import SinkhornDistance, LOT, SupConLoss, SimCLR_Loss
from pytorch_metric_learning import losses
from models.models import ResClassifier_MME, classifier,classifier2
import matplotlib.pyplot as plt
from torch.autograd import Variable

def plot_tensor(x):
    plt.plot((x.detach().cpu().numpy()))
    plt.ylim((-4, 4))
    plt.show()


def plot_mat_tensor(x):
    plt.matshow(torch.abs(x.detach().cpu()))
    plt.show()


def plot_2_tensor(x1, x2, title_str):
    fig, ax = plt.subplots(2, 1)
    ax[0].plot((x1.detach().cpu().numpy()))
    ax[0].set_ylim((-4, 4))
    plt.title(title_str)
    ax[1].plot((x2.detach().cpu().numpy()))
    ax[1].set_ylim((-4, 4))
    plt.show()


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
        self.hann = torch.hamming_window(fl, periodic=False, alpha=0.54, beta=0.46, \
                                         device='cuda')

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]  #Size btch, channel, length
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x = torch.cos(x) #Size btch, channel, length
        # x = self.hann * x
        x_ft = torch.fft.rfft(x, norm='ortho')
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        r = out_ft[:, :, :self.modes1].abs()
        p = out_ft[:, :, :self.modes1].angle()

        # r[:,:,32:] = p[:,:,:32]
        # return torch.concat([r,p],-1), out_ft
        return r, out_ft


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


class CNN_indv(nn.Module):
    def __init__(self, configs,input_channel = 1,mid_channel = 5,output_channel = 10):
        super(CNN_indv, self).__init__()
        self.width = configs.input_channels
        self.channel = configs.input_channels
        self.fl = configs.sequence_len
        self.fc0 = nn.Linear(self.channel, self.width)  # input channel is 2: (a(x), x)
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channel, mid_channel, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(mid_channel),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(mid_channel, mid_channel, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(mid_channel
                           ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(mid_channel, output_channel, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(output_channel),
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
class CNN_list(nn.Module):
    def __init__(self, configs,output_per_channel,no_classes,mid_channel=10):
        super(CNN_list, self).__init__()
        self.output_per_channel = output_per_channel
        self.input_channels =  configs.input_channels
        self.fl = configs.sequence_len

        # input channel is 2: (a(x), x)

        self.ind_cnn_list =  nn.ModuleList([CNN_indv(configs=configs,input_channel=1,mid_channel=mid_channel,output_channel=self.output_per_channel)
                                            for i in range(self.input_channels)])
        self.full_con_layer = nn.Sequential(nn.Linear(self.input_channels*output_per_channel,mid_channel),
                                            nn.ReLU(),nn.Linear(mid_channel,mid_channel))

    def forward(self, x):
        concat_ind = []
        for i in range(1,self.input_channels+1):
            concat_ind.append(self.ind_cnn_list[i-1](x[:,i-1].unsqueeze(1)))
        all_tensors = torch.concat(concat_ind,dim=1)
        x_full = self.full_con_layer(all_tensors)
        return x_full,all_tensors
class tf_encoder(nn.Module):
    def __init__(self, configs,output_per_channel):
        super(tf_encoder, self).__init__()
        self.output_per_channel = output_per_channel
        self.width = configs.input_channels
        self.channel = configs.input_channels
        self.modes1 = configs.fourier_modes
        self.fl = configs.sequence_len
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1, self.fl)
        self.bn_freq = nn.BatchNorm1d(configs.fourier_modes)
        self.bn_freq = nn.LayerNorm(self.modes1)
        self.cnn = CNN(configs).to('cuda')
        self.cnn = CNN_list(configs,output_per_channel=self.output_per_channel,no_classes=configs.num_classes).to('cuda')
        self.con1 = nn.Conv1d(self.width, 1, kernel_size=3,
                              stride=configs.stride, bias=False, padding=(3 // 2))
        self.lin = nn.Linear(configs.final_out_channels, configs.out_dim)
        self.recons = None

    def forward(self, x):
        ef, out_ft = self.conv0(x)
        ef = self.bn_freq(self.con1(ef).squeeze())
        et = self.cnn(x)
        ef = 0 * et[:, 0:64]
        f = torch.concat([ef, et], -1)
        emb,ind_emb = self.cnn(x)
        return F.normalize(emb),ind_emb

class tf_decoder(nn.Module):
    def __init__(self, configs):
        super(tf_decoder, self).__init__()
        self.input_channels, self.sequence_len = configs.input_channels, configs.sequence_len
        self.nn = nn.LayerNorm([self.input_channels, self.sequence_len], eps=1e-04)
        self.fc1 = nn.Linear(64, 3 * 128)
        self.convT = torch.nn.ConvTranspose1d(configs.final_out_channels, self.sequence_len, self.input_channels,
                                              stride=1)
        self.modes = configs.fourier_modes
        self.conv_block1 = nn.Sequential(
            nn.ConvTranspose1d(configs.final_out_channels, configs.mid_channels, kernel_size=3,
                               stride=1),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )
        self.conv_block2 = nn.Sequential(
            nn.ConvTranspose1d(configs.mid_channels, configs.sequence_len, \
                               kernel_size=1, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.lin = nn.Linear(configs.final_out_channels, self.input_channels * self.sequence_len)

    def forward(self, f, out_ft):
        x_low = self.nn(torch.fft.irfft(out_ft, n=128))
        et = f[:, self.modes:]
        # x_high = self.conv_block1(et.unsqueeze(2))
        # x_high = self.conv_block2(x_high).permute(0,2,1)
        # x_high = self.nn2(F.gelu((self.fc1(time).reshape(-1, 3, 128))))
        # print(x_low.shape, time.shape)
        x_high = self.nn(F.relu(self.convT(et.unsqueeze(2))).permute(0, 2, 1))
        # x_high = self.nn(F.relu(self.lin(et).reshape(-1,  self.input_channels, self.sequence_len)))
        return x_low + x_high


class CUSTOM_METHOD(Algorithm):
    def __init__(self, configs, args, device):
        super(CUSTOM_METHOD, self).__init__(configs)
        self.configs = configs
        self.emb_dims = 10
        self.feature_extractor = tf_encoder(configs,output_per_channel=self.emb_dims).to(device)
        #self.decoder = tf_decoder(args).to(device)

        self.classifier = classifier2(configs,emb_dims=self.emb_dims).to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + \
            list(self.classifier.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        self.coptimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) ,
            lr=1 *args.lr,
            weight_decay=args.weight_decay
        )

        self.args = args
        self.recons = nn.L1Loss(reduction='sum').to(device)
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        self.loss_func = losses.ContrastiveLoss(pos_margin=0.5)
        self.sup_cont_loss = SupConLoss(temperature=0.07)
        self.unsup_cont_loss = SimCLR_Loss(batch_size=args.batch_size)
        self.sink = SinkhornDistance(eps=1e-3, max_iter=1000, reduction='sum')
        #self.LOT = LOT(n_src_ancs=6, n_trgt_ancs=6, eps=1e-3, eps_z=1e-3, intensity_vector=[1, 1, 1], device=device)

    def update(self, src_x, src_y, trg_x, trg_y):
        var = 0.01
        self.optimizer.zero_grad()
        src_feat,ind_feat_s= self.feature_extractor(src_x)
        trg_feat,ind_feat_t = self.feature_extractor(trg_x)

        loss_sink_list_cnl = []
        loss_sink_ind_channel = 0
        for i in range(1,self.configs.input_channels+1):
            loss_sink_n_cnl=  self.sink(ind_feat_s[:,(i-1)*self.emb_dims:i*self.emb_dims],
                                               ind_feat_t[:,(i-1)*self.emb_dims:i*self.emb_dims])[0]
            loss_sink_ind_channel = loss_sink_ind_channel + loss_sink_n_cnl

            loss_sink_list_cnl.append(loss_sink_n_cnl)

        loss_sink_ind_channel.backward(retain_graph=True)
        k = 3
        j = 1
        idx_s = torch.where(src_y == k)[0]
        idx_t = torch.where(trg_y == k)[0]

        # plot_mat_tensor(src_x[idx_s[0],:64].reshape(4,-1))
        # plot_mat_tensor(trg_x[idx_t[0],:64].reshape(4,-1))

        k = 4
        j = 1
        idx_s = torch.where(src_y == k)[0]
        idx_t = torch.where(trg_y == k)[0]
        # plot_2_tensor (src_x[idx_s[j], :].T, trg_x[idx_t[j], :].T,title_str = str(k))

        sup_cont_loss = 0 * self.sup_cont_loss(src_feat.unsqueeze(1), src_y)
        sup_cont_loss.backward(retain_graph=True)

        #unsup_trgt_cont_loss = 0 * self.unsup_cont_loss(trg_feat, trg_feat_aug)
        #unsup_trgt_cont_loss.backward(retain_graph=True)

        # dr,Cx,Cy = self.LOT(src_feat, trg_feat)
        dr, _, _ = self.sink(src_feat, trg_feat)
        sink_loss = 0.5 * dr
        sink_loss.backward(retain_graph=True)
        src_pred = self.classifier(src_feat)
        loss_cls = 1 * self.cross_entropy(src_pred,
                                          src_y)
        # + 0.0*torch.mean(torch.norm(src_feat,dim=1,p=1)) + 0.0*torch.mean(torch.norm(trg_feat,dim=1,p=1))
        loss_cls.backward(retain_graph=True)
        self.optimizer.step()
        return {'Src_cls_loss': loss_cls.item(), 'Sink': sink_loss.item(), 'Sink channels':loss_sink_ind_channel}
        # return {'Src_cls_loss': loss_cls.item(), 'Sink': 0}

    def correct(self, src_x, src_y, trg_x):
        self.coptimizer.zero_grad()
        src_feat, out_s = self.feature_extractor(src_x)
        trg_feat, out_t = self.feature_extractor(trg_x)
        src_recon = self.decoder(src_feat, out_s)
        trg_recon = self.decoder(trg_feat, out_t)
        recons = 0e-4 * (self.recons(trg_recon, trg_x) + self.recons(src_recon, src_x))
        recons.backward()
        self.coptimizer.step()
        return {'recon': recons.item()}


class CLUDA(Algorithm):

    def __init__(self, args, input_channels_dim, input_static_dim):

        super(CLUDA, self).__init__(args)

        self.input_channels_dim = input_channels_dim
        self.input_static_dim = input_static_dim

        # different from other algorithms, we import entire model at onces. (i.e. no separate feature extractor or classifier)
        self.model = CLUDA_NN(num_inputs=(1 + args.use_mask) * input_channels_dim, output_dim=self.output_dim,
                              num_channels=self.num_channels, num_static=input_static_dim,
                              mlp_hidden_dim=args.hidden_dim_MLP, use_batch_norm=args.use_batch_norm,
                              kernel_size=args.kernel_size_TCN,
                              stride=args.stride_TCN, dilation_factor=args.dilation_factor_TCN, dropout=args.dropout,
                              K=args.queue_size, m=args.momentum)

        self.augmenter = None
        self.concat_mask = concat_mask

        self.criterion_CL = nn.CrossEntropyLoss()

        self.cuda()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay, betas=(0.5, 0.99)
        )

        self.init_metrics()

    def step(self, sample_batched_src, sample_batched_trg, **kwargs):
        # For Augmenter, Cutout length is calculated relative to the sequence length
        # If there is only one channel, there will be no spatial dropout
        if self.augmenter is None:
            self.get_augmenter(sample_batched_src)

        p = float(kwargs.get("count_step")) / 1000
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # Go through augmentations first
        seq_q_src, seq_mask_q_src = self.augmenter(sample_batched_src['sequence'], sample_batched_src['sequence_mask'])
        seq_k_src, seq_mask_k_src = self.augmenter(sample_batched_src['sequence'], sample_batched_src['sequence_mask'])

        seq_q_trg, seq_mask_q_trg = self.augmenter(sample_batched_trg['sequence'], sample_batched_trg['sequence_mask'])
        seq_k_trg, seq_mask_k_trg = self.augmenter(sample_batched_trg['sequence'], sample_batched_trg['sequence_mask'])

        # Concat mask if use_mask = True
        seq_q_src = self.concat_mask(seq_q_src, seq_mask_q_src, self.args.use_mask)
        seq_k_src = self.concat_mask(seq_k_src, seq_mask_k_src, self.args.use_mask)
        seq_q_trg = self.concat_mask(seq_q_trg, seq_mask_q_trg, self.args.use_mask)
        seq_k_trg = self.concat_mask(seq_k_trg, seq_mask_k_trg, self.args.use_mask)

        # compute output
        output_s, target_s, output_t, target_t, output_ts, target_ts, output_disc, target_disc, pred_s = self.model(
            seq_q_src, seq_k_src, sample_batched_src.get('static'), seq_q_trg, seq_k_trg,
            sample_batched_trg.get('static'), alpha)

        # Compute all losses
        loss_s = self.criterion_CL(output_s, target_s)
        loss_t = self.criterion_CL(output_t, target_t)
        loss_ts = self.criterion_CL(output_ts, target_ts)
        loss_disc = F.binary_cross_entropy(output_disc, target_disc)

        # Task classification  Loss
        src_cls_loss = self.pred_loss.get_prediction_loss(pred_s, sample_batched_src['label'])

        loss = self.args.weight_loss_src * loss_s + self.args.weight_loss_trg * loss_t + \
               self.args.weight_loss_ts * loss_ts + self.args.weight_loss_disc * loss_disc + self.args.weight_loss_pred * src_cls_loss

        # If in training mode, do the backprop
        if self.training:
            # zero grad
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1 = accuracy(output_s, target_s, topk=(1,))
        self.losses_s.update(loss_s.item(), seq_q_src.size(0))
        self.top1_s.update(acc1[0][0], seq_q_src.size(0))

        acc1 = accuracy(output_t, target_t, topk=(1,))
        self.losses_t.update(loss_t.item(), seq_q_trg.size(0))
        self.top1_t.update(acc1[0][0], seq_q_trg.size(0))

        acc1 = accuracy(output_ts, target_ts, topk=(1,))
        self.losses_ts.update(loss_t.item(), seq_q_trg.size(0))
        self.top1_ts.update(acc1[0][0], seq_q_trg.size(0))

        acc1 = accuracy_score(output_disc.detach().cpu().numpy().flatten() > 0.5,
                              target_disc.detach().cpu().numpy().flatten())
        self.losses_disc.update(loss_disc.item(), output_disc.size(0))
        self.top1_disc.update(acc1, output_disc.size(0))

        self.losses_pred.update(src_cls_loss.item(), seq_q_src.size(0))

        pred_meter_src = PredictionMeter(self.args)

        pred_meter_src.update(sample_batched_src['label'], pred_s)

        metrics_pred_src = pred_meter_src.get_metrics()

        self.score_pred.update(metrics_pred_src[self.main_pred_metric], sample_batched_src['sequence'].size(0))

        self.losses.update(loss.item(), sample_batched_src['sequence'].size(0))

        if not self.training:
            # keep track of prediction results (of source) explicitly
            self.pred_meter_val_src.update(sample_batched_src['label'], pred_s)

            # keep track of prediction results (of target) explicitly
            pred_t = self.model.predict(seq_q_trg, sample_batched_trg.get('static'), is_target=True)

            self.pred_meter_val_trg.update(sample_batched_trg['label'], pred_t)

    def init_metrics(self):

        self.losses_s = AverageMeter('Loss Source', ':.4e')
        self.top1_s = AverageMeter('Acc@1', ':6.2f')
        self.losses_t = AverageMeter('Loss Target', ':.4e')
        self.top1_t = AverageMeter('Acc@1', ':6.2f')
        self.losses_ts = AverageMeter('Loss Sour-Tar CL', ':.4e')
        self.top1_ts = AverageMeter('Acc@1', ':6.2f')
        self.losses_disc = AverageMeter('Loss Sour-Tar Disc', ':.4e')
        self.top1_disc = AverageMeter('Acc@1', ':6.2f')
        self.losses_pred = AverageMeter('Loss Pred', ':.4e')
        self.score_pred = self.init_score_pred()
        self.losses = AverageMeter('Loss TOTAL', ':.4e')

    def return_metrics(self):
        return [self.losses_s, self.top1_s, self.losses_t, self.top1_t, self.losses_ts, self.top1_ts,
                self.losses_disc, self.top1_disc, self.losses_pred, self.score_pred, self.losses]

    def save_state(self, experiment_folder_path):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()},
                   os.path.join(experiment_folder_path, "model_best.pth.tar"))

    def load_state(self, experiment_folder_path):
        checkpoint = torch.load(experiment_folder_path + "/model_best.pth.tar")
        self.model.load_state_dict(checkpoint['state_dict'])

    # We need to overwrite below functions for CLUDA
    def predict_trg(self, sample_batched):

        seq_t = self.concat_mask(sample_batched['sequence'], sample_batched['sequence_mask'], self.args.use_mask)
        y_pred_trg = self.model.predict(seq_t, sample_batched.get('static'), is_target=True)

        self.pred_meter_val_trg.update(sample_batched['label'], y_pred_trg, id_patient=sample_batched.get('patient_id'),
                                       stay_hour=sample_batched.get('stay_hour'))

    def predict_src(self, sample_batched):

        seq_s = self.concat_mask(sample_batched['sequence'], sample_batched['sequence_mask'], self.args.use_mask)
        y_pred_src = self.model.predict(seq_s, sample_batched.get('static'), is_target=False)

        self.pred_meter_val_src.update(sample_batched['label'], y_pred_src, id_patient=sample_batched.get('patient_id'),
                                       stay_hour=sample_batched.get('stay_hour'))

    def get_embedding(self, sample_batched):

        seq = self.concat_mask(sample_batched['sequence'], sample_batched['sequence_mask'], self.args.use_mask)
        feat = self.model.get_encoding(seq)

        return feat

    def get_augmenter(self, sample_batched):

        seq_len = sample_batched['sequence'].shape[1]
        num_channel = sample_batched['sequence'].shape[2]
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

