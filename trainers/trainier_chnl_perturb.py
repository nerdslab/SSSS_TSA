'code for running 1 scenario, but different number of pertubations.'

import torch
import torch.nn.functional as F
import os
from models.loss import SinkhornDistance, LOT
import pandas as pd
import numpy as np
import warnings

import json
import random
import sklearn.exceptions
from visualizations.plot_funcs import plot_input
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import umap
from pathlib import Path

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
import collections
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from dataloader.dataloader import data_generator, few_shot_data_generator, generator_percentage_of_data
from configs.data_model_configs_channel_perturb import get_dataset_class
from configs.hparams import get_hparams_class
from algorithms.utils import fix_randomness, copy_Files, starting_logs, save_checkpoint, _calc_metrics
from algorithms.utils import calc_dev_risk, calculate_risk
from algorithms.algorithms import get_algorithm_class
from algorithms.RAINCOAT import RAINCOAT
from models.models import get_backbone_class
from algorithms.utils import AverageMeter
from sklearn.metrics import f1_score
from torch import nn

torch.backends.cudnn.benchmark = True
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def get_percent_selection(array_weights,corrupted_channels,thresh=0.15):
    weights = torch.diagonal(array_weights, dim1=1, dim2=2)
    weights_filtered = weights[:,corrupted_channels]
    corruputed_weights_count = torch.zeros(corrupted_channels.shape)
    weights_filtered[weights_filtered >=thresh] = 1
    weights_filtered[weights_filtered <= thresh] = 0
    large_weight_corrupt = torch.sum(weights_filtered)

    weights_orig = torch.diagonal(array_weights, dim1=1, dim2=2)
    weights_filtered = weights_orig[:, corrupted_channels]
    list_cut_off = [0,0.10,0.20,0.30,0.40,0.50,0.65,0.70,0.80,0.90,1.0]

    counts, bin_edges = np.histogram(weights_filtered.cpu().numpy() , list_cut_off)

    return large_weight_corrupt,weights_filtered.shape[0]*weights_filtered.shape[1],counts
def balance_samples(x, y):
    no_labels = torch.unique(y)
    no_labels_lens = []
    for i in list(no_labels):
        no_labels_lens.append(len(torch.where(y == i)[0]))

    max_lbls = max(no_labels_lens)
    x_empty = torch.from_numpy(np.asarray([]))
    y_empty = torch.from_numpy(np.asarray([]))
    for ix, i in enumerate(list(no_labels)):
        idx = torch.where(y == i)[0]

        x_temp = x[idx, :, :]

        '''
        if i == 0:
            x_temp[:, :, 2] = 0
        if i == 2:
            x_temp[:,:,0] = 0
            #x_temp[:,:,:] = 0

        elif i == 3:

            x_temp[:,:,0] = 0
        '''
        y_temp = y[idx].reshape(-1, )
        if float(max_lbls / no_labels_lens[ix]) >= 2:
            x_temp = x_temp.repeat(int(2 * max_lbls / no_labels_lens[i]), 1, 1) + torch.normal(mean=0.0, std=0.001,
                                                                                               size=(len(x_temp) * int(
                                                                                                   2 * max_lbls /
                                                                                                   no_labels_lens[i]),
                                                                                                     1, 1)).cuda()
            y_temp = y_temp.repeat(int(2 * max_lbls / no_labels_lens[i]))
        idx = random_integers = np.random.randint(0, x_temp.shape[0], size=20)
        x_empty = torch.concat((x_empty, x_temp[idx, :, :]), 0) if len(x_empty) else x_temp[idx, :, :]

        y_empty = torch.concat((y_empty, y_temp[idx]), 0) if len(y_empty) else y_temp[idx]

    return x_empty, y_empty


class cross_domain_trainer(object):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        self.da_method = args.da_method  # Selected  DA Method
        self.dataset = args.dataset  # Selected  Dataset
        self.backbone = args.backbone
        self.device = torch.device(args.device)  # device
        self.sinkdis = SinkhornDistance(eps=1e-3, max_iter=1000, reduction='sum')
        self.cross_entropy = nn.CrossEntropyLoss()
        self.run_description = args.run_description
        self.experiment_description = args.experiment_description
        self.satur_value = 2
        self.best_acc = 0
        self.best_val_loss = 1e10
        # paths
        self.home_path = os.getcwd()
        self.save_dir = args.save_dir
        self.data_path = os.path.join(args.data_path, self.dataset)
        self.create_save_dir()
        self.src_weight_wrong_selection_count = []
        self.src_weight_wrong_selection_all = []

        self.trg_weight_wrong_selection_count = []
        self.trg_weight_wrong_selection_all = []
        # Specify runs
        self.num_runs = args.num_runs

        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()




        self.add_gauss = args.add_gauss
        self.drop_chnl  = args.drop_chnl
        self.satur_chnl = args.satur_chnl
        if self.add_gauss and self.drop_chnl:
            raise ValueError("Can not set both Gauss  add and Drop channel at same time for the training run")
        if self.add_gauss:
            if hasattr(self.dataset_configs, "no_channel_affect"):

                self.no_channel_affect = self.dataset_configs.no_channel_affect
                # self.chnl_drop_list = np.random.choice(np.arange(0, self.dataset_configs.input_channels), self.dataset_configs.no_channel_affect,
                #                                       replace=False)
                self.noise = self.dataset_configs.noise
                # containting all channel list
                self.channel_affect_list = []
                # for the current channel affected list used by subfucntions
                self.current_channel_affect_list = []


        elif self.drop_chnl:

            if hasattr(self.dataset_configs, "no_channel_affect"):

                self.no_channel_affect = self.dataset_configs.no_channel_affect
                self.channel_affect_list = []
                self.current_channel_affect_list = []

        elif self.satur_chnl:
            if hasattr(self.dataset_configs, "no_channel_affect"):

                self.no_channel_affect = self.dataset_configs.no_channel_affect
                self.channel_affect_list = []
                self.current_channel_affect_list = []
                # self.chnl_drop_list = np.random.choice(np.arange(0,self.dataset_configs.input_channels),self.no_channel_affect,replace=False )
            else:

                self.no_channel_affect = 0
                self.channel_affect_list = 0
                self.current_channel_affect_list = 0
        # to fix dimension of features in classifier and discriminator networks.
        self.dataset_configs.final_out_channels = self.dataset_configs.tcn_final_out_channles if args.backbone == "TCN" else self.dataset_configs.final_out_channels

        # Specify number of hparams (Requires da method)
        self.default_hparams = {**self.hparams_class.alg_hparams[self.da_method],
                                **self.hparams_class.train_params}

    def train(self, args=None):

        result_dict = {}


        config_vars = vars(args)
        config_dataset = vars(self.dataset_configs)
        config_run = self.default_hparams
        balanced = config_dataset['balanced']
        config = config_vars


        now = datetime.now()
        dt_string = now.strftime("%d_%m_%YTime_%H:%M:%S")

        print("date and time =", dt_string)

        if balanced:
            self.dataset = self.dataset + '_balanced'

        if self.add_gauss:
            self.dataset = f"{self.dataset}'_add_gauss_{self.no_channel_affect}"
        elif self.drop_chnl:
            self.dataset = f"{self.dataset}'_drop_chnl_{self.no_channel_affect}"
        elif self.satur_chnl:
            self.dataset = f"{self.dataset}'_satur_chnl_{self.no_channel_affect}"
        exp_name = self.da_method + '_' + self.dataset + '_' + dt_string

        run_name = f"{self.run_description}"
        self.hparams = self.default_hparams
        # Logging
        self.exp_log_dir = os.path.join(self.save_dir, self.experiment_description, run_name)
        os.makedirs(self.exp_log_dir, exist_ok=True)

        scenarios = self.dataset_configs.scenarios  # return the scenarios given a specific dataset.
        df_a = pd.DataFrame(columns=['scenario', 'run_id', 'accuracy', 'f1', 'H-score'])


        if self.add_gauss:
            path_string =f"./results/drop_channels/add_gaussian"
        elif self.drop_chnl:
            path_string = f"./results/drop_channels/drop_channel"
        elif self.satur_chnl:
            path_string = f"./results/drop_channels/sat_channel"
        else:
            path_string = f"./results/drop_channels/"
        result_path = Path(path_string)
        result_path.mkdir(parents=True, exist_ok=True)
        result_path= f"{result_path}/{exp_name}.json"


        self.trg_acc_list = []

        f1_list = []
        f1_best_list = []
        f1_best_val_list = []

        f1_list_all = []


        f1_best_list_all = []
        f1_best_val_list_all = []

        f1_list_run_all = []

        f1_best_list_run_all = []


        f1_list_std = []
        f1_best_list_std = []
        f1_best_val_list_std = []

        acc_list = []
        acc_best_list = []
        acc_best_val_list = []

        acc_list_std = []
        acc_best_list_std = []
        acc_best_val_list_std = []

        cm_list = []
        scenario_list = []


        for no_chnl_drop in self.dataset_configs.no_channel_affect:
            #assuming only 1 scenario that would be modified
            src_id = scenarios[0][0]
            trg_id = scenarios[0][1]
            src_weight_over_run_all = []
            trg_weight_over_run_all = []

            src_weight_tot_run_all = []
            trg_weight_tot_run_all = []

            src_weight_hist_counts = np.asarray([])
            trg_weight_hist_counts = np.asarray([])

            src_weight_hist_edges = []
            trg_weight_hist_edges = []
            print(f"Channels affected: {no_chnl_drop}")

            dict = {}
            loggers = {}
            f1_list_run = []



            f1_list_run_best_val = []
            acc_list_run = []
            f1_list_run_best = []
            acc_list_run_best = []
            acc_list_run_best_val = []
            for run_id in range(0, self.num_runs):
                # specify number of consecutive runs
                # fixing random seed
                # run_id = 2025
                # run_id = 8

                self.f1_run_score = []
                torch.cuda.empty_cache()
                fix_randomness(run_id)

                self.chnl_drop_list = np.random.choice(np.arange(0, self.dataset_configs.input_channels),
                                                       no_chnl_drop,replace=False)
                trg_id_str = f"{trg_id}_no_chnl_{no_chnl_drop}"



                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir,
                                                                   src_id, trg_id, run_id)
                path_save = os.path.join(self.home_path, self.scenario_log_dir, self.dataset)
                if not os.path.exists(path_save):
                    os.mkdir(path_save)
                self.fpath = os.path.join(self.home_path, self.scenario_log_dir, self.dataset,'backbone.pth')
                self.cpath = os.path.join(self.home_path, self.scenario_log_dir,self.dataset, 'classifier.pth')
                self.best_acc = 0
                self.best_val_loss = 1e10
                # Load data
                self.load_data(src_id, trg_id)

                # get algorithm

                backbone_fe = get_backbone_class(self.backbone)
                if self.da_method == 'RAINCOAT':
                    algorithm = RAINCOAT(self.dataset_configs, self.hparams, self.device)
                else:
                    algorithm_class = get_algorithm_class(self.da_method)
                    algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
                algorithm.to(self.device)
                self.algorithm = algorithm
                # Average meters
                loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # training..
                # self.eval()
                loss_total = []
                Dom_loss = []
                Src_cls_loss = []


                for epoch in range(1, self.hparams["num_epochs"] + 1):  # self.hparams["num_epochs"] + 1):
                    joint_loaders = enumerate(zip(self.src_train_dl, self.trg_train_dl))
                    len_dataloader = min(len(self.src_train_dl), len(self.trg_train_dl))
                    algorithm.train()

                    for step, ((src_x, src_y), (trg_x, trg_y)) in joint_loaders:
                        src_x, src_y, trg_x, trg_y = src_x.float().to(self.device), src_y.long().to(self.device), \
                                                     trg_x.float().to(self.device), trg_y.to(self.device)

                        if config['plot_input_sig']:
                            plot_input(src_x, trg_x, src_y, trg_y, self.dataset, src_id=src_id, trg_id=trg_id)
                            plt.show()
                            print("Plotting raw source and target inputs")

                        if balanced:
                            try:
                                src_x, src_y = balance_samples(src_x, src_y)
                                trg_x, trg_y = balance_samples(trg_x, trg_y)
                            except IndexError:
                                continue
                        if self.add_gauss:
                            trg_x[:, self.chnl_drop_list, :] = trg_x[:, self.chnl_drop_list, :] + \
                                                               torch.normal(0, self.noise,
                                                                            size=trg_x[:, self.chnl_drop_list,
                                                                                 :].shape).to(self.device)
                        elif self.drop_chnl:
                            trg_x[:, self.chnl_drop_list, :] = 0

                        elif self.satur_chnl :
                            trg_x[:, self.chnl_drop_list, :] = self.satur_value
                        len_min = min(len(src_x), len(trg_x))

                        if self.da_method in ["DANN", "CoDATS", "AdaMatch", "SepReps", "CLUDA"]:
                            losses = algorithm.update(src_x, src_y, trg_x, step, epoch, len_dataloader)
                        elif self.da_method in {"SSSS_TSA","CDAN", "MMDA", "Supervised", "SinkDiv_Alignment", "DDC", "AdvSKM",
                                                "RAINCOAT", "Deep_Coral"}:
                            losses = algorithm.update(src_x, src_y, trg_x)
                        else:
                            losses = algorithm.update(src_x, src_y, trg_x, trg_y, step, epoch, len_dataloader)

                        for key, val in losses.items():
                            loss_avg_meters[key].update(val, src_x.size(0))



                    losses_val = self.loss_val()


                    # loss_total.append(losses['Total_loss'])
                    # Dom_loss.append(losses[ 'Domain_loss'])
                    # Src_cls_loss.append(losses['Src_cls_loss'])
                    # Src_cls_loss.append(losses)
                    # logging
                    self.logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
                    if self.da_method == 'RAINCOAT':
                        (acc, f1, cm), (acc_src, f1_src, cm_src) = self.eval()

                    elif self.da_method == "SepReps":
                        acc, f1, cm = self.algorithm.eval(self.trg_test_dl)
                    else:

                        (acc, f1, cm), (acc_src, f1_src, cm_src) = self.evaluate()
                    if f1 >= self.best_acc:
                        self.best_acc = f1
                        print(self.best_acc)
                        torch.save(self.algorithm.feature_extractor.state_dict(), self.fpath)
                        torch.save(self.algorithm.classifier.state_dict(), self.cpath)

                    if losses_val['Total_loss'] <= self.best_val_loss:
                        torch.save(self.algorithm.feature_extractor.state_dict(), f"{self.fpath}_best_val")
                        torch.save(self.algorithm.classifier.state_dict(), f"{self.cpath}_best_val")
                    self.f1_run_score.append(f1)



                if self.da_method == "SepReps":
                    acc, f1, cm = self.algorithm.eval(self.trg_test_dl, self.fpath, self.cpath, final=False)
                    acc_best, f1_trg_best, cm_best = self.algorithm.eval(self.trg_test_dl, self.fpath, self.cpath,
                                                                         final=True)
                elif self.da_method == 'RAINCOAT':

                    if self.da_method == 'RAINCOAT':
                        print("===== Correct ====")
                        for epoch in range(1, 100):  # self.hparams["num_epochs"] + 1):
                            joint_loaders = enumerate(zip(self.src_train_dl, self.trg_train_dl))
                            len_dataloader = min(len(self.src_train_dl), len(self.trg_train_dl))
                            algorithm.train()
                            for step, ((src_x, src_y), (trg_x, _)) in joint_loaders:
                                src_x, src_y, trg_x = src_x.float().to(self.device), src_y.long().to(self.device), \
                                                      trg_x.float().to(self.device)
                                algorithm.correct(src_x, src_y, trg_x)
                            (acc, f1, cm), (acc_src, f1_src, cm_src) = self.eval()
                            if f1 >= self.best_acc:
                                self.best_acc = f1
                                # print(self.best_acc)
                                torch.save(self.algorithm.feature_extractor.state_dict(), self.fpath)
                                torch.save(self.algorithm.classifier.state_dict(), self.cpath)

                        (acc, f1, cm), (acc_src, f1_src, cm_src) = self.eval(final=False)

                        (acc_best, f1_trg_best, cm_best), (acc_src, f1_src_best, cm_src) = self.eval(final=True)
                        (acc_best_val, f1_best_val, cm_best_val), (
                        acc_src_best_val, f1_src_best_val, cm_src_best_val) = self.eval(best_val=True)
                        # cm = cms[0]
                else:
                    (acc, f1, cm), (acc_src, f1_src, cm_src) = self.evaluate(final=False)

                    (acc_best, f1_trg_best, cm_best), (acc_src, f1_src_best, cm_src) = self.evaluate(final=True)
                    (acc_best_val, f1_best_val, cm_best_val), (
                    acc_src_best_val, f1_src_best_val, cm_src_best_val) = self.evaluate(best_val=True)
                f1_list_run.append(f1)
                f1_list_run_best.append(f1_trg_best)
                f1_list_run_best_val.append(f1_best_val)
                cm_list.append(cm)

                acc_list_run.append(acc)
                acc_list_run_best.append(acc_best)
                acc_list_run_best_val.append(acc_best_val)

                # plt.plot(self.f1_run_score)
                # plt.title(f"{self.da_method}..{src_id}_to_{trg_id}")
                #

                vis = 0
                if vis:
                    (greater_weights_src, total_weights_src),  (greater_weights_trg, total_weights_trg),(hist_count_src,hist_count_trg) =self.visualize()
                    src_weight_over_run_all.append(greater_weights_src.cpu().item())
                    trg_weight_over_run_all.append(greater_weights_trg.cpu().item())

                    #src_weight_over_run_all.append(greater_weights_src)
                    src_weight_tot_run_all.append(total_weights_src)
                    trg_weight_tot_run_all.append(total_weights_trg)

                    src_weight_hist_counts = np.concatenate((src_weight_hist_counts,hist_count_src.reshape(1,-1)),axis=0) if len(src_weight_hist_counts) else hist_count_src.reshape(1,-1)
                    trg_weight_hist_counts = np.concatenate((trg_weight_hist_counts,hist_count_trg.reshape(1,-1)),axis=0) if len(trg_weight_hist_counts) else hist_count_trg.reshape(1,-1)
                # plt.show()

                # f1_list.append(f1)
                # log = {'scenario':i,'run_id':run_id,'accuracy':acc,'f1':f1}
                # df_a = df_a.append(log, ignore_index=True)
                # print("visualization after correction")
            f1_list_run_all.append(f1_list_run)
            f1_list.append(np.mean(f1_list_run))
            f1_list_std.append(np.std(f1_list_run))

            acc_list.append(np.mean(acc_list_run))
            acc_list_std.append(np.std(acc_list_run))

            f1_best_list_run_all.append(f1_list_run_best)
            f1_best_list.append(np.mean(f1_list_run_best))
            f1_best_list_std.append(np.std(f1_list_run_best))

            acc_best_list.append(np.mean(acc_list_run_best))
            acc_best_list_std.append(np.std(acc_list_run_best))

            f1_best_val_list.append(np.mean(f1_list_run_best_val))
            f1_best_val_list_std.append(np.std(f1_list_run_best_val))

            acc_best_val_list.append(np.mean(acc_list_run_best_val))
            acc_best_val_list_std.append(np.std(acc_list_run_best_val))

            f1_best_list_all.append(f1_list_run_best)
            f1_list_all.append(f1_list_run)

            scenario_list.append(f"{src_id} to {trg_id_str}")
            # mean_acc, std_acc, mean_f1, std_f1 = self.avg_result(df_a,'average_align.csv')
            # print("\n\n End of training results (no trgt labels to stop)")

            print(f1_list)
            print(acc_list)

            print("F1 Mean of {len(f1_list}} each case:")
            for i in range(0, len(f1_list)):
                print(f"{f1_list[i]}")

            print("\nAccuracy Mean of each case:")
            for i in range(0, len(acc_list)):
                print(f"{acc_list[i]}")

            print(f"F1 Mean: {np.mean(f1_list)}, std: {np.mean(f1_list_std)}")
            print(f"Accuracy Mean: {np.mean(acc_list)}, std: {np.mean(acc_list_std)}")

            print(f1_list)
            print(acc_list)
            print("\n\n Results: Best on hold out set (val target labels to stop)")

            print(f"F1 Mean: {np.mean(f1_best_list)}, std: {np.mean(f1_best_list_std)}")
            print(f"Accuracy Mean: {np.mean(acc_best_list)}, std: {np.mean(acc_best_list_std)}")

            print("F1 Mean of each case")
            for i in range(0, len(f1_best_list)):
                print(f"{f1_best_list[i]}")

            print("\nAccuracy Mean of each case")
            for i in range(0, len(acc_best_list)):
                print(f"{acc_best_list[i]}")

            # log = {'scenario':mean_acc,'run_id':std_acc,'accuracy':mean_f1,'f1':std_f1}
            # df_a = df_a.append(log, ignore_index=True)
            # print(df_a)
            # path =  os.path.join(self.exp_log_dir, 'average_align.csv')
            # df_a.to_csv(path,sep = ',')
            print(f1_best_list)
            print(acc_best_list)

            result_dict["method"] = self.da_method
            result_dict["dataset"] = self.dataset


            result_dict['best_val_trg_lbl'] = {}
            result_dict["end_train"] = {}
            result_dict["best_val_loss_align"] = {}
            result_dict["scenario_list"] = scenario_list

            result_dict['best_val_trg_lbl']['f1_mean'] = np.mean(f1_best_list)
            result_dict['best_val_trg_lbl']['f1_mean_std'] = np.mean(f1_best_list_std)
            result_dict['best_val_trg_lbl']['result List'] = f1_best_list
            result_dict['best_val_trg_lbl']['F1result ListAll'] = f1_best_list_run_all

            result_dict['best_val_trg_lbl']['acc_mean'] = np.mean(acc_best_list)
            result_dict['best_val_trg_lbl']['acc_std'] = np.mean(acc_best_list_std)
            result_dict['best_val_trg_lbl']['result_list_acc'] = acc_best_list

            result_dict['end_train']['f1_mean'] = np.mean(f1_list)
            result_dict['end_train']['f1_mean_std'] = np.mean(f1_list_std)
            result_dict['end_train']['result List'] = f1_list
            result_dict['end_train']['F1result ListAll'] = f1_list_run_all

            result_dict['end_train']['acc_mean'] = np.mean(acc_list)
            result_dict['end_train']['acc_mean_std'] = np.mean(acc_list_std)
            result_dict['end_train']['result List acc'] = acc_list

            result_dict['best_val_loss_align']['f1_mean'] = np.mean(f1_best_val_list)
            result_dict['best_val_loss_align']['f1_mean_std'] = np.mean(f1_best_val_list_std)
            result_dict['best_val_loss_align']['result List'] = f1_best_val_list

            result_dict['best_val_loss_align']['acc_mean'] = np.mean(acc_best_val_list)
            result_dict['best_val_loss_align']['acc_mean_std'] = np.mean(acc_best_val_list_std)
            result_dict['best_val_loss_align']['result List acc'] = acc_best_val_list

            with open(result_path, "w") as file:
                # Use json.dump() to write JSON data with formatting
                json.dump(result_dict, file, indent=4, sort_keys=False)
            print(f"Saved results in {result_path} ")



    def visualize(self):
        # set to if plot or not. Other visualizaitons only for
        plot = 0
        visualize_chnl_algn = 1

        reducer = umap.UMAP()
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)

        # classifier.load_state_dict(torch.load(self.cpath))
        # feature_extractor.load_state_dict(torch.load(self.fpath))
        feature_extractor.eval()
        classifier.eval()

        self.trg_pred_labels = np.array([])

        self.trg_true_labels = np.array([])
        self.trg_all_features = []
        self.src_true_labels = np.array([])
        self.src_all_features = []
        self.src_all_data = torch.from_numpy(np.asarray([]))
        self.trg_all_data = torch.from_numpy(np.asarray([]))
        self.src_all_weights = torch.from_numpy(np.asarray([]))
        self.trg_all_weights = torch.from_numpy(np.asarray([]))
        with torch.no_grad():
            # for data, labels in self.trg_test_dl:
            for data, labels in self.trg_test_dl:
                data = data.float().to(self.device)
                if self.add_gauss:
                    data[:, self.chnl_drop_list, :] = data[:, self.chnl_drop_list, :] + \
                                                      torch.normal(0, self.noise, \
                                                                   size=data[:, self.chnl_drop_list, :].shape).to( \
                                                          self.device)
                elif self.drop_chnl:
                    data[:, self.chnl_drop_list, :] = 0

                elif self.satur_chnl:
                    data[:, self.chnl_drop_list, :] = self.satur_value

                self.trg_all_data = torch.concat((self.trg_all_data, data), dim=0) if len(self.trg_all_data) else data
                labels = labels.view((-1)).long().to(self.device)
                features = feature_extractor(data)
                self.trg_all_features.append(features.cpu().numpy())
                self.trg_true_labels = np.append(self.trg_true_labels, labels.data.cpu().numpy())

            for data, labels in self.src_test_dl:
                data = data.float().to(self.device)

                self.src_all_data = torch.concat((self.src_all_data, data), dim=0) if len(self.src_all_data) else data
                labels = labels.view((-1)).long().to(self.device)
                # features = feature_extractor(data)[0]
                features = feature_extractor(data)
                self.src_all_features.append(features.cpu().numpy())
                self.src_true_labels = np.append(self.src_true_labels, labels.data.cpu().numpy())
            self.src_all_features = np.vstack(self.src_all_features)
            self.trg_all_features = np.vstack(self.trg_all_features)
            # dr, Cx, Cy = self.LOT(torch.from_numpy(self.src_all_features).to(self.device), torch.from_numpy(self.trg_all_features).to(self.device))
            transformed_data = np.concatenate(
                (self.src_all_features, self.trg_all_features), axis=0)
            # transformed_data = np.concatenate((self.src_all_features,self.trg_all_features,Cx.detach().cpu().numpy(),Cy.detach().cpu().numpy()),axis=0)
            src_trgt_lbl = np.concatenate(
                (0 * np.ones(len(self.src_all_features)), 1 * np.ones(len(self.trg_all_features))), axis=0)
            # src_trgt_lbl = np.concatenate((0 * np.ones(len(self.src_all_features)),
            #                               1 * np.ones(len(self.trg_all_features)), 2 * np.ones(len(Cx)),
            #                               3 * np.ones(len(Cy)),), axis=0)

            if plot == 1:
                transformed_data = reducer.fit_transform(transformed_data.astype(dtype='float32'))

                labels = np.concatenate((self.src_true_labels, self.trg_true_labels), axis=0)

                fig1, axs2 = plt.subplots(1, 3, figsize=(15, 5))

                axs2[0].scatter(transformed_data[:len(self.src_all_features), 0],
                                transformed_data[:len(self.src_all_features), 1], c='red', alpha=0.6)
                axs2[0].scatter(transformed_data[len(self.src_all_features):, 0],
                                transformed_data[len(self.src_all_features):, 1], c='blue', alpha=0.6)
                clr_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink']
                label_list = ['Upstairs', 'Jogging', 'Sitting', 'standing', 'Downstairs', 'Walking']
                for j in range(0, self.algorithm.configs.num_classes):
                    j_idx = np.where((labels == j) & (src_trgt_lbl == 0))[0]
                    axs2[1].scatter(transformed_data[j_idx, 0], transformed_data[j_idx, 1], c=clr_list[j],
                                    label=label_list[j],
                                    alpha=0.6)
                for j in range(0, self.algorithm.configs.num_classes):
                    j_idx = np.where((labels == j) & (src_trgt_lbl == 1))[0]
                    axs2[2].scatter(transformed_data[j_idx, 0], transformed_data[j_idx, 1], c=clr_list[j],
                                    label=label_list[j],
                                    alpha=0.6)

                x_min = min(np.concatenate((transformed_data[:, 0], transformed_data[:, 0])))
                x_max = max(np.concatenate((transformed_data[:, 0], transformed_data[:, 0])))

                y_min = min(np.concatenate((transformed_data[:, 1], transformed_data[:, 1])))
                y_max = max(np.concatenate((transformed_data[:, 1], transformed_data[:, 1])))
                axs2[1].set_title("Source")
                axs2[2].set_title("Target")
                axs2[0].set_title("Combined Domains")

                for k in range(0, self.algorithm.configs.input_channels):
                    axs2[k].set_ylim((y_min - 1, y_max + 1))
                    axs2[k].set_xlim((x_min - 1, x_max + 1))
                axs2[2].legend()

            if visualize_chnl_algn and self.da_method == 'SSSS_TSA':
                'visualoize channel specific alignment, and attention scores. Made for our custom method. Fails for '

                preds_trg = classifier(torch.from_numpy(self.trg_all_features).to(0)).argmax(dim=1).detach().cpu()
                preds_src = classifier(torch.from_numpy(self.src_all_features).to(0)).argmax(dim=1).detach().cpu()
                print("Source CM")
                cm_src = confusion_matrix(self.src_true_labels, preds_src)
                print(cm_src)
                print("Target CM")
                cm_trg = confusion_matrix(self.trg_true_labels, preds_trg)
                print(cm_trg)
                f1_trg = f1_score(self.trg_true_labels, preds_trg, average='macro')
                f1_src = f1_score(self.src_true_labels, preds_src, average='macro')
                # torch.sum(self.algorithm.domain_classifier(torch.from_numpy(self.trg_all_features[trg_j]).to(0)).argmax(
                #    dim=1) == 0)
                # self.algorithm.feature_extractor.fetch_att_weights(self.trg_all_data[trg_j, :, :])
                prob_trg, pred_trg = self.algorithm.get_ind_scores(self.trg_all_data)
                prob_src, pred_src = self.algorithm.get_ind_scores(self.src_all_data)
                # plt.show()

                f1_list_chnl_src = []
                f1_list_chnl_trg = []

                for k in range(0, self.algorithm.configs.input_channels):
                    print(f" CM Source channel {k} ")
                    cm_src = confusion_matrix(self.src_true_labels, pred_src[k])

                    print(cm_src)

                    f1_list_chnl_src.append(f1_score(self.src_true_labels, pred_src[k], average='macro'))
                    print(f" CM Target channel {k} ")
                    cm_trg = confusion_matrix(self.trg_true_labels, pred_trg[k])
                    f1_list_chnl_trg.append(f1_score(self.trg_true_labels, pred_trg[k], average='macro'))
                    print(cm_trg)
                src_attn = np.asarray([])
                trg_attn = np.asarray([])
                for j in range(0, self.algorithm.configs.num_classes):
                    trg_j = torch.where(torch.from_numpy(self.trg_true_labels) == j)[0]
                    src_j = torch.where(torch.from_numpy(self.src_true_labels) == j)[0]

                    attn_src = self.algorithm.feature_extractor.fetch_att_weights(self.src_all_data[src_j, :, :])
                    attn_trg = self.algorithm.feature_extractor.fetch_att_weights(self.trg_all_data[trg_j, :, :])
                    self.src_all_weights = torch.concat((attn_src,self.src_all_weights),dim=0) if len(self.src_all_weights) else attn_src
                    self.trg_all_weights = torch.concat((attn_trg, self.trg_all_weights), dim=0) if len(
                        self.trg_all_weights) else attn_trg
                greater_weights_src,total_weights_src,hist_counts_src =get_percent_selection(self.src_all_weights , self.chnl_drop_list)
                greater_weights_trg, total_weights_trg,hist_counts_trg = get_percent_selection(self.trg_all_weights,
                                                                                 self.chnl_drop_list)

                print("Here. Stop to analyze channel alignment")
        return  (greater_weights_src,total_weights_src),(greater_weights_trg, total_weights_trg),(hist_counts_src,hist_counts_trg)
    def evaluate(self, final=False, best_val=False):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)
        # self.algorithm.ema.apply_shadow()
        # # evaluate
        # self.algorithm.ema.restore()
        if final == True:
            feature_extractor.load_state_dict(torch.load(self.fpath))
            classifier.load_state_dict(torch.load(self.cpath))
            dloader = self.trg_test_dl
        else:
            dloader = self.trg_val_dl

        if best_val == True:
            feature_extractor.load_state_dict(torch.load(f"{self.fpath}_best_val"))
            classifier.load_state_dict(torch.load(f"{self.cpath}_best_val"))
            dloader = self.trg_test_dl
        feature_extractor.eval()
        classifier.eval()

        total_loss_ = []

        self.trg_pred_labels = np.array([])
        self.trg_true_labels = np.array([])
        self.trg_x_feats = torch.from_numpy(np.array([]))
        with torch.no_grad():
            for data, labels in dloader:
                data = data.float().to(self.device)
                if self.add_gauss:
                    data[:, self.chnl_drop_list, :] = data[:, self.chnl_drop_list, :] + \
                                                      torch.normal(0, self.noise, \
                                                                   size=data[:, self.chnl_drop_list, :].shape).to( \
                                                          self.device)
                elif self.drop_chnl:
                    data[:, self.chnl_drop_list, :] = 0
                elif self.satur_chnl:
                    data[:, self.chnl_drop_list, :] = self.satur_value
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = feature_extractor(data)
                predictions = classifier(features)

                # compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss_.append(loss.item())
                pred = predictions.detach().argmax(dim=1)  # get the index of the max log-probability

                self.trg_pred_labels = np.append(self.trg_pred_labels, pred.cpu().numpy())
                self.trg_true_labels = np.append(self.trg_true_labels, labels.data.cpu().numpy())
        accuracy_trg = accuracy_score(self.trg_true_labels, self.trg_pred_labels)
        f1_trg = f1_score(self.trg_true_labels, self.trg_pred_labels, pos_label=None, average="macro")
        cm_trg = confusion_matrix(self.trg_true_labels, self.trg_pred_labels, normalize=None)

        if final == True:
            feature_extractor.load_state_dict(torch.load(self.fpath))
            classifier.load_state_dict(torch.load(self.cpath))
            dloader = self.src_test_dl
        else:
            dloader = self.src_val_dl

        total_loss_ = []

        self.src_pred_labels = np.array([])
        self.src_true_labels = np.array([])
        self.src_x_feats = torch.from_numpy(np.array([]))
        with torch.no_grad():
            for data, labels in dloader:
                data = data.float().to(self.device)

                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = feature_extractor(data)
                predictions = classifier(features)

                # compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss_.append(loss.item())
                pred = predictions.detach().argmax(dim=1)  # get the index of the max log-probability

                self.src_pred_labels = np.append(self.src_pred_labels, pred.cpu().numpy())
                self.src_true_labels = np.append(self.src_true_labels, labels.data.cpu().numpy())
        accuracy_src = accuracy_score(self.src_true_labels, self.src_pred_labels)
        f1_src = f1_score(self.src_true_labels, self.src_pred_labels, pos_label=None, average="macro")
        cm_src = confusion_matrix(self.src_true_labels, self.src_pred_labels, normalize=None)

        return (accuracy_trg * 100, f1_trg, cm_trg), (accuracy_src * 100, f1_src, cm_src)

    def eval(self, final=False, best_val=False):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)
        if final == True:
            feature_extractor.load_state_dict(torch.load(self.fpath))
            classifier.load_state_dict(torch.load(self.cpath))
            dloader = self.trg_test_dl
        elif best_val == True:
            feature_extractor.load_state_dict(torch.load(f"{self.fpath}_best_val"))
            classifier.load_state_dict(torch.load(f"{self.cpath}_best_val"))
            dloader = self.trg_test_dl
        else:
            dloader = self.trg_val_dl
        feature_extractor.eval()
        classifier.eval()

        total_loss_ = []

        self.trg_pred_labels = np.array([])
        self.trg_true_labels = np.array([])

        with torch.no_grad():
            for data, labels in dloader:
                data = data.float().to(self.device)
                if self.add_gauss:
                    data[:, self.chnl_drop_list, :] = data[:, self.chnl_drop_list, :] + \
                                                      torch.normal(0, self.noise, \
                                                                   size=data[:, self.chnl_drop_list, :].shape).to( \
                                                          self.device)
                elif self.drop_chnl:
                    data[:, self.chnl_drop_list, :] = 0
                elif self.satur_chnl:
                    data[:, self.chnl_drop_list, :] = self.satur_value
                labels = labels.view((-1)).long().to(self.device)

                # forward pass

                features, _ = feature_extractor(data)

                predictions = classifier(features)

                # compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss_.append(loss.item())
                pred = predictions.detach().argmax(dim=1)  # get the index of the max log-probability

                self.trg_pred_labels = np.append(self.trg_pred_labels, pred.cpu().numpy())
                self.trg_true_labels = np.append(self.trg_true_labels, labels.data.cpu().numpy())
        accuracy = accuracy_score(self.trg_true_labels, self.trg_pred_labels)
        f1 = f1_score(self.trg_true_labels, self.trg_pred_labels, pos_label=None, average="macro")
        confusion_matrix_trgt = confusion_matrix(self.trg_true_labels, self.trg_pred_labels)
        self.src_pred_labels = np.array([])
        self.src_true_labels = np.array([])

        with torch.no_grad():
            for data, labels in self.src_test_dl:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features, _ = feature_extractor(data)

                predictions = classifier(features)

                # compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss_.append(loss.item())
                pred = predictions.detach().argmax(dim=1)  # get the index of the max log-probability

                self.src_pred_labels = np.append(self.src_pred_labels, pred.cpu().numpy())
                self.src_true_labels = np.append(self.src_true_labels, labels.data.cpu().numpy())
        accuracy_src = accuracy_score(self.src_true_labels, self.src_pred_labels)
        f1_src = f1_score(self.src_pred_labels, self.src_true_labels, pos_label=None, average="macro")
        confusion_matrix_src = confusion_matrix(self.src_true_labels, self.src_pred_labels)

        return (accuracy * 100, f1, confusion_matrix_trgt), (accuracy_src, f1_src, confusion_matrix_src)

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.dataset)
        return dataset_class(), hparams_class()

    def load_data(self, src_id, trg_id):
        self.src_train_dl, self.src_test_dl, self.src_val_dl = data_generator(self.data_path, src_id,
                                                                              self.dataset_configs,
                                                                              self.hparams)
        self.trg_train_dl, self.trg_test_dl, self.trg_val_dl = data_generator(self.data_path, trg_id,
                                                                              self.dataset_configs,
                                                                              self.hparams)
        # self.few_shot_dl = few_shot_data_generator(self.trg_test_dl)

    def create_save_dir(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def avg_result(self, df, name):
        mean_acc = df.groupby('scenario', as_index=False, sort=False)['accuracy'].mean()
        mean_f1 = df.groupby('scenario', as_index=False, sort=False)['f1'].mean()
        std_acc = df.groupby('run_id', as_index=False, sort=False)['accuracy'].mean()
        std_f1 = df.groupby('run_id', as_index=False, sort=False)['f1'].mean()
        return mean_acc.mean().values, std_acc.std().values, mean_f1.mean().values, std_f1.std().values

    def loss_val(self):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)
        joint_loaders = enumerate(zip(self.src_val_dl, self.trg_val_dl))
        self.algorithm.eval()

        loss_sup = []
        loss_align = []
        with torch.no_grad():
            for step, ((src_x, src_y), (trg_x, trg_y)) in joint_loaders:
                src_x, src_y, trg_x, _ = src_x.float().to(self.device), src_y.long().to(self.device), \
                                         trg_x.float().to(self.device), trg_y.to(self.device)

            if self.da_method == 'RAINCOAT':
                features_src, _ = feature_extractor(src_x)
                features_trg, _ = feature_extractor(trg_x)
            else:
                features_src = feature_extractor(src_x)
                features_trg = feature_extractor(trg_x)
            predictions = classifier(features_src)

            loss_sup.append(F.cross_entropy(predictions, src_y).item())
            loss_align.append(self.sinkdis(features_src, features_trg)[0].item())

        return {'Total_loss': np.mean(loss_sup) + np.mean(loss_align), 'Src_cls_loss': np.mean(loss_sup),
                'Domain_loss': np.mean(loss_align)}

