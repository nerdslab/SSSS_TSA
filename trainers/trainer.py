import torch
import torch.nn.functional as F
import os
from models.loss import SinkhornDistance
import pandas as pd
import numpy as np
import warnings
import wandb
import json
import random
import sklearn.exceptions
from visualizations.plot_funcs import plot_input
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import umap
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
import collections
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
from dataloader.dataloader import data_generator, few_shot_data_generator, generator_percentage_of_data
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from algorithms.utils import fix_randomness, copy_Files, starting_logs, save_checkpoint, _calc_metrics
from algorithms.utils import calc_dev_risk, calculate_risk
from algorithms.algorithms import get_algorithm_class
from algorithms.RAINCOAT import RAINCOAT
from models.models import get_backbone_class
from algorithms.utils import AverageMeter
from sklearn.metrics import f1_score
from torch import nn
from sklearn.metrics import roc_auc_score
torch.backends.cudnn.benchmark = True  
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)        
def balance_samples(x,y):
    no_labels = torch.unique(y)
    no_labels_lens = []
    for  i in list(no_labels):
        no_labels_lens.append(len(torch.where(y==i)[0]))

    max_lbls = max(no_labels_lens)
    x_empty = torch.from_numpy(np.asarray([]))
    y_empty = torch.from_numpy(np.asarray([]))
    for ix,i in enumerate(list(no_labels)):
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
        if float(max_lbls/no_labels_lens[ix])>= 2:

            x_temp = x_temp.repeat(int(2*max_lbls / no_labels_lens[i]),1,1) + torch.normal(mean=0.0, std=0.001, size=(len(x_temp)*int(2*max_lbls / no_labels_lens[i]),1,1)).cuda()
            y_temp = y_temp.repeat(int(2*max_lbls / no_labels_lens[i]))
        idx = random_integers = np.random.randint(0, x_temp.shape[0], size=20)
        x_empty = torch.concat((x_empty, x_temp[idx,:,:]), 0) if len(x_empty) else x_temp[idx,:,:]

        y_empty = torch.concat((y_empty, y_temp[idx]), 0) if len(y_empty) else y_temp[idx]

    return x_empty,y_empty
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



        self.best_acc = 0
        self.best_val_loss = 1e10
        # paths
        self.home_path = os.getcwd()
        self.save_dir = args.save_dir
        self.data_path = os.path.join(args.data_path, self.dataset)
        self.create_save_dir()

        # Specify runs
        self.num_runs = args.num_runs

        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()




        # to fix dimension of features in classifier and discriminator networks.
        self.dataset_configs.final_out_channels = self.dataset_configs.tcn_final_out_channles if args.backbone == "TCN" else self.dataset_configs.final_out_channels

        # Specify number of hparams (Requires da method)
        self.default_hparams = {**self.hparams_class.alg_hparams[self.da_method],
                                **self.hparams_class.train_params}


    def train(self,args=None):

        result_dict = {}


        os.environ["WANDB_SILENT"] = "true"
        os.environ['WANDB_DISABLED'] = "false"
        config_vars = vars(args)
        config_dataset = vars(self.dataset_configs)
        config_run = self.default_hparams
        balanced = config_dataset['balanced']
        config = config_vars
        #config = config.update(config_run)
        #wandb.config = omegaconf.OmegaConf.to_container(
        #    cfg, resolve=True, throw_on_missing=True
        #)
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%YTime_%H:%M:%S")

        print("date and time =", dt_string)


        if balanced:
            self.dataset = self.dataset + '_balanced'



        wandb_name = self.da_method + '_' + self.dataset+'_'+dt_string
        wandb.init(config=config , project="Domain_Adapt", name=wandb_name)
        run_name = f"{self.run_description}"
        self.hparams = self.default_hparams
        # Logging
        self.exp_log_dir = os.path.join(self.save_dir, self.experiment_description, run_name)
        os.makedirs(self.exp_log_dir, exist_ok=True)

        scenarios = self.dataset_configs.scenarios  # return the scenarios given a specific dataset.
        df_a = pd.DataFrame(columns=['scenario','run_id','accuracy','f1','H-score'])

        result_path = f"./results/{wandb_name}.json"



        self.trg_acc_list = []



        f1_list = []
        f1_best_list = []
        f1_best_val_list = []




        f1_list_all = []
        f1_best_list_all = []
        f1_best_val_list_all = []



        f1_list_std = []
        f1_best_list_std = []
        f1_best_val_list_std =[]

        acc_list = []
        acc_best_list = []
        acc_best_val_list = []

        acc_list_std = []
        acc_best_list_std = []
        acc_best_val_list_std = []

        cm_list = []
        scenario_list =[]


        for i in scenarios:
            src_id = i[0]
            trg_id = i[1]
            dict={}
            loggers = {}
            f1_list_run=[]
            f1_list_run_best_val = []
            acc_list_run = []
            f1_list_run_best = []
            acc_list_run_best = []
            acc_list_run_best_val = []
            for run_id in range(0,self.num_runs):
                # specify number of consecutive runs
                # fixing random seed
                #run_id = 2025
                #run_id = 8
                wandb_val_string = f"Val_{str(src_id)}_to_{str(trg_id)}_run_{run_id}_{args.da_method}"
                wandb_trn_string = f"Trn_{str(src_id)}_to_{str(trg_id)}_run_{run_id}_{args.da_method}"

                self.f1_run_score =[]
                torch.cuda.empty_cache()
                fix_randomness(run_id)

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir,
                                                                   src_id, trg_id, run_id)
                self.fpath = os.path.join(self.home_path, self.scenario_log_dir, 'backbone.pth')
                self.cpath = os.path.join(self.home_path, self.scenario_log_dir, 'classifier.pth')
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
                #self.eval()
                loss_total = []
                Dom_loss = []
                Src_cls_loss = []
                wandb.define_metric("epoch")
                wandb.define_metric(f"*", step_metric="epoch")

                for epoch in range(1,self.hparams["num_epochs"] + 1):# self.hparams["num_epochs"] + 1):
                    joint_loaders = enumerate(zip(self.src_train_dl, self.trg_train_dl))
                    len_dataloader = min(len(self.src_train_dl), len(self.trg_train_dl))
                    algorithm.train()

                    for step, ((src_x, src_y), (trg_x, trg_y)) in joint_loaders:
                        src_x, src_y, trg_x,trg_y = src_x.float().to(self.device), src_y.long().to(self.device), \
                                              trg_x.float().to(self.device), trg_y.to(self.device)

                        if config['plot_input_sig']:
                            plot_input(src_x,trg_x,src_y,trg_y,self.dataset,src_id=src_id, trg_id=trg_id)
                            plt.show()
                            print("Plotting raw source and target inputs")

                        if balanced:
                            try:
                                src_x, src_y = balance_samples(src_x, src_y)
                                trg_x, trg_y = balance_samples(trg_x, trg_y)
                            except IndexError:
                                continue

                        len_min = min(len(src_x), len(trg_x))

                        if self.da_method in ["DANN", "CoDATS", "AdaMatch","SepReps","SepRepTranAlignEnd","SepAligThenAttn","SepAligThenAttnSinkFreq","CLUDA"]:
                            losses = algorithm.update(src_x, src_y, trg_x,  step, epoch, len_dataloader)
                        elif self.da_method in {"CDAN","MMDA","Supervised","SinkDiv_Alignment","DDC","AdvSKM","RAINCOAT","Deep_Coral"}:
                            losses = algorithm.update(src_x, src_y, trg_x)
                        else:
                            losses = algorithm.update(src_x, src_y, trg_x,trg_y,step,epoch,len_dataloader)

                        for key, val in losses.items():
                            loss_avg_meters[key].update(val, src_x.size(0))


                    wandb.log({f"{wandb_trn_string}/Train_TotalLoss": losses['Total_loss'],"epoch":epoch})
                    wandb.log({f"{wandb_trn_string}/Train_SrcClfrLoss": losses['Src_cls_loss'],"epoch":epoch})
                    wandb.log({f"{wandb_trn_string}/Train_DomLoss": losses['Domain_loss'],"epoch":epoch})



                    losses_val = self.loss_val()
                    wandb.log({f"{wandb_val_string}/Val_TotalLoss": losses_val['Total_loss'], "epoch": epoch})
                    wandb.log({f"{wandb_val_string}/Val_SrcClfrLoss": losses_val['Src_cls_loss'], "epoch": epoch})
                    wandb.log({f"{wandb_val_string}/Val_DomLoss(Sink)": losses_val['Domain_loss'], "epoch": epoch})




                    #loss_total.append(losses['Total_loss'])
                    #Dom_loss.append(losses[ 'Domain_loss'])
                    #Src_cls_loss.append(losses['Src_cls_loss'])
                    #Src_cls_loss.append(losses)
                    # logging
                    self.logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
                    if self.da_method =='RAINCOAT':
                        (acc, f1, cm),(acc_src, f1_src, cm_src)= self.eval()

                    elif self.da_method =="SepReps":
                        acc, f1,cm = self.algorithm.eval(self.trg_test_dl)
                    else:

                        (acc, f1, cm),(acc_src, f1_src, cm_src) = self.evaluate()
                    if f1>=self.best_acc:
                        self.best_acc = f1
                        print(self.best_acc)
                        torch.save(self.algorithm.feature_extractor.state_dict(), self.fpath)
                        torch.save(self.algorithm.classifier.state_dict(), self.cpath)

                    if losses_val['Total_loss']<= self.best_val_loss:
                        torch.save(self.algorithm.feature_extractor.state_dict(), f"{self.fpath}_best_val")
                        torch.save(self.algorithm.classifier.state_dict(), f"{self.cpath}_best_val")
                    self.f1_run_score.append(f1)

                    wandb.log({f"{wandb_val_string}/Val_Src_f1": f1_src ,"epoch": epoch})
                    wandb.log({f"{wandb_val_string}/Val_Trg_f1": f1, "epoch": epoch})

                    #losses = algorithm.eval_update(self.src_test_dl,self.trg_test_dl)
                    #wandb.log({f"{wandb_val_string}/Train_TotalLoss": losses['Total_loss']})
                    #wandb.log({f"{wandb_val_string}/Train_SrcClfrLoss": losses['Src_cls_loss']})
                    #wandb.log({f"{wandb_val_string}/Train_DomLoss": losses['Domain_loss']})
                    #wandb.log({f"{wandb_val_string}/Val_F1": f1})

                # self.algorithm = algorithm
                # save_checkpoint(self.home_path, self.algorithm, scenarios, self.dataset_configs,
                #                 self.scenario_log_dir, self.hparams)
                # acc, f1,cm = self.evaluate(final=True)
                # log = {'scenario':i,'run_id':run_id,'accuracy':acc,'f1':f1}
                # df_a = df_a.append(log, ignore_index=True)

                #fig1 = self.visualize()
                if self.da_method == "SepReps":
                    acc, f1, cm = self.algorithm.eval(self.trg_test_dl, self.fpath,self.cpath,final=False)
                    acc_best, f1_trg_best, cm_best = self.algorithm.eval(self.trg_test_dl, self.fpath, self.cpath, final=True)
                elif self.da_method =='RAINCOAT':

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
                            (acc, f1, cm),(acc_src, f1_src, cm_src)  = self.eval()
                            if f1 >= self.best_acc:
                                self.best_acc = f1
                                print(self.best_acc)
                                torch.save(self.algorithm.feature_extractor.state_dict(), self.fpath)
                                torch.save(self.algorithm.classifier.state_dict(), self.cpath)

                        (acc, f1, cm),(acc_src, f1_src, cm_src) = self.eval(final_end=True)

                        (acc_best, f1_trg_best, cm_best), (acc_src, f1_src_best, cm_src)= self.eval(final=True)
                        (acc_best_val, f1_best_val, cm_best_val),(acc_src_best_val, f1_src_best_val, cm_src_best_val) = self.eval(best_val=True)
                        #cm = cms[0]
                else:
                    (acc, f1, cm),(acc_src, f1_src, cm_src) = self.evaluate(final_end=True)

                    (acc_best, f1_trg_best, cm_best), (acc_src, f1_src_best, cm_src) = self.evaluate(final=True)
                    (acc_best_val, f1_best_val, cm_best_val), (acc_src_best_val, f1_src_best_val, cm_src_best_val) = self.evaluate(best_val=True)
                f1_list_run.append(f1)
                f1_list_run_best.append(f1_trg_best)
                f1_list_run_best_val.append(f1_best_val)
                cm_list.append(cm)

                acc_list_run.append(acc)
                acc_list_run_best.append(acc_best)
                acc_list_run_best_val.append(acc_best_val)

                #plt.plot(self.f1_run_score)
                #plt.title(f"{self.da_method}..{src_id}_to_{trg_id}")
                vis = 0
                if vis:
                    self.visualize()
                #plt.show()


                #f1_list.append(f1)
                #log = {'scenario':i,'run_id':run_id,'accuracy':acc,'f1':f1}
                #df_a = df_a.append(log, ignore_index=True)
                #print("visualization after correction")

            f1_list.append(np.mean(f1_list_run))
            f1_list_std.append(np.std(f1_list_run))


            acc_list.append(np.mean(acc_list_run))
            acc_list_std.append(np.std(acc_list_run))

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

            scenario_list.append(f"{src_id} to {trg_id}")
        #mean_acc, std_acc, mean_f1, std_f1 = self.avg_result(df_a,'average_align.csv')
            #print("\n\n End of training results (no trgt labels to stop)")







            print(f1_list)
            print(acc_list)

            print("F1 Mean of {len(f1_list}} each case:")
            for i in range(0,len(f1_list)):
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

            #log = {'scenario':mean_acc,'run_id':std_acc,'accuracy':mean_f1,'f1':std_f1}
            #df_a = df_a.append(log, ignore_index=True)
            #print(df_a)
            #path =  os.path.join(self.exp_log_dir, 'average_align.csv')
            #df_a.to_csv(path,sep = ',')
            print(f1_best_list)
            print(acc_best_list)


            result_dict["method"] = self.da_method
            result_dict["dataset"] = self.dataset
            result_dict['best_val_trg_lbl'] ={}
            result_dict["end_train"] ={}
            result_dict["best_val_loss_align"] = {}
            result_dict["scenario_list"] = scenario_list

            result_dict['best_val_trg_lbl']['f1_mean']= np.mean(f1_best_list)
            result_dict['best_val_trg_lbl']['f1_mean_std'] = np.mean(f1_best_list_std)
            result_dict['best_val_trg_lbl']['result List'] = f1_best_list


            result_dict['best_val_trg_lbl']['acc_mean'] = np.mean(acc_best_list)
            result_dict['best_val_trg_lbl']['acc_std'] = np.mean(acc_best_list_std)
            result_dict['best_val_trg_lbl']['result_list_acc'] = acc_best_list

            result_dict['end_train']['f1_mean'] = np.mean(f1_list)
            result_dict['end_train']['f1_mean_std'] = np.mean(f1_list_std)
            result_dict['end_train']['result List'] = f1_list

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

        wandb.log(result_dict)



    def visualize(self):
        #set to if plot or not. Other visualizaitons only for
        plot = 0
        visualize_chnl_algn = 1


        reducer = umap.UMAP()
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)


        #classifier.load_state_dict(torch.load(self.cpath))
        #feature_extractor.load_state_dict(torch.load(self.fpath))
        feature_extractor.eval()
        classifier.eval()

        self.trg_pred_labels = np.array([])

        self.trg_true_labels = np.array([])
        self.trg_all_features = []
        self.src_true_labels = np.array([])
        self.src_all_features = []
        self.src_all_data = torch.from_numpy(np.asarray([]))
        self.trg_all_data = torch.from_numpy(np.asarray([]))
        with torch.no_grad():
            # for data, labels in self.trg_test_dl:
            for data, labels in self.trg_test_dl:
                data = data.float().to(self.device)
                self.trg_all_data = torch.concat((self.trg_all_data, data), dim=0) if len(self.trg_all_data) else data
                labels = labels.view((-1)).long().to(self.device)
                features = feature_extractor(data)
                self.trg_all_features.append(features.cpu().numpy())
                self.trg_true_labels = np.append(self.trg_true_labels, labels.data.cpu().numpy())
        
            
            for data, labels in self.src_test_dl:
                data = data.float().to(self.device)

                self.src_all_data = torch.concat((self.src_all_data, data), dim=0) if len(self.src_all_data) else data
                labels = labels.view((-1)).long().to(self.device)
                #features = feature_extractor(data)[0]
                features = feature_extractor(data)
                self.src_all_features.append(features.cpu().numpy())
                self.src_true_labels = np.append(self.src_true_labels, labels.data.cpu().numpy())
            self.src_all_features = np.vstack(self.src_all_features)
            self.trg_all_features = np.vstack(self.trg_all_features)
            #dr, Cx, Cy = self.LOT(torch.from_numpy(self.src_all_features).to(self.device), torch.from_numpy(self.trg_all_features).to(self.device))
            transformed_data = np.concatenate(
                (self.src_all_features, self.trg_all_features),axis=0)
            #transformed_data = np.concatenate((self.src_all_features,self.trg_all_features,Cx.detach().cpu().numpy(),Cy.detach().cpu().numpy()),axis=0)
            src_trgt_lbl = np.concatenate((0*np.ones(len(self.src_all_features)),1*np.ones(len(self.trg_all_features))),axis=0)
            #src_trgt_lbl = np.concatenate((0 * np.ones(len(self.src_all_features)),
            #                               1 * np.ones(len(self.trg_all_features)), 2 * np.ones(len(Cx)),
            #                               3 * np.ones(len(Cy)),), axis=0)



            if plot ==1 :
                transformed_data = reducer.fit_transform(transformed_data.astype(dtype='float32'))

                labels = np.concatenate((self.src_true_labels,self.trg_true_labels),axis=0)

                fig1, axs2 = plt.subplots(1, 3, figsize=(15, 5))

                axs2[0].scatter(transformed_data[ :len(self.src_all_features),0], transformed_data[:len(self.src_all_features),1], c='red', alpha=0.6)
                axs2[0].scatter(transformed_data[ len(self.src_all_features):,0], transformed_data[len(self.src_all_features):,1], c='blue', alpha=0.6)
                clr_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown','tab:pink']
                label_list = ['Upstairs', 'Jogging', 'Sitting', 'standing', 'Downstairs','Walking']
                for j in range(0, self.algorithm.configs.num_classes):
                    j_idx = np.where((labels== j) & (src_trgt_lbl == 0)  )[0]
                    axs2[1].scatter(transformed_data[j_idx, 0], transformed_data[j_idx, 1], c=clr_list[j], label=label_list[j],
                                    alpha=0.6)
                for j in range(0, self.algorithm.configs.num_classes):
                    j_idx = np.where((labels== j)& (src_trgt_lbl == 1)  )[0]
                    axs2[2].scatter(transformed_data[j_idx, 0], transformed_data[j_idx, 1], c=clr_list[j], label=label_list[j],
                                    alpha=0.6)

                x_min =  min(np.concatenate((transformed_data[:, 0], transformed_data[:, 0])))
                x_max = max(np.concatenate((transformed_data[:, 0], transformed_data[:, 0])))

                y_min = min(np.concatenate((transformed_data[:, 1], transformed_data[:, 1])))
                y_max = max(np.concatenate((transformed_data[:, 1], transformed_data[:, 1])))
                axs2[1].set_title("Source")
                axs2[2].set_title("Target")
                axs2[0].set_title("Combined Domains")

                for k in range(0,self.algorithm.configs.input_channels):
                    axs2[k].set_ylim((y_min-1,y_max+1))
                    axs2[k].set_xlim((x_min-1,x_max+1))
                axs2[2].legend()
                plt.tight_layout()

            if visualize_chnl_algn and self.da_method == 'SepAligThenAttnSink' or  self.da_method == 'SepAligThenAttnSinkFreq' :
                'visualoize channel specific alignment, and attention scores. Made for our custom method. Fails for '
                dist_mtx_src = np.zeros((self.algorithm.configs.num_classes,self.algorithm.configs.num_classes))
                dist_mtx_trg = np.zeros((self.algorithm.configs.num_classes,self.algorithm.configs.num_classes))
                dist_mtx_src_trg =np.zeros((self.algorithm.configs.num_classes,self.algorithm.configs.num_classes))

                '''
                for j in range(0,self.algorithm.configs.input_channels):
                    src_j = torch.where(torch.from_numpy(self.src_true_labels) == j)[0]
                    trg_j = torch.where(torch.from_numpy(self.trg_true_labels) == j)[0]
                    for k in range(0,self.algorithm.configs.input_channels):
                        src_k = torch.where(torch.from_numpy(self.src_true_labels) == k)[0]

                        trg_k = torch.where(torch.from_numpy(self.trg_true_labels)==k)[0]
                        min_l_j_k_src  = min(len(src_j),len(src_k))
                        min_l_j_k_src_trg = min(len(src_j), len(trg_k))
                        min_l_j_k_trg = min(len(trg_j), len(trg_k))
                        numb_j_k_src = int(min_l_j_k_src /2)
                        numb_j_k_src_trg = int(min_l_j_k_src_trg  / 2)
                        numb_j_k_trg = int(min_l_j_k_trg/2)
                        dist_s_s = torch.mean(torch.cdist(torch.from_numpy(self.src_all_features[src_j][0:numb_j_k_src]),torch.from_numpy(self.src_all_features[src_k][numb_j_k_src:])))
                        dist_t_t = torch.mean(torch.cdist(torch.from_numpy(self.trg_all_features[trg_j][0: numb_j_k_trg]),torch.from_numpy(self.trg_all_features[trg_k][ numb_j_k_trg:])))
                        dist_s_t = torch.mean(torch.cdist(torch.from_numpy(self.src_all_features[src_j][0:numb_j_k_src_trg]),
                                                          torch.from_numpy(self.trg_all_features[trg_k][0:numb_j_k_src_trg])))
                        dist_mtx_src[j,k] = dist_s_s
                        dist_mtx_trg[j,k] = dist_t_t
                        dist_mtx_src_trg[j,k] = dist_s_t
                '''
                preds_trg = classifier(torch.from_numpy(self.trg_all_features).to(0)).argmax(dim=1).detach().cpu()
                preds_src = classifier(torch.from_numpy(self.src_all_features).to(0)).argmax(dim=1).detach().cpu()
                print("Source CM")
                cm_src = confusion_matrix(self.src_true_labels, preds_src)
                print(cm_src)
                print("Target CM")
                cm_trg = confusion_matrix(self.trg_true_labels,preds_trg)
                print(cm_trg)
                f1_trg= f1_score(self.trg_true_labels,preds_trg,average='macro')
                f1_src  = f1_score(self.src_true_labels, preds_src, average='macro')
                #torch.sum(self.algorithm.domain_classifier(torch.from_numpy(self.trg_all_features[trg_j]).to(0)).argmax(
                #    dim=1) == 0)
                #self.algorithm.feature_extractor.fetch_att_weights(self.trg_all_data[trg_j, :, :])
                prob_trg, pred_trg = self.algorithm.get_ind_scores(self.trg_all_data)
                prob_src, pred_src = self.algorithm.get_ind_scores(self.src_all_data)
                #plt.show()

                f1_list_chnl_src = []
                f1_list_chnl_trg = []


                if self.da_method == 'SepAligThenAttnSink':
                    no_channels = self.algorithm.configs.input_channels
                elif self.da_method == 'SepAligThenAttnSinkFreq':
                    no_channels = self.algorithm.configs.input_channels *2
                for k in range(0,no_channels):
                    print(f" CM Source channel {k} ")
                    cm_src = confusion_matrix(self.src_true_labels,pred_src[k])

                    print(cm_src)

                    f1_list_chnl_src.append(f1_score(self.src_true_labels, pred_src[k], average='macro'))
                    print(f" CM Target channel {k} ")
                    cm_trg = confusion_matrix(self.trg_true_labels, pred_trg[k])
                    f1_list_chnl_trg.append(f1_score(self.trg_true_labels, pred_trg[k], average='macro'))
                    print(cm_trg)

                src_attn = np.asarray([])
                trg_attn = np.asarray([])

                for j in range(0,self.algorithm.configs.num_classes):
                    trg_j = torch.where(torch.from_numpy(self.trg_true_labels) == j)[0]
                    src_j = torch.where(torch.from_numpy(self.src_true_labels) == j)[0]
                    fig, axes = plt.subplots(nrows=no_channels, ncols=3)
                    #plt.figure(figsize=(24, 24))
                    #f1_score_trg = f1_score(self.trg_true_labels[trg_j], preds_trg[trg_j], labels=[j],
                    #                       average='binary')
                    #f1_score_src = f1_score(self.src_true_labels[src_j], preds_src[src_j], labels=[j],
                     #                       average='binary')
                    plt.suptitle(f"Src(lft)  and Trg(right)  true class {j} across {self.algorithm.configs.input_channels} chnls")
                    for k in range(0,no_channels):
                        'set j to the class being interested in'



                        axes[k,0].matshow(prob_src[k][src_j, :][0:10].detach().cpu())
                        axes[k,0].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
                        axes[k,0].set_xticklabels([])
                        axes[k,0].set_yticklabels([])
                        #axes[k,0].set_title(f"Src {j} Cnl {k}")
                        axes[k,1].matshow(prob_trg[k][trg_j,:][0:10].detach().cpu())
                        #axes[k, 1].set_title(f"Trg {j} Cnl {k}")
                        axes[k, 1].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
                        axes[k, 1].set_xticklabels([])
                        axes[k, 1].set_yticklabels([])
                        axes[k, 2].axis('off')



                        'get attention weights for the kth channel for the jth class'
                    attn_src = self.algorithm.feature_extractor.fetch_att_weights(self.src_all_data[src_j, :, :])[0].detach().cpu()
                    attn_trg = self.algorithm.feature_extractor.fetch_att_weights(self.trg_all_data[trg_j, :, :])[0].detach().cpu()
                      # Adjust spacing between subplots

                    src_attn = np.concatenate((src_attn,torch.diag(attn_src).cpu().reshape(1,-1)),axis=0) if len(src_attn) else torch.diag(attn_src).cpu().reshape(1,-1)
                    trg_attn = np.concatenate((trg_attn, torch.diag(attn_trg).cpu().reshape(1, -1)), axis=0) if len(
                        trg_attn) else torch.diag(attn_trg).cpu().reshape(1, -1)

                    #plt.tight_layout()
                    axes[0, 2].matshow(attn_src)
                    axes[0,2].set_title("Src Atn")
                    axes[k, 2].matshow(attn_trg)
                    axes[k, 2].set_title("T Atn")
                    #plt.show()

            #np.savez("figures/HHAR_0_2/data_for_plots/attn.npz", src=src_attn, trg=trg_attn)
            print("Here. Stop to analyze channel alignment")
    
    def evaluate(self, final_end = False,final=False,best_val=False):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)
        # self.algorithm.ema.apply_shadow()
        # # evaluate
        # self.algorithm.ema.restore()
        dloader = self.trg_val_dl

        if final_end == True:
            dloader = self.trg_test_dl

        if final == True:
            feature_extractor.load_state_dict(torch.load(self.fpath))
            classifier.load_state_dict(torch.load(self.cpath))
            dloader = self.trg_test_dl


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
        f1_trg = f1_score( self.trg_true_labels,self.trg_pred_labels, pos_label=None, average="macro")
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
        f1_src = f1_score( self.src_true_labels,self.src_pred_labels, pos_label=None, average="macro")
        cm_src = confusion_matrix(self.src_true_labels, self.src_pred_labels, normalize=None)
        #src_roc = roc_auc_score(self.src_true_labels, self.src_pred_labels)
        #trg_roc = roc_auc_score(self.trg_true_labels, self.trg_pred_labels)
        #return (trg_roc*100, f1_src,cm_trg),(src_roc *100, f1_trg,cm_src)
        return (accuracy_trg * 100, f1_trg, cm_trg), (accuracy_src * 100, f1_src, cm_src)
    def eval(self, final_end = False,final=False,best_val=False):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)
        dloader = self.trg_val_dl

        if final_end ==True:
            dloader = self.trg_test_dl
        if final == True:
            feature_extractor.load_state_dict(torch.load(self.fpath))
            classifier.load_state_dict(torch.load(self.cpath))
            dloader = self.trg_test_dl
        if best_val == True:
            feature_extractor.load_state_dict(torch.load(f"{self.fpath}_best_val"))
            classifier.load_state_dict(torch.load(f"{self.cpath}_best_val"))
            dloader = self.trg_test_dl

        feature_extractor.eval()
        classifier.eval()

        total_loss_ = []

        self.trg_pred_labels = np.array([])
        self.trg_true_labels = np.array([])

        with torch.no_grad():
            for data, labels in dloader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass

                features ,_ = feature_extractor(data)

                predictions = classifier(features)

                # compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss_.append(loss.item())
                pred = predictions.detach().argmax(dim=1)  # get the index of the max log-probability

                self.trg_pred_labels = np.append(self.trg_pred_labels, pred.cpu().numpy())
                self.trg_true_labels = np.append(self.trg_true_labels, labels.data.cpu().numpy())
        accuracy = accuracy_score(self.trg_true_labels, self.trg_pred_labels)
        f1 = f1_score(self.trg_true_labels,self.trg_pred_labels, pos_label=None, average="macro")
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

        return (accuracy*100, f1,confusion_matrix_trgt),(accuracy_src,f1_src,confusion_matrix_src)

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.dataset)
        return dataset_class(), hparams_class()

    def load_data(self, src_id, trg_id):
        self.src_train_dl, self.src_test_dl,self.src_val_dl = data_generator(self.data_path, src_id, self.dataset_configs,
                                                             self.hparams)
        self.trg_train_dl, self.trg_test_dl,self.trg_val_dl = data_generator(self.data_path, trg_id, self.dataset_configs,
                                                             self.hparams)
        #self.few_shot_dl = few_shot_data_generator(self.trg_test_dl)

    def create_save_dir(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def avg_result(self, df, name):
        mean_acc = df.groupby('scenario', as_index=False, sort=False)['accuracy'].mean()
        mean_f1 = df.groupby('scenario', as_index=False, sort=False)['f1'].mean()
        std_acc = df.groupby('run_id', as_index=False, sort=False)['accuracy'].mean()
        std_f1 =  df.groupby('run_id', as_index=False, sort=False)['f1'].mean()
        return mean_acc.mean().values, std_acc.std().values, mean_f1.mean().values, std_f1.std().values

    
    def loss_val(self):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)
        joint_loaders = enumerate(zip(self.src_val_dl, self.trg_val_dl))
        self.algorithm.eval()

        loss_sup = []
        loss_align =[]
        with torch.no_grad():
            for step, ((src_x, src_y), (trg_x, trg_y)) in joint_loaders:
                src_x, src_y, trg_x, _ = src_x.float().to(self.device), src_y.long().to(self.device), \
                                             trg_x.float().to(self.device), trg_y.to(self.device)

            if self.da_method == 'RAINCOAT':
                features_src,_ = feature_extractor(src_x)
                features_trg,_ = feature_extractor(trg_x)
            else:
                features_src = feature_extractor(src_x)
                features_trg = feature_extractor(trg_x)
            predictions = classifier(features_src)

            loss_sup.append(F.cross_entropy(predictions, src_y).item())
            loss_align.append(self.sinkdis(features_src,features_trg)[0].item())

        return {'Total_loss': np.mean(loss_sup)+np.mean(loss_align), 'Src_cls_loss': np.mean(loss_sup), 'Domain_loss': np.mean(loss_align)}

