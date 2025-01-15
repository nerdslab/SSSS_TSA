import os
import argparse
import warnings
import sklearn.exceptions
import pickle
from sklearn.manifold import TSNE
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
from trainers.trainer import cross_domain_trainer
# from trainers.trainer2 import cross_domain_trainer_ours

parser = argparse.ArgumentParser()


# ========  Experiments Name ================
parser.add_argument('--save_dir',               default='experiments_logs',         type=str, help='Directory containing all experiments')
parser.add_argument('--experiment_description', default='SSSS_TSA-exp',               type=str, help='Name of your experiment (EEG, HAR, HHAR_SA, WISDM')
parser.add_argument('--run_description',        default='SSSS_TSA-exp',                     type=str, help='name of your runs')

# ========= Select the DA methods ============
parser.add_argument('--da_method',              default='SSSS_TSA',               type=str, help='SSSS_TSA,CoDATS, SepReps,Deep_Coral, RAINCOAT, MMDA, VADA, DIRT, CDAN, AdaMatch, HoMM, CoDATS,CUSTOM')

# ========= Select the DATASET ==============
parser.add_argument('--data_path',              default=r'./datasets/',                  type=str, help='Path containing dataset')
parser.add_argument('--dataset',                default='HAR_UCI',                      type=str, help='pkecg,Dataset of choice: ( WISDM - pkecg - HAR_UCI - HHAR_SA, Mean_shift_simulations)')

# ========= Select the BACKBONE ==============
parser.add_argument('--backbone',               default='CNN',                      type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')

# ========= Experiment settings ===============
parser.add_argument('--num_runs',               default=5,                          type=int, help='Number of consecutive run with different seeds')
parser.add_argument('--device',                 default='cuda',                   type=str, help='cpu or cuda')
parser.add_argument('--plot_input_sig',                 default=False,                   type=bool, help='Plot input values')
args = parser.parse_args()

if __name__ == "__main__":
    trainer = cross_domain_trainer(args)


    trainer.train(args)
    # trainer.visualize()
    # dic = {'1':trainer.src_all_features,'2':trainer.src_true_labels,'3':trainer.trg_all_features,'4':trainer.trg_true_labels,'acc': trainer.trg_acc_list}
    # with open('saved_dictionary2.pickle', 'wb') as handle:
    #     pickle.dump(dic, handle)
    