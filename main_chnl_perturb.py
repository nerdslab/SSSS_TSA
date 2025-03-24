import os
import argparse
import warnings
import sklearn.exceptions
import pickle
from sklearn.manifold import TSNE
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
from trainers.trainier_chnl_perturb import cross_domain_trainer


parser = argparse.ArgumentParser()

##Number of channel affected variable is set in configs/data_model_configs_channel_perturb.py (look for self.no_channel_affect variable in HAR_UCI
# ========  Experiments Name ================
parser.add_argument('--save_dir',               default='experiments_logs',         type=str, help='Directory containing all experiments')
parser.add_argument('--experiment_description', default='HAR_UCI-Gauss_Perturb_SSSS_TSA',               type=str, help='Name of your experiment (EEG, HAR, HHAR_SA, WISDM')
parser.add_argument('--run_description',        default='HAR_UCI-Gauss_Perturb_SSSS_TSA',                     type=str, help='name of your runs')
# ========= Select the DA methods ============
parser.add_argument('--da_method',              default='SSSS_TSA',               type=str, help='SepAligThenNoAttnSink,SepAligThenSum,SepAligThenAttnSinkFreq,SinkDiv_Alignment,SepAligThenAttn,SepRepTranAlignEnd,CoDATS, SepReps,Deep_Coral, RAINCOAT, MMDA, VADA, DIRT, CDAN, AdaMatch, HoMM, CoDATS,CUSTOM')
# ========= Select the DATASET ==============
parser.add_argument('--data_path',              default=r'./datasets/',                  type=str, help='Path containing dataset')
parser.add_argument('--dataset',                default='HAR_UCI',                      type=str, help='WISDM - HAR_UCI - HHAR_SA. ')

# ========= Select the BACKBONE ==============
parser.add_argument('--backbone',               default='CNN',                      type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')

# ========= Experiment settings ===============
parser.add_argument('--num_runs',               default=5,                          type=int, help='Number of consecutive run with different seeds')
parser.add_argument('--device',                 default='cuda',                   type=str, help='cpu or cuda')
parser.add_argument('--plot_input_sig',                 default=False,                   type=bool, help='Plot input values')
parser.add_argument('--add_gauss',                 default=True,                   type=bool, help='add gauss to channels')
parser.add_argument('--drop_chnl',                 default=False,                   type=bool, help='Channels to be dropped')
parser.add_argument('--satur_chnl',                 default=False,                   type=bool, help='Channels to be dropped')
args = parser.parse_args()

if __name__ == "__main__":
    trainer = cross_domain_trainer(args)


    trainer.train(args)
