import os
import argparse
import warnings
import sklearn.exceptions
import pickle
from sklearn.manifold import TSNE
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
from trainers.trainer import cross_domain_trainer


parser = argparse.ArgumentParser()


# ========  Experiments Name ================
parser.add_argument('--save_dir',               default='experiments_logs',         type=str, help='Directory containing all experiments')
parser.add_argument('--experiment_description', default='WISDM-SSSS_TSA',               type=str, help='Name of your experiment (EEG, HAR, HHAR_SA, WISDM')
parser.add_argument('--run_description',        default='WISDM-SSSS_TSA',                     type=str, help='name of your runs')

# ========= Select the DA methods ============
parser.add_argument('--da_method',              default='SSSS_TSA',               type=str, help='TestTime_adapt,CoTMix,SepAligThenAttnSink,SepAligSameEncThenAttnSink,SepAligThenNoAttnSink,SepAligThenNoAttnMMD,SepAligThenSum,SepAligThenAttnSinkFreq,SinkDiv_Alignment,SepAligThenAttn,SepRepTranAlignEnd,CoDATS, SepReps,Deep_Coral, RAINCOAT, MMDA, VADA, DIRT, CDAN, AdaMatch, HoMM, CoDATS,CUSTOM')

# ========= Select the DATASET ==============
parser.add_argument('--data_path',              default=r'./datasets/',                  type=str, help='Path containing dataset')
parser.add_argument('--dataset',                default='WISDM',                      type=str, help=' WISDM - HAR_UCI - HHAR_SA. ')

# ========= Select the BACKBONE ==============
parser.add_argument('--backbone',               default='CNN',                      type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')
parser.add_argument('--load',               default=0,                      type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')
# ========= Experiment settings ===============
parser.add_argument('--num_runs',               default=5,                          type=int, help='Number of consecutive run with different seeds')
parser.add_argument('--sink_eps',               default=1e-3,                          type=float, help='Sinkhorn divergence epsilon/gamma')

parser.add_argument('--tau_temp',               default=0.1,                         type=float, help='tau for attention')
parser.add_argument('--device',                 default='cuda',                   type=str, help='cpu or cuda')
parser.add_argument('--plot_input_sig',                 default=False,                   type=bool, help='Plot input values')
args = parser.parse_args()

if __name__ == "__main__":
    trainer = cross_domain_trainer(args)


    trainer.train(args)
