
## The cuurent hyper-parameters values are not necessarily the best ones for a specific risk.
def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class HAR_UCI():
    def __init__(self):
        super(HAR_UCI, self).__init__()
        self.train_params = {
                #'num_epochs':2,
                #'num_epochs': 150,
            'num_epochs': 350,
                #'num_epochs': 1,
                'batch_size': 64,

                'weight_decay': 1e-4,

        }
        self.alg_hparams = {
            'RAINCOAT':       {'learning_rate':5e-4,     'src_cls_loss_wt': 0.5,    'domain_loss_wt': 0.5},
            'TestTime_adapt': {'learning_rate': 5e-4, 'src_cls_loss_wt': 0.5, 'domain_loss_wt': 0.5},
            'SASA': {'learning_rate': 5e-4, 'src_cls_loss_wt': 0.5, 'domain_loss_wt': 0.5},
            #'DANN':         {'learning_rate': 1e-2, 'src_cls_loss_wt': 9.74, 'domain_loss_wt': 5.43},
            'DANN': {'learning_rate': 1e-3, 'src_cls_loss_wt': 9.74, 'domain_loss_wt': 5.43},
            'SepAligThenAttn': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligThenNoAttnMMD': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligThenAttnMMD': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'CLUDA':{'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'JDOT': {'learning_rate': 0.001, 'src_cls_loss_wt': 0.5, 'sinkdiv_loss_wt': 0.5},
            'SSSS_TSA':{'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligThenAttnSinkFreq': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligSameEncThenAttnSink': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligThenNoAttnSink': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligThenSum': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligThenAttnKLDiv': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'NoSepAligThenAttnSink': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'CoTMix': {
                'learning_rate': 0.001,
                'mix_ratio': 0.9,
                'temporal_shift': 14,
                'src_cls_weight': 0.78,
                'src_supCon_weight': 0.1,
                'trg_cont_weight': 0.1,
                'trg_entropy_weight': 0.05
            },
            'Supervised': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.0},
            'Supervised_trg': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.0},
            'SinkDiv_Alignment': {'learning_rate': 5e-4, 'src_cls_loss_wt': 0.5,'sinkdiv_loss_wt':0.5},
            'Deep_Coral':   {'learning_rate': 5e-3, 'src_cls_loss_wt': 8.67, 'coral_wt': 0.44},
            'DDC':          {'learning_rate': 5e-3, 'src_cls_loss_wt': 6.24, 'domain_loss_wt': 6.36},
            'HoMM':         {'learning_rate': 1e-3, 'src_cls_loss_wt': 2.15, 'domain_loss_wt': 9.13},
            'CoDATS':       {'learning_rate': 1e-3, 'src_cls_loss_wt': 6.21, 'domain_loss_wt': 1.72},
            'DSAN':         {'learning_rate': 5e-4, 'src_cls_loss_wt': 1.76, 'domain_loss_wt': 1.59},
            'AdvSKM':       {'learning_rate': 5e-3, 'src_cls_loss_wt': 3.05, 'domain_loss_wt': 2.876},
            'MMDA':         {'learning_rate': 1e-3, 'src_cls_loss_wt': 6.13, 'mmd_wt': 2.37, 'coral_wt': 8.63, 'cond_ent_wt': 7.16},
            'CDAN':         {'learning_rate': 1e-2, 'src_cls_loss_wt': 5.19, 'domain_loss_wt': 2.91, 'cond_ent_wt': 1.73},
            "AdaMatch":      {'learning_rate': 3e-3,     'tau': 0.9, 'max_segments': 5, 'jitter_scale_ratio': 0.01, 'jitter_ratio': 0.2},
            'DIRT':         {'learning_rate': 5e-4, 'src_cls_loss_wt': 7.00, 'domain_loss_wt': 4.51, 'cond_ent_wt': 0.79, 'vat_loss_wt': 9.31}
        }


class Mean_shift_simulations():
    def __init__(self):
        super(Mean_shift_simulations, self).__init__()
        self.train_params = {
                'num_epochs':   100,
                'batch_size': 1024,
                'weight_decay': 1e-4,

        }
        self.alg_hparams = {
            'DANN': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepReps': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepRepTranAlignEnd': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'Supervised': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.0},
            'SASA': {'learning_rate': 5e-4, 'src_cls_loss_wt': 0.5, 'domain_loss_wt': 0.5},
            'CLUDA': {'learning_rate': 1e-4, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligThenAttn': {'learning_rate': 1e-4, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SSSS_TSA': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligThenSum': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligThenAttnSinkFreq': {'learning_rate': 1e-4, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligThenNoAttnSink': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},

            'NoSepAligThenAttnSink': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            # 'DANN': {'learning_rate': 1e-2, 'src_cls_loss_wt': 9.74, 'domain_loss_wt': 0.0001},
            'Supervised': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.0},
            'SinkDiv_Alignment': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.0, 'sinkdiv_loss_wt': 1.0},
            'Deep_Coral': {'learning_rate': 0.005, 'src_cls_loss_wt': 8.876, 'coral_wt': 5.56},
            'DDC': {'learning_rate': 1e-3, 'src_cls_loss_wt': 7.01, 'domain_loss_wt': 7.595},
            'HoMM': {'learning_rate': 1e-3, 'src_cls_loss_wt': 0.1913, 'domain_loss_wt': 4.239},
            'CoDATS': {'learning_rate': 1e-3, 'src_cls_loss_wt': 7.187, 'domain_loss_wt': 6.439},
            'DSAN': {'learning_rate': 1e-3, 'src_cls_loss_wt': 0.1, 'domain_loss_wt': 0.1},
            'AdvSKM': {'learning_rate': 3e-4, 'src_cls_loss_wt': 3.05, 'domain_loss_wt': 2.876},
            'MMDA': {'learning_rate': 1e-3, 'src_cls_loss_wt': 0.1, 'mmd_wt': 0.1, 'coral_wt': 0.1,
                     'cond_ent_wt': 0.4753},
            "AdaMatch": {'learning_rate': 3e-3, 'tau': 0.9, 'max_segments': 5, 'jitter_scale_ratio': 0.01,
                         'jitter_ratio': 0.2},
            'CDAN': {'learning_rate': 1e-3, 'src_cls_loss_wt': 9.54, 'domain_loss_wt': 3.283, 'cond_ent_wt': 0.1},
            'DIRT': {'learning_rate': 1e-3, 'src_cls_loss_wt': 0.5, 'domain_loss_wt': 0.5, 'cond_ent_wt': 0.1,
                     'vat_loss_wt': 0.1},
            'RAINCOAT': {'learning_rate': 1e-3, 'src_cls_loss_wt': 0.5, 'domain_loss_wt': 0.5},
            "DANCE": {'learning_rate': 1e-3, 'momentum': 0.5, 'eta': 0.05, "thr": 1.71, "margin": 0.5, 'temp': 0.05}
        }

class Mean_shift_simulations_gauss_noise():
    def __init__(self):
        super(Mean_shift_simulations_gauss_noise, self).__init__()
        self.train_params = {
                'num_epochs':   100,
                'batch_size': 1024,
                'weight_decay': 1e-4,

        }
        self.alg_hparams = {
            'DANN': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepReps': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SASA': {'learning_rate': 5e-4, 'src_cls_loss_wt': 0.5, 'domain_loss_wt': 0.5},
            'SepRepTranAlignEnd': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'Supervised': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.0},
            'CLUDA': {'learning_rate': 1e-4, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligThenAttn': {'learning_rate': 1e-4, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SSSS_TSA': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligThenSum': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligThenAttnSinkFreq': {'learning_rate': 1e-4, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligThenNoAttnSink': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},

            'NoSepAligThenAttnSink': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            # 'DANN': {'learning_rate': 1e-2, 'src_cls_loss_wt': 9.74, 'domain_loss_wt': 0.0001},
            'Supervised': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.0},
            'SinkDiv_Alignment': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.0, 'sinkdiv_loss_wt': 1.0},
            'Deep_Coral': {'learning_rate': 0.005, 'src_cls_loss_wt': 8.876, 'coral_wt': 5.56},
            'DDC': {'learning_rate': 1e-3, 'src_cls_loss_wt': 7.01, 'domain_loss_wt': 7.595},
            'HoMM': {'learning_rate': 1e-3, 'src_cls_loss_wt': 0.1913, 'domain_loss_wt': 4.239},
            'CoDATS': {'learning_rate': 1e-3, 'src_cls_loss_wt': 7.187, 'domain_loss_wt': 6.439},
            'DSAN': {'learning_rate': 1e-3, 'src_cls_loss_wt': 0.1, 'domain_loss_wt': 0.1},
            'AdvSKM': {'learning_rate': 3e-4, 'src_cls_loss_wt': 3.05, 'domain_loss_wt': 2.876},
            'MMDA': {'learning_rate': 1e-3, 'src_cls_loss_wt': 0.1, 'mmd_wt': 0.1, 'coral_wt': 0.1,
                     'cond_ent_wt': 0.4753},
            "AdaMatch": {'learning_rate': 3e-3, 'tau': 0.9, 'max_segments': 5, 'jitter_scale_ratio': 0.01,
                         'jitter_ratio': 0.2},
            'CDAN': {'learning_rate': 1e-3, 'src_cls_loss_wt': 9.54, 'domain_loss_wt': 3.283, 'cond_ent_wt': 0.1},
            'DIRT': {'learning_rate': 1e-3, 'src_cls_loss_wt': 0.5, 'domain_loss_wt': 0.5, 'cond_ent_wt': 0.1,
                     'vat_loss_wt': 0.1},
            'RAINCOAT': {'learning_rate': 1e-3, 'src_cls_loss_wt': 0.5, 'domain_loss_wt': 0.5},
            "DANCE": {'learning_rate': 1e-3, 'momentum': 0.5, 'eta': 0.05, "thr": 1.71, "margin": 0.5, 'temp': 0.05}
        }


class WISDM():
    def __init__(self):
        super(WISDM, self).__init__()
        self.train_params = {

               'num_epochs': 450,
            #'num_epochs': 1,
                'batch_size':128,
                #'batch_size': 64,
                'weight_decay': 1e-4,

        }
        self.alg_hparams = {
            'DANN':         {'learning_rate': 1e-3,     'src_cls_loss_wt': 1.613,   'domain_loss_wt': 1.857},
            'SASA': {'learning_rate': 5e-4, 'src_cls_loss_wt': 0.5, 'domain_loss_wt': 0.5},
            'SepReps': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepRepTranAlignEnd': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SSSS_TSA': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligThenAttnKLDiv': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'TestTime_adapt': {'learning_rate': 5e-4, 'src_cls_loss_wt': 0.5, 'domain_loss_wt': 0.5},
            'SepAligThenAttnSink':{'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligSameEncThenAttnSink': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligThenNoAttnMMD': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'JDOT': {'learning_rate': 0.001, 'src_cls_loss_wt': 0.5, 'sinkdiv_loss_wt': 0.5},
            'SepAligThenNoAttnSink': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligThenAttnMMD': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligThenSum': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'Supervised_trg': {'learning_rate': 1e-4, 'src_cls_loss_wt': 1.0},
            'NoSepAligThenAttnSink': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligThenAttnSinkFreq': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            #'DANN': {'learning_rate': 1e-2, 'src_cls_loss_wt': 9.74, 'domain_loss_wt': 0.0001},
            'CLUDA': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'Supervised': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.0},
            'CoTMix': {
                'learning_rate': 0.001,
                'mix_ratio': 0.72,
                'temporal_shift': 14,
                'src_cls_weight': 0.98,
                'src_supCon_weight': 0.1,
                'trg_cont_weight': 0.1,
                'trg_entropy_weight': 0.05,
            },
            'SinkDiv_Alignment': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.0,'sinkdiv_loss_wt':1.0},
            'Deep_Coral':   {'learning_rate': 0.005,    'src_cls_loss_wt': 8.876,   'coral_wt': 5.56},
            'DDC':          {'learning_rate': 1e-3,     'src_cls_loss_wt': 7.01,    'domain_loss_wt': 7.595},
            'HoMM':         {'learning_rate': 1e-3,     'src_cls_loss_wt': 0.1913,  'domain_loss_wt': 4.239},
            'CoDATS':       {'learning_rate': 1e-3,     'src_cls_loss_wt': 7.187,   'domain_loss_wt': 6.439},
            'DSAN':         {'learning_rate': 1e-3,     'src_cls_loss_wt': 0.1,     'domain_loss_wt': 0.1},
            'AdvSKM':       {'learning_rate': 3e-4,     'src_cls_loss_wt': 3.05,    'domain_loss_wt': 2.876},
            'MMDA':         {'learning_rate': 1e-3,     'src_cls_loss_wt': 0.1,     'mmd_wt': 0.1, 'coral_wt': 0.1, 'cond_ent_wt': 0.4753},
            "AdaMatch":      {'learning_rate': 3e-3,     'tau': 0.9, 'max_segments': 5, 'jitter_scale_ratio': 0.01, 'jitter_ratio': 0.2},
            'CDAN':         {'learning_rate': 1e-3,     'src_cls_loss_wt': 9.54,    'domain_loss_wt': 3.283,        'cond_ent_wt': 0.1},
            'DIRT':         {'learning_rate': 1e-3,     'src_cls_loss_wt': 0.5,     'domain_loss_wt': 0.5,          'cond_ent_wt': 0.1, 'vat_loss_wt': 0.1},
            'RAINCOAT':       {'learning_rate':1e-3,     'src_cls_loss_wt': 0.5,    'domain_loss_wt': 0.5},
            "DANCE":        {'learning_rate': 1e-3,    'momentum': 0.5, 'eta': 0.05, "thr": 1.71,   "margin": 0.5, 'temp': 0.05}
        }




class HHAR_SA():
    def __init__(self):
        super(HHAR_SA, self).__init__()
        self.train_params = {
                'num_epochs': 400,
                #'num_epochs': 1,
                'batch_size': 64,
                'weight_decay': 1e-4,
        }
        self.alg_hparams = {
            'RAINCOAT':       {'learning_rate':0.001,     'src_cls_loss_wt': 0.5,    'domain_loss_wt': 0.5},
            'SASA': {'learning_rate': 5e-4, 'src_cls_loss_wt': 0.5, 'domain_loss_wt': 0.5},
            'SinkDiv_Alignment': {'learning_rate': 0.001, 'src_cls_loss_wt': 0.5, 'sinkdiv_loss_wt': 0.5},
            'JDOT': {'learning_rate': 0.001, 'src_cls_loss_wt': 0.5, 'sinkdiv_loss_wt': 0.5},
            'TestTime_adapt': {'learning_rate': 5e-4, 'src_cls_loss_wt': 0.5, 'domain_loss_wt': 0.5},
            'Supervised': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.0},
            'SSSS_TSA': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligThenSum': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligSameEncThenAttnSink': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligThenAttnMMD': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligThenAttnKLDiv': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligThenNoAttnMMD':{'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligThenAttnSink': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligThenAttnSinkFreq': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligThenNoAttnSink': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'Supervised_trg': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.0},
            'NoSepAligThenAttnSink': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'DANN':         {'learning_rate':  1e-3,   'src_cls_loss_wt': 0.9603,  'domain_loss_wt':0.9238},
            'Deep_Coral':   {'learning_rate': 0.0005,   'src_cls_loss_wt': 0.05931, 'coral_wt': 8.452},
            'CLUDA': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'DDC':          {'learning_rate': 0.01,     'src_cls_loss_wt':  0.1593, 'domain_loss_wt': 0.2048},
            'HoMM':         {'learning_rate':0.001,     'src_cls_loss_wt': 0.2429,  'domain_loss_wt': 0.9824},
            'CoDATS':       {'learning_rate': 0.0005,   'src_cls_loss_wt': 0.5416,  'domain_loss_wt': 0.5582},
            'DSAN':         {'learning_rate': 0.005,    'src_cls_loss_wt':0.4133,   'domain_loss_wt': 0.16},
            'AdvSKM':       {'learning_rate': 0.001,    'src_cls_loss_wt': 0.4637,  'domain_loss_wt': 0.1511},
            'MMDA':         {'learning_rate': 0.001,    'src_cls_loss_wt': 0.9505,  'mmd_wt': 0.5476,           'cond_ent_wt': 0.5167,  'coral_wt': 0.5838, },
            'CDAN':         {'learning_rate': 0.001,    'src_cls_loss_wt': 0.6636,  'domain_loss_wt': 0.1954,   'cond_ent_wt':0.0124},
             "AdaMatch":      {'learning_rate': 3e-3,     'tau': 0.9, 'max_segments': 5, 'jitter_scale_ratio': 0.01, 'jitter_ratio': 0.2},
            'DIRT':         {'learning_rate': 0.001,    'src_cls_loss_wt': 0.9752,  'domain_loss_wt': 0.3892,   'cond_ent_wt': 0.09228,  'vat_loss_wt': 0.1947},
            'CoTMix': {
                'learning_rate': 0.001,
                'mix_ratio': 0.52,
                'temporal_shift': 14,
                'src_cls_weight': 0.8,
                'src_supCon_weight': 0.1,
                'trg_cont_weight': 0.1,
                'trg_entropy_weight': 0.05,
            }
        }




class pkecg():
    def __init__(self):
        super(pkecg, self).__init__()
        self.train_params = {
                #'num_epochs': 300,
            'num_epochs': 300,
                'batch_size':  64,
                'weight_decay': 1e-4,
        }
        self.alg_hparams = {
            'RAINCOAT':         {'learning_rate': 0.0005,   'src_cls_loss_wt': 0.9603,  'domain_loss_wt':0.9238},
            'DANN':         {'learning_rate': 0.0005,   'src_cls_loss_wt': 0.9603,  'domain_loss_wt':0.9238},
            'Deep_Coral':   {'learning_rate': 0.0005,   'src_cls_loss_wt': 0.05931, 'coral_wt': 8.452},
            'SASA': {'learning_rate': 5e-4, 'src_cls_loss_wt': 0.5, 'domain_loss_wt': 0.5},
            'DDC':          {'learning_rate': 0.01,     'src_cls_loss_wt':  0.1593, 'domain_loss_wt': 0.2048},
            'HoMM':         {'learning_rate':0.001,     'src_cls_loss_wt': 0.2429,  'domain_loss_wt': 0.9824},
            'CoDATS':       {'learning_rate': 0.0005,   'src_cls_loss_wt': 0.5416,  'domain_loss_wt': 0.5582},
            'TestTime_adapt': {'learning_rate': 5e-4, 'src_cls_loss_wt': 0.5, 'domain_loss_wt': 0.5},
            'SepReps': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
           'SepRepTranAlignEnd': {
                                     'learning_rate': 1e-3,
                                     'src_cls_loss_wt': 1.613,
                                     'domain_loss_wt': 1.857},
                                 'SepAligThenAttn': {
            'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
        'SSSS_TSA': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'SepAligThenAttnMMD': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
        'SepAligThenNoAttnSink': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
        'SepAligThenSum': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
        'NoSepAligThenAttnSink': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
        'SepAligThenAttnSinkFreq': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
        # 'DANN': {'learning_rate': 1e-2, 'src_cls_loss_wt': 9.74, 'domain_loss_wt': 0.0001},
        'CLUDA': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
        'Supervised': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.0},
            'Supervised_trg': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.0},
            'TestTime_adapt': {'learning_rate': 5e-4, 'src_cls_loss_wt': 0.5, 'domain_loss_wt': 0.5},
        'SinkDiv_Alignment': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.0, 'sinkdiv_loss_wt': 1.0},
            'DSAN':         {'learning_rate': 0.005,    'src_cls_loss_wt':0.4133,   'domain_loss_wt': 0.16},
            'AdvSKM':       {'learning_rate': 0.001,    'src_cls_loss_wt': 0.4637,  'domain_loss_wt': 0.1511},
            'MMDA':         {'learning_rate': 0.001,    'src_cls_loss_wt': 0.9505,  'mmd_wt': 0.5476,           'cond_ent_wt': 0.5167,  'coral_wt': 0.5838, },
            'CDAN':         {'learning_rate': 0.001,    'src_cls_loss_wt': 0.6636,  'domain_loss_wt': 0.1954,   'cond_ent_wt':0.0124},
            'DIRT':         {'learning_rate': 0.001,    'src_cls_loss_wt': 0.9752,  'domain_loss_wt': 0.3892,   'cond_ent_wt': 0.09228,  'vat_loss_wt': 0.1947},
            'CoTMix': {
                'learning_rate': 0.001,
                'mix_ratio': 0.9,
                'temporal_shift': 14,
                'src_cls_weight': 0.78,
                'src_supCon_weight': 0.1,
                'trg_cont_weight': 0.1,
                'trg_entropy_weight': 0.05
            }
        }
