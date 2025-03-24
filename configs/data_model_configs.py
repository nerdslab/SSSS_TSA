"This is for channel pertubated case. 1 scenario only"


def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class HAR_UCI():
    def __init__(self):
        super(HAR_UCI, self)
        self.scenarios = [("2", "11"), ("6", "23"),("7", "13"),("9", "18"),("12", "16"),\
            ("13", "19"),  ("18", "21"), ("20", "6"),("23", "13"),("24", "12")]
        #self.scenarios = [("12", "16"),("6", "23"), ("23", "13")]
        self.scenarios = [("12", "16")]
        self.class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']
        self.sequence_len = 128
        self.shuffle = True
        self.drop_last = True
        self.normalize = True
        self.balanced = 0
        # model configs
        self.input_channels = 9
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6
        self.fourier_modes = 64
        self.out_dim = 192

        self.temp = 10
        # CNN and RESNET features
        
        self.mid_channels = 64
        self.final_out_channels = 64
        self.true_final_out_channels =64
        self.features_len = 1


        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.hidden_dim = 500
        self.DSKN_disc_hid = 128

        # CLUDA paramaters

        self.K = 128000  # que size
        self.m = 0.999  # momentum
        self.num_neighbors = 1
        self.T = 0.07
        self.use_batch_norm = True
        self.mlp_hidden_dim = 256
        self.use_mask = False
        self.input_static_dim= 0
class Mean_shift_simulations():
    def __init__(self):
        super(Mean_shift_simulations, self).__init__()
        self.class_names = ['1', '2', '3', '4']
        self.sequence_len = 128
        self.balanced = 0
        #self.scenarios = [('10_chnl_4_bad_1', '11_chnl_4_bad_1')]  # ,\
        self.scenarios = [('10_chnl_24_bad_5', '11_chnl_24_bad_5')]  # ,\
        #self.scenarios = [('10_chnl_24_bad_15', '11_chnl_24_bad_15')]  # ,\
        self.scenarios  =[('10_chnl_12_bad_9','11_chnl_12_bad_9')]
        self.num_classes = 4
        self.shuffle = True
        self.drop_last = False
        self.normalize = False

        # model configs
        #self.input_channels = 4
        self.input_channels = 12
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.01
        #self.temp=1 for 4 channel example
        self.temp = 1e-5 #for 24 channel bad 5
        self.temp = 5
        self.temp=1
        self.width = 64  # for FNN
        self.fourier_modes = 64
        # features

        self.mid_channels = 128
        # self.final_out_channels = 150
        #self.final_out_channels = 64
        self.final_out_channels = 64
        self.true_final_out_channels = 64
        self.out_dim = 128
        self.features_len = 1

        # TCN features
        self.tcn_layers = [64, 64, 64]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 5
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500

        # CLUDA paramaters

        self.K = 100000  # que size
        self.m = 0.999  # momentum
        self.num_neighbors = 1
        self.T = 0.07
        self.use_batch_norm = True
        self.mlp_hidden_dim = 256

        self.input_static_dim = 0
        self.use_mask = False


class Mean_shift_simulations_gauss_noise():
    def __init__(self):
        super(Mean_shift_simulations_gauss_noise, self).__init__()
        self.class_names = ['1', '2', '3', '4']
        self.sequence_len = 128
        self.balanced = 1

        self.add_gauss = True
        self.drop_chnl = False
        self.noise = 2

        self.no_channel_affect = 11

        #self.scenarios = [('10_chnl_4_bad_1', '11_chnl_4_bad_1')]  # ,\
        self.scenarios = [('10_chnl_24_bad_5', '11_chnl_24_bad_5')]  # ,\
        #self.scenarios = [('10_chnl_24_bad_15', '11_chnl_24_bad_15')]  # ,\
        self.scenarios  =[('10_chnl_12_bad_9',f'10_chnl_12_bad_9')]

        self.num_classes = 4
        self.shuffle = True
        self.drop_last = False
        self.normalize = False

        # model configs
        #self.input_channels = 4
        self.input_channels = 12
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.01
        #self.temp=1 for 4 channel example
        self.temp = 1e-5 #for 24 channel bad 5
        self.temp = 5
        self.temp=1

        self.width = 64  # for FNN
        self.fourier_modes = 64
        # features

        self.mid_channels = 128
        # self.final_out_channels = 150
        #self.final_out_channels = 64
        self.final_out_channels = 64
        self.true_final_out_channels = 64
        self.out_dim = 128
        self.features_len = 1

        # TCN features
        self.tcn_layers = [64, 64, 64]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 5
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500

        # CLUDA paramaters

        self.K = 100000  # que size
        self.m = 0.999  # momentum
        self.num_neighbors = 1
        self.T = 0.07
        self.use_batch_norm = True
        self.mlp_hidden_dim = 256

        self.input_static_dim = 0
        self.use_mask = False
class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        # data parameters
        self.num_classes = 5
        self.class_names = ['W', 'N1', 'N2', 'N3', 'REM']
        self.sequence_len = 3000
        self.scenarios = [("0", "11"), ("2", "5"), ("12", "5"), ("7", "18"), ("16", "1"), ("9", "14"),\
            ("4", "12"),("10", "7"),("6", "3"),("8", "10")]
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 1
        self.kernel_size = 25
        self.stride = 6
        self.dropout = 0.2

        # features
        self.mid_channels = 32
        self.final_out_channels = 128
        self.features_len = 1
        self.fourier_modes = 300
        self.out_dim = 256
        # TCN features
        self.tcn_layers = [32,64]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 15 # 25
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        self.disc_hid_dim = 100

        # CLUDA paramaters

        self.K = 1000  # que size
        self.m = 0.999  # momentum
        self.num_neighbors = 1
        self.T = 0.07
        self.use_batch_norm = True
        self.mlp_hidden_dim = 256

        self.input_static_dim = 0

class WISDM(object):
    def __init__(self):
        super(WISDM, self).__init__()
        self.class_names = ['walk', 'jog', 'sit', 'stand', 'upstairs', 'downstairs']
        self.sequence_len = 128
        self.balanced = 1



        self.scenarios = [('20', '30'), ('12', '19'), ('30', '20'), ("2", "32"), ("7", "30"), ('12', '7'), ('18', '20'), \
                          ("19", "30"), ('4','19'), ('26', '2')]



        self.num_classes = 6
        self.shuffle = True

        #FOR CLUDA has to be true.. otherwise cras
        self.drop_last = False
        self.normalize = True

        #self.temp = 5


        self.temp = 3
        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.01
        self.num_classes = 6
        self.width = 64  # for FNN
        self.fourier_modes = 64
        # features
        self.mid_channels = 128
        #self.final_out_channels = 150
        self.final_out_channels = 64
        self.true_final_out_channels = 64
        self.out_dim = 128
        self.features_len = 1


        # TCN features
        self.tcn_layers = [64,64,64]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 5
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500


        #CLUDA paramaters


        self.K = 128000 #que size
        self.m = 0.999 #momentum
        self.num_neighbors =1
        self.T =0.07
        self.use_batch_norm = True
        self.mlp_hidden_dim = 256

        self.input_static_dim = 0
        self.use_mask = False

class HHAR_SA(object):  ## HHAR dataset, SAMSUNG device.
    def __init__(self):
        super(HHAR_SA, self).__init__()
        self.sequence_len = 128
        self.balanced = 0
        # self.scenarios = [("0", "2")]
        self.scenarios = [("0", "2"), ("1", "6"),("2", "4"),("4", "0"),("4", "5"),\
            ("5", "1"),("5", "2"),("7", "2"),("7", "5"),("8", "4")]
        #seems like 2 is a bad subject
        #self.scenarios = [ ("0", "2"), ("8", "4")]
        self.class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']
        self.num_classes = 6
        self.shuffle = True
        self.drop_last = True
        self.normalize = True
        self.fourier_modes = 32
        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.temp = 3
        #self.temp = 1
        # features
        self.mid_channels = 64 #* 2
        self.final_out_channels = 64
        self.true_final_out_channels = 64


        self.features_len = 1
        self.out_dim = 128

        # TCN features
        self.tcn_layers = [75,150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500

        # CLUDA paramaters

        self.K = 128000  # que size
        self.m = 0.999  # momentum
        self.num_neighbors = 1
        self.T = 0.07
        self.use_batch_norm = True
        self.mlp_hidden_dim = 256

        self.input_static_dim = 0
        self.use_mask = False
class Boiler(object):
    def __init__(self):
        super(Boiler, self).__init__()
        self.class_names = ['0','1']
        self.sequence_len = 32
        self.scenarios = [("1", "3"),("1","2"),("2", "3"),("2","1"),("3", "1"),("3", "2")]
        self.num_classes = 2
        self.sequence_len = 32
        self.shuffle = True
        self.drop_last = True
        self.normalize = True
        self.fourier_modes = 10
        self.balanced = 1
        self.temp = 0.1
        # model configs
        self.input_channels = 20
        self.kernel_size = 3
        self.stride = 1
        self.dropout = 0.2

        # features
        self.mid_channels = 32
        self.final_out_channels = 64
        self.true_final_out_channels = 64
        self.features_len = 1

        # TCN features
        self.tcn_layers = [32,64]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 25
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        self.disc_hid_dim = 100

        # CLUDA paramaters

        self.K = 128000  # que size
        self.m = 0.999  # momentum
        self.num_neighbors = 1
        self.T = 0.07
        self.use_batch_norm = True
        self.mlp_hidden_dim = 256

        self.input_static_dim = 0
        self.use_mask = False

class miniTimeMatch(object):
    def __init__(self):
        super(miniTimeMatch, self).__init__()
        self.class_names = ['0','1']
        self.sequence_len = 62
        self.scenarios = [("0", "1"),("0","2"),("0", "3"),("3","1"),("3", "2"),("2", "1")]
        self.num_classes = 8
        self.sequence_len = 62
        self.shuffle = True
        self.drop_last = True
        self.normalize = True
        self.fourier_modes = 10
        self.balanced = 0
        self.temp = 10
        # model configs
        self.input_channels = 10
        self.kernel_size = 3
        self.stride = 1
        self.dropout = 0.2

        # features
        self.mid_channels = 32
        self.final_out_channels = 64
        self.true_final_out_channels = 64
        self.features_len = 1

        # TCN features
        self.tcn_layers = [32,64]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 25
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        self.disc_hid_dim = 100

        # CLUDA paramaters

        self.K = 128000  # que size
        self.m = 0.999  # momentum
        self.num_neighbors = 1
        self.T = 0.07
        self.use_batch_norm = True
        self.mlp_hidden_dim = 256

        self.input_static_dim = 0
        self.use_mask = False

class mimic(object):
    def __init__(self):
        super(mimic, self).__init__()
        self.class_names = ['0','1']
        self.sequence_len = 300
        self.scenarios = [("1", "2"),("1","3"),("1", "4"),("2","1"),("2", "3"),("2", "4"),("3", "1"),("3", "2"),("3", "4"),("4", "1"),("4", "2"),("4", "3")]
        self.num_classes = 2
        self.sequence_len = 300
        self.shuffle = True
        self.drop_last = True
        self.normalize = True
        self.fourier_modes = 10
        self.balanced = 1
        self.temp = 4
        # model configs
        self.input_channels = 45
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.2

        # features
        self.mid_channels = 32
        self.final_out_channels = 4
        self.true_final_out_channels = 4
        self.features_len = 1

        # TCN features
        self.tcn_layers = [32,64]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 25
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        self.disc_hid_dim = 100

        # CLUDA paramaters

        self.K = 128000  # que size
        self.m = 0.999  # momentum
        self.num_neighbors = 1
        self.T = 0.07
        self.use_batch_norm = True
        self.mlp_hidden_dim = 256

        self.input_static_dim = 0
        self.use_mask = False
class robots(object):
    def __init__(self):
        super(robots, self).__init__()
        self.class_names = ['0','1']
        self.sequence_len = 24
        self.scenarios = [("a", "a"),("b", "b")]
        self.num_classes = 5
        self.sequence_len = 24
        self.temp = 1
        self.shuffle = True
        self.drop_last = True
        self.normalize = True
        self.fourier_modes = 10
        self.balanced = 0
        # model configs
        self.input_channels = 12
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.2


        self.noise = 1

        self.no_channel_affect = [2,4,6,8,10]

        # features
        self.mid_channels = 32
        self.final_out_channels = 64
        self.true_final_out_channels = 64
        self.features_len = 1

        # TCN features
        self.tcn_layers = [32,64]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 15# 25
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        self.disc_hid_dim = 100

        # CLUDA paramaters

        self.K = 128000  # que size
        self.m = 0.999  # momentum
        self.num_neighbors = 1
        self.T = 0.07
        self.use_batch_norm = True
        self.mlp_hidden_dim = 256

        self.input_static_dim = 0
        self.use_mask = False

class pkecg(object):
    def __init__(self):
        super(pkecg, self).__init__()
        self.class_names = ['0','1','2','3','4','5']
        self.sequence_len = 1000
        self.scenarios = [("0", "1"),("1", "0"),("0","2"),("1","2"),("2","0"),("2","1")]
        self.num_classes = 6
        self.sequence_len = 1000
        self.temp = 1
        self.shuffle = True
        self.drop_last = True
        self.normalize = True
        self.fourier_modes = 10
        self.balanced = 0
        # model configs
        self.input_channels = 12
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.2


        self.noise = 1

        self.no_channel_affect = [2,4,6,8,10]

        # features
        self.mid_channels = 32
        self.final_out_channels = 64
        self.true_final_out_channels = 64



        self.features_len = 1

        # TCN features
        self.tcn_layers = [32,64]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 15# 25
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        self.disc_hid_dim = 100

        # CLUDA paramaters

        self.K = 128000  # que size
        self.m = 0.999  # momentum
        self.num_neighbors = 1
        self.T = 0.07
        self.use_batch_norm = True
        self.mlp_hidden_dim = 256

        self.input_static_dim = 0
        self.use_mask = False
class mtrimg(object):
    def __init__(self):
        super(mtrimg, self).__init__()
        self.class_names = ['0','1']
        self.sequence_len = 3000
        self.scenarios = [("b", "a"),("a", "b")]
        self.num_classes = 2
        self.sequence_len = 3000
        self.temp = 4
        self.shuffle = True
        self.drop_last = True
        self.normalize = True
        self.fourier_modes = 10
        self.balanced = 0
        # model configs
        self.input_channels = 64
        self.kernel_size = 10
        self.stride = 1
        self.dropout = 0.2

        # features
        self.mid_channels = 32
        self.final_out_channels = 64
        self.true_final_out_channels = 64
        self.features_len = 1

        # TCN features
        self.tcn_layers = [32,64]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 15# 25
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        self.disc_hid_dim = 100

        # CLUDA paramaters

        self.K = 128000  # que size
        self.m = 0.999  # momentum
        self.num_neighbors = 1
        self.T = 0.07
        self.use_batch_norm = True
        self.mlp_hidden_dim = 256

        self.input_static_dim = 0
        self.use_mask = False
