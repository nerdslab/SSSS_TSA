def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class HAR_UCI():
    def __init__(self):
        super(HAR_UCI, self)
        self.scenarios = [("2", "11")]
        #self.scenarios = [("12", "16"),("6", "23"), ("23", "13")]
        self.class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']
        self.sequence_len = 128
        self.shuffle = True
        self.drop_last = True
        self.normalize = True
        self.balanced = 0

        self.noise = 2


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
        self.final_out_channels = 128
        self.true_final_out_channels =128
        self.features_len = 1

        #for dropping channels
        self.temp = 4
        #self.no_channel_affect = [4,5,6]
        self.no_channel_affect = [1,2,3,4,5,6,7,8]
        self.no_channel_affect = [1,2,3,4,5,6,7,8]
        self.no_channel_affect = [ 2,  4,  6]
        if max(self.no_channel_affect) > self.input_channels:
            raise ValueError("Can not have no of channel pertubs to be greater than than the numb of channels")

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

        self.K = 1000  # que size
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
        self.scenarios = [('10_chnl_4_bad_1', '11_chnl_4_bad_1')]  # ,\
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



class WISDM(object):
    def __init__(self):
        super(WISDM, self).__init__()
        self.class_names = ['walk', 'jog', 'sit', 'stand', 'upstairs', 'downstairs']
        self.sequence_len = 128
        self.balanced = 0
        # Closed Set DA
        #self.scenarios = [('20','30'),('12','19'),('4','15'),("2", "32"), ("7", "30"),('12','7'),('18','20'),\
        #                ("21", "31") ,("25", "29"), ('26','2')]

        self.scenarios = [ ('20','30'),('12','19'),('4','15'), ("2", "32"), ("7", "30"), ('12', '7'), ('18', '20'), \
                 ("21", "31") ,("25", "29"), ('26','2')]


        #self.scenarios = [('20', '30'), ('12', '19'), ('4', '15')]
        self.scenarios = [("2", "32"), ("7", "30"),('12','7'),('18','20'),
                      ("21", "31"),("25", "29"), ('26','2')]

        #self.scenarios = [('20','30'),('12','19'),("4", "15"),("2", "32")] #,("25", "29"), ('26','2')]
        #self.scenarios =[("2", "32"),('12','7'),('18','20'),\
        #                ("21", "31")]
        #self.scenarios =[("7", "30")]
        #self.scenarios =[('12','7')]
        self.num_classes = 6
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        self.temp = 5
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

        self.K = 1000  # que size
        self.m = 0.999  # momentum
        self.num_neighbors = 1
        self.T = 0.07
        self.use_batch_norm = True
        self.mlp_hidden_dim = 256

        self.input_static_dim = 0
        self.use_mask = False



