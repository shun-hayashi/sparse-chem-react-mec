import numpy as np

param = {"path_tc": "time_course",    
         "tc_list_all": ['0_1', '0_2', '0_3', '0_4', \
                         '1_1', '1_2', '1_3', '1_4', \
                         '2_1', '2_2', '2_3', '2_4', \
                         '3_1', '3_2', '3_3', '3_4', \
                         '4_1', '4_2', '4_3', '4_4'],
         "tc_list_split_train": ['0_2', '0_3', '1_2', '1_3', \
                                 '2_2', '2_3', '3_2', '3_3'],
         "tc_list_split_test": ['0_1', '1_1', '2_1', '3_1', \
                                '0_4', '1_4', '2_4', '3_4', \
                                '4_1', '4_2', '4_3', '4_4'],
         "points": 50
        }

def generate_d(mode="split"):
    path = param["path_tc"]
    data_all = param["tc_list_all"]
    data_split_train = param["tc_list_split_train"]
    data_split_test = param["tc_list_split_test"]
    if mode == "all":
        d = generate_d_from_data(path, data_all)
        return d, None
    elif mode == "split":
        d_train, d_test = generate_d_train_test_from_data(path, data_split_train, data_split_test)
    else:
        raise ValueError("error in generate_d") 
    return d_train, d_test

def generate_d_from_data(path_time, use_data):
    num_plot= param["points"]
    conc = np.empty((len(use_data), 4, num_plot,)) #print(conc.shape) # (20, 4, 50)
    for i in range(len(use_data)):
        conc[i,:,:] = np.loadtxt(path_time +"/"+ use_data[i] + "/conc.csv", delimiter=",")
    d = np.empty(conc.shape)
    t_max = np.max(conc[:,0,:]) # print(t_max) # 1672.17
    conc_max = np.max(conc[:,1:,:]) # print(conc_max) # 0.0001156096
    d[:,0,:] = conc[:,0,:] / t_max
    d[:,1:,:] = conc[:,1:,:] / conc_max
    return d

def generate_d_train_test_from_data(path_time, use_data_train, use_data_test):
    num_plot= param["points"]
    conc_train = np.empty((len(use_data_train), 4, num_plot,)) #print(conc.shape) # (20, 4, 50)
    for i in range(len(use_data_train)):
        conc_train[i,:,:] = np.loadtxt(path_time +"/"+ use_data_train[i] + "/conc.csv", delimiter=",")
    conc_test = np.empty((len(use_data_test), 4, num_plot,)) #print(conc.shape) # (20, 4, 50)
    for i in range(len(use_data_test)):
        conc_test[i,:,:] = np.loadtxt(path_time +"/"+ use_data_test[i] + "/conc.csv", delimiter=",")
    d_train = np.empty(conc_train.shape)
    d_test = np.empty(conc_test.shape)
    t_max = max(np.max(conc_train[:,0,:]),np.max(conc_test[:,0,:]))# print(t_max) # 1672.17
    conc_max = max(np.max(conc_train[:,1:,:]),np.max(conc_test[:,1:,:])) # print(conc_max) # 0.0001156096
    d_train[:,0,:] = conc_train[:,0,:] / t_max
    d_train[:,1:,:] = conc_train[:,1:,:] / conc_max
    d_test[:,0,:] = conc_test[:,0,:] / t_max
    d_test[:,1:,:] = conc_test[:,1:,:] / conc_max
    return d_train, d_test
    