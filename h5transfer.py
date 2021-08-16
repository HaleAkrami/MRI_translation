import numpy as np
import h5py
disk_path = '/ImagePTE1/akrami/CamCan/'
x_train = np.load(disk_path + 't1_data_train_CAMCAN.npz')['data']
y_train = np.load(disk_path + 't2_data_train_CAMCAN.npz')['data']
hf = h5py.File('/big_disk/akrami/data_train.h5', 'w')
hf.create_dataset('x_train', data= x_train)
hf.create_dataset('y_train', data= y_train)