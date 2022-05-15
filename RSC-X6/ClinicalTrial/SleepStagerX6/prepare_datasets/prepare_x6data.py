import os
from scipy.io import loadmat,savemat
import  numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# parameters define
pid = 18
sf_EEG = 250
sf_ACC = 20
N =0

#function define
def ButterFilter(data, order=2,cutoff=0.3,fs = 128,model = 'highpass'):
    wn = 2 * cutoff / fs
    b, a = signal.butter(order, wn, model, analog=False)
    output = signal.filtfilt(b, a, data, axis=0)
    return output


path = '..\DataForTrain\DataForTrain'

for pid in range(len(os.listdir(path))):
    Data_total = []
    Label_total = []

    data_path=r'..\DataForTrain\DataForTrain\\'+str(pid+1)+'\Labled_Data.mat'
    label_path=r'..\DataForTrain\DataForTrain\\'+str(pid+1)+'\Labels.mat'
    L = loadmat(label_path)

    # data and label load
    EEG = np.array(loadmat(data_path)['EEG_total'])
    ACC = np.array(loadmat(data_path)['ACC_total'])
    # plt.plot(EEG.flatten())
    # plt.show()

    EEG = ButterFilter(EEG.flatten(),order=4,cutoff=0.3,fs=sf_EEG,model='highpass')
    # EEG[abs(EEG) > 150] = 0
    EEG = ButterFilter(EEG,order=4,cutoff=100,fs=sf_EEG,model='lowpass')
    # plt.plot(EEG.flatten())
    # plt.show()

    n_epochs = EEG.size//(30 * sf_EEG)
    EEG = np.array(np.split(EEG,n_epochs,axis=0)).reshape(n_epochs, sf_EEG * 30, 1)
    ACC = np.array(np.split(ACC,n_epochs,axis=1)).reshape(n_epochs,sf_ACC*30,1)


    Labels = loadmat(label_path)['Label_total']
    if len(Labels.flatten()) >n_epochs:
        Labels = Labels[0:-1]
    elif len(Labels.flatten()) <n_epochs:
        EEG = EEG[0:-1,:,:]
        ACC = ACC[0:-1,:,:]
        n_epochs = n_epochs-1
    # Label_total += Labels.flatten().tolist()
    Label_total = Labels.flatten()

    for i in range(EEG.shape[0]):
        EEG_T = EEG[i,:].flatten()
        ACC_T = ACC[i,:].flatten()
        Data_T = np.append(EEG_T,ACC_T*sf_ACC*30).tolist()
        Data_total.append(Data_T)

    print('loading patients data of No.'+str(pid+1))

    Data_total = np.array(Data_total)
    Label_total = np.array(Label_total)
    data_mat_path = '../DataForTrain/DataForTrain_NN/'+str(pid+1)+'/'
    label_mat_path = '../DataForTrain/DataForTrain_NN/'+str(pid+1)+'/'
    ptex = os.path.exists(data_mat_path)

    if not ptex:
        os.makedirs(data_mat_path)
    ptex = os.path.exists(label_mat_path)
    if not ptex:
        os.makedirs(label_mat_path)

    savemat(data_mat_path+'Data.mat', {'data': Data_total})
    savemat(label_mat_path+'Label.mat', {'label': Label_total})
    N =N+len(Label_total)

print('Total epoches = '+str(N))





