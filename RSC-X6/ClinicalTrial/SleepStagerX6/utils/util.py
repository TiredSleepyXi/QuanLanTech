import json
import tkinter
from pathlib import Path
from collections import OrderedDict
from itertools import repeat
import pandas as pd
import os
import numpy as np
from glob import glob
import math
import copy
import scipy.signal as signal
import matplotlib.pyplot as plt

def load_folds_data_shhs(np_data_path, n_folds):
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    r_p_path = r"utils/r_permute_shhs.npy"
    r_permute = np.load(r_p_path)
    npzfiles = np.asarray(files , dtype='<U200')[r_permute]
    train_files = np.array_split(npzfiles, n_folds)
    folds_data = {}
    for fold_id in range(n_folds):
        subject_files = train_files[fold_id]
        training_files = list(set(npzfiles) - set(subject_files))
        folds_data[fold_id] = [training_files, subject_files]
    return folds_data

def load_folds_data(np_data_path, n_folds):
    # files = sorted(glob(np_data_path+'\\'))
    files = []
    for f in os.listdir(np_data_path):
        files.append(np_data_path+f+'\\')


    # if "78" in np_data_path:
    #     r_p_path = r"utils/r_permute_78.npy"
    # else:
    #     r_p_path = r"utils/r_permute_20.npy"
    #
    # if os.path.exists(r_p_path):
    #     r_permute = np.load(r_p_path)
    # else:
    #     print ("============== ERROR =================")

    datalist =np.array(range(0,len(files)))

    r_permute  = np.random.shuffle(datalist)
    # files_dict = dict()
    files_pairs = []
    for i in files:
        files_pairs.append([i])
    files_pairs = np.array(files_pairs)

    files_pairs = files_pairs[r_permute].flatten()

    train_files = np.array_split(files_pairs, n_folds)
    folds_data = {}
    for fold_id in range(n_folds):
        subject_files = train_files[fold_id]
        subject_files = [item for item in subject_files ]
        files_pairs2 = [item for item in files_pairs]
        training_files = list(set(files_pairs2) - set(subject_files))
        folds_data[fold_id] = [training_files, subject_files]
    return folds_data


def calc_class_weight(labels_count):
    total = np.sum(labels_count)
    class_weight = dict()
    num_classes = len(labels_count)

    factor = 1 / num_classes
    mu = [factor * 1.5, factor * 2, factor * 1.5, factor, factor * 1.5] # THESE CONFIGS ARE FOR SLEEP-EDF-20 ONLY

    for key in range(num_classes):
        score = math.log(mu[key] * total / float(labels_count[key]))
        class_weight[key] = score if score > 1.0 else 1.0
        class_weight[key] = round(class_weight[key] * mu[key], 2)

    class_weight = [class_weight[i] for i in range(num_classes)]

    return class_weight


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rb') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def sen_spe(output,labels):
    """N3"""
    O_N3 = copy.deepcopy(output)
    O_N3[O_N3!=0] =6
    O_N3[O_N3==0] =1
    O_N3[O_N3==6] =0


    L_N3 = copy.deepcopy(labels)
    L_N3[L_N3!=0] =6
    L_N3[L_N3==0] =1
    L_N3[L_N3==6] =0


    cp = L_N3 == O_N3
    T = len(cp[cp == True])
    L = len(output)
    O_N3 =len(O_N3[O_N3 == 1])
    L_N3 =len(L_N3[L_N3 == 1])

    TP = T
    FP = L_N3
    FN = O_N3
    TN = L-True

    sen_N3 = round(TP/(TP+FN)*100,1)
    spe_N3 = round(TN/(TN+FP)*100,1)

    """N12"""
    O_N12 = copy.deepcopy(output)
    O_N12[O_N12!=1] =6
    O_N12[O_N12==1] =1
    O_N12[O_N12==6] =0


    L_N12 = copy.deepcopy(labels)
    L_N12[L_N12!=1] =6
    L_N12[L_N12==1] =1
    L_N12[L_N12==6] =0


    cp = L_N12 == O_N12
    T = len(cp[cp == True])
    L = len(output)
    O_N12 =len(O_N12[O_N12 == 1])
    L_N12 =len(L_N12[L_N12 == 1])

    TP = T
    FP = L_N12
    FN = O_N12
    TN = L-True

    sen_N12 = round(TP/(TP+FN)*100,1)
    spe_N12 = round(TN/(TN+FP)*100,1)

    """REM"""
    O_REM = copy.deepcopy(output)
    O_REM[O_REM!=2] =6
    O_REM[O_REM==2] =1
    O_REM[O_REM==6] =0


    L_REM = copy.deepcopy(labels)
    L_REM[L_REM!=2] =6
    L_REM[L_REM==2] =1
    L_REM[L_REM==6] =0


    cp = L_REM == O_REM
    T = len(cp[cp == True])
    L = len(output)
    O_REM =len(O_REM[O_REM == 1])
    L_REM =len(L_REM[L_REM == 1])

    TP = T
    FP = L_REM
    FN = O_REM
    TN = L-True

    sen_REM = round(TP/(TP+FN)*100,1)
    spe_REM = round(TN/(TN+FP)*100,1)


    """WAKE"""
    O_WAKE = copy.deepcopy(output)
    O_WAKE[O_WAKE!=3] =6
    O_WAKE[O_WAKE==3] =1
    O_WAKE[O_WAKE==6] =0


    L_WAKE = copy.deepcopy(labels)
    L_WAKE[L_WAKE!=3] =6
    L_WAKE[L_WAKE==3] =1
    L_WAKE[L_WAKE==6] =0


    cp = L_WAKE == O_WAKE
    T = len(cp[cp == True])
    L = len(output)
    O_WAKE =len(O_WAKE[O_WAKE == 1])
    L_WAKE =len(L_WAKE[L_WAKE == 1])

    TP = T
    FP = L_WAKE
    FN = O_WAKE
    TN = L-True

    sen_WAKE = round(TP/(TP+FN)*100,1)
    spe_WAKE = round(TN/(TN+FP)*100,1)

    sen = {'N3':sen_N3,'N1/2':sen_N12,'REM':sen_REM,'WAKE':sen_WAKE}
    spe = {'N3': spe_N3, 'N1/2': spe_N12, 'REM': spe_REM, 'WAKE': spe_WAKE}
    return  sen,spe

def DataloaderX6(filepth,signed=False):#for ACC, the sign is Ture
    headerL=91
    with open(filepth,'rb') as f:
       dataTL = len(f.read())

    f = open(filepth,'rb')
    f.seek(headerL,0)
    Data = []
    for j in range((dataTL - headerL)//2):
        b = f.read(2)
        # dataT = b[0]+b[1]*(16**2)
        dataT = int.from_bytes(b, byteorder='little', signed=signed)
        Data.append(dataT)
    Data = np.array(Data)
    return Data

#
# def DataloaderX6(filepth):
#     headerL = 91
#     with open(filepth, 'rb') as f:
#         dataTL = len(f.read())
#
#     f = open(filepth, 'rb')
#     f.seek(headerL, 0)
#     Data = []
#     for j in range((dataTL - headerL) // 2):
#         b = f.read(2)
#         dataT = int.from_bytes(b, byteorder='little', signed=True)
#
#         Data.append(dataT)
#     Data = np.array(Data)
#     return Data

def ButterFilter(data, order=2,cutoff=0.3,fs = 128,model = 'highpass'):
    wn = 2 * cutoff / fs
    b, a = signal.butter(order, wn, model, analog=False)
    output = signal.filtfilt(b, a, data, axis=0)
    return output






def stft_specgram(x, fs,co=1):    #picname是给图像的名字，为了保存图像
# def stft_specgram(x, fs):
    f, t, Zxx = signal.stft(x, fs, nperseg=25000)
    # amp = 2 * np.sqrt(2)
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=co*min(x), vmax=co*max(x))
    # plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp)
    plt.title('STFT Magnitude')
    plt.ylim(0,60)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
