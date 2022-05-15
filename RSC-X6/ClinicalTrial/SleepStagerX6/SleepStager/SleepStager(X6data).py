import torch
import numpy as np
import torch.nn.functional as F
from scipy.io import loadmat
from scipy import  signal
from model import model
from parse_config import ConfigParser
from matplotlib import  pyplot as plt
from utils.util import *

sf_EEG = 250
sf_ACC = 20
pid =18

"""function define"""
def DataPreprocess(EEG,ACC):

    def ButterFilter(data, order=2,cutoff=0.3,fs = 128,model = 'highpass'):
        wn = 2 * cutoff / fs
        b, a = signal.butter(order, wn, model, analog=False)
        output = signal.filtfilt(b, a, data, axis=0)
        return output

    EEG = ButterFilter(EEG, order=4, cutoff=0.3, fs=sf_EEG, model='highpass')
    EEG = ButterFilter(EEG, order=4, cutoff=100, fs=sf_EEG, model='lowpass')
    DataTotal = [EEG, ACC]
    return DataTotal



"""Parameter Define"""
model_path = r'..\Model_best\model_best.pth'
device ='cpu'
data_path = fr'..\DataForTrain\RawData\{pid}'

EEG = np.array(DataloaderX6(data_path+'\EEG.eeg',signed=False))
ACC_p = np.array(DataloaderX6(data_path+'\ACC.acc',signed=True))

ACCx = ACC_p.reshape((-1,3))[1:,0]-ACC_p.reshape((-1,3))[0:-1,0]
ACCy = ACC_p.reshape(-1,3)[1:,1]-ACC_p.reshape(-1,3)[0:-1,1]
ACCz = ACC_p.reshape(-1,3)[1:,2]-ACC_p.reshape(-1,3)[0:-1,2]
ACC = np.abs(ACCx)+np.abs(ACCy)+np.abs(ACCz)

EEG=EEG[:len(EEG)//sf_EEG*sf_EEG]
ACC=ACC[:len(EEG)//sf_EEG*sf_ACC]


# plt.figure(1)
# plt.plot(EEG)
# plt.figure(2)
# plt.plot(ACC)
# plt.show()

label_path  =  fr'..\DataForTrain\DataForTrain\{pid}\Labels.mat'


"""Model load"""
SleepStager = torch.load(model_path)
SleepStager.eval().to(device)
print(SleepStager)

DataTotal = DataPreprocess(EEG, ACC)
SleepEpoch_n = len(DataTotal[0]) // (sf_EEG* 30)

"""Model run"""
OutPut_Total = []
for i in range(SleepEpoch_n):
    EEG_temp = DataTotal[0][i*sf_EEG*30:(i+1)*sf_EEG*30]
    ACC_temp = DataTotal[1][i*sf_ACC*30:(i+1)*sf_ACC*30]
    Data_temp = torch.from_numpy(np.concatenate((EEG_temp,ACC_temp),axis=0))
    Data_temp = Data_temp.type(torch.FloatTensor).resize(1,1,len(Data_temp))
    Data_temp.to(device)

    OutPut_T = SleepStager(Data_temp)
    OutPut_T = OutPut_T.data
    OutPut_T = F.softmax(OutPut_T)
    OutPut = OutPut_T.argmax(1)
    OutPut = int(OutPut.cpu().tolist()[0])
    # if OutPut == 4:
    #     OutPut = 2.5
    OutPut_Total.append(OutPut)

OutPut_Total = np.array(OutPut_Total).flatten()
# label = np.array(loadmat(label_path)['Label_total']).flatten().astype(int)

np.savetxt(r"C:\Users\Administrator\Desktop\\1.txt",OutPut_Total)


x1 = np.array(range(0,len(OutPut_Total)))*0.5/60
plt.subplot(211)
ax = plt.plot(x1,OutPut_Total)
plt.xlabel('时间（h）',fontsize = 15)
plt.ylabel('睡眠分期',fontsize = 15)
plt.yticks([0,1,2,3,4],['N3','N1/N2','REM','WAKE','PoorContact'],fontsize = 18)
plt.xticks(fontsize = 15)
plt.title('AI分期结果',fontsize = 20)

# x2 = np.array(range(0,len(label)))*0.5/60
# plt.subplot(212)
# plt.plot(x2,label)
# plt.xlabel('时间（h）',fontsize = 15)
# plt.ylabel('睡眠分期',fontsize = 15)
# plt.yticks([0,1,2,3,4],['N3','N1/N2','REM','WAKE','PoorContact'],fontsize = 18)
# plt.xticks(fontsize = 15)
# plt.title('专家分期结果',fontsize = 20)

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.show()


# A = OutPut_Total ==label
# B = A[A==True]
# acc = round(len(B)/len(label)*100,1)

# sen,spe = sen_spe(OutPut_Total,label)
# print('Accurancy',acc)
# print('Sensitivity',sen)
# print('Specificity',spe)






"""result test"""

