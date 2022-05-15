import torch
import numpy as np
import torch.nn.functional as F
from scipy.io import loadmat
from scipy import  signal
from model import model
from parse_config import ConfigParser
from matplotlib import  pyplot as plt
from utils.util import sen_spe

sf_EEG = 250
sf_ACC = 20


"""function define"""
def DataPreprocess(EEG,ACC):

    def ButterFilter(data, order=2,cutoff=0.3,fs = 128,model = 'highpass'):
        wn = 2 * cutoff / fs
        b, a = signal.butter(order, wn, model, analog=False)
        output = signal.filtfilt(b, a, data, axis=0)
        return output

    EEG = ButterFilter(EEG, order=4, cutoff=0.3, fs=sf_EEG, model='highpass')
    # EEG[abs(EEG) > 150] = 0
    EEG = ButterFilter(EEG, order=4, cutoff=35, fs=sf_EEG, model='lowpass')
    DataTotal = [EEG, ACC]
    return DataTotal



"""Parameter Define"""
model_path = r'..\Model_best\model_best.pth'
device ='cpu'
data_path = r'..\DataForTrain\DataForTrain\1\Labled_Data.mat'
EEG = np.array(loadmat(data_path)['EEG_total'][0, :])
ACC = np.array(loadmat(data_path)['ACC_total'][0, :])
label_path  =  r'..\DataForTrain\DataForTrain\1\Labels.mat'


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
    OutPut_Total.append(OutPut)

OutPut_Total = np.array(OutPut_Total).flatten()
label = np.array(loadmat(label_path)['Label_total']).flatten().astype(int)

np.savetxt(r"C:\Users\Administrator\Desktop\\1.txt",OutPut_Total)


x = np.array(range(0,len(label)))
plt.subplot(211)
ax = plt.plot(x,OutPut_Total)
plt.xlabel('时间（h）',fontsize = 15)
plt.ylabel('睡眠分期',fontsize = 15)
plt.yticks([0,1,2,3],['N3','N1/N2','REM','WAKE'],fontsize = 18)
plt.xticks(fontsize = 15)
plt.title('AI分期结果',fontsize = 20)

plt.subplot(212)
plt.plot(x,label)
plt.xlabel('时间（h）',fontsize = 15)
plt.ylabel('睡眠分期',fontsize = 15)
plt.yticks([0,1,2,3],['N3','N1/N2','REM','WAKE'],fontsize = 18)
plt.xticks(fontsize = 15)
plt.title('专家分期结果',fontsize = 20)

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


A = OutPut_Total ==label
B = A[A==True]
acc = round(len(B)/len(label)*100,1)
plt.show()
sen,spe = sen_spe(OutPut_Total,label)
print('Accurancy',acc)
print('Sensitivity',sen)
print('Specificity',spe)






"""result test"""

