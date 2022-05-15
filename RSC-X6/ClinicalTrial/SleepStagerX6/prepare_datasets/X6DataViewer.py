from utils.util import DataloaderX6,ButterFilter,DataloaderX6,stft_specgram
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio

sf_eeg=250
sf_acc=20
pid = 21
EEG_p=fr'E:\QuanLanProject\RSC\X6\文档\算法开发\睡眠分期算法开发\ModelDeveloper\X6SleepStager2.0 (X6data)\AttnSleep-main - X6\DataForTrain\RawData\{pid}\EEG.eeg'
ACC_p=fr'E:\QuanLanProject\RSC\X6\文档\算法开发\睡眠分期算法开发\ModelDeveloper\X6SleepStager2.0 (X6data)\AttnSleep-main - X6\DataForTrain\RawData\{pid}\ACC.acc'
EEG = np.array(DataloaderX6(EEG_p,signed=False))
ACCx = np.array(DataloaderX6(ACC_p,signed=True)).reshape(-1,3)[1:,0]-np.array(DataloaderX6(ACC_p,signed=True)).reshape(-1,3)[0:-1,0]
ACCy = np.array(DataloaderX6(ACC_p,signed=True)).reshape(-1,3)[1:,1]-np.array(DataloaderX6(ACC_p,signed=True)).reshape(-1,3)[0:-1,1]
ACCz = np.array(DataloaderX6(ACC_p,signed=True)).reshape(-1,3)[1:,2]-np.array(DataloaderX6(ACC_p,signed=True)).reshape(-1,3)[0:-1,2]
ACC = np.abs(ACCx)+np.abs(ACCy)+np.abs(ACCz)
# ACC_raw=DataloaderX6_ACC(ACC_p)
# ACCx = np.array(DataloaderX6_ACC(ACC_p)).reshape(-1,3)[:,0]
# ACCy = np.array(DataloaderX6_ACC(ACC_p)).reshape(-1,3)[:,1]
# ACCz = np.array(DataloaderX6_ACC(ACC_p)).reshape(-1,3)[:,2]


x = np.array(range(len(EEG)))/sf_eeg/60
# ACC = np.abs(ACCx)+np.abs(ACCy)+np.abs(ACCz)
plt.plot(x,EEG)
# plt.plot(x,ACC)
plt.show()
scio.savemat(fr'E:\QuanLanProject\RSC\X6\文档\算法开发\睡眠分期算法开发\ModelDeveloper\X6SleepStager2.0 (X6data)\AttnSleep-main - X6\DataForTrain\RawData\{pid}\Data.mat',{'EEG_total':EEG,'ACC_total':ACC})


x1 = np.arange(len(EEG))/sf_eeg
x2 = np.arange(len(ACCx))/sf_acc

print('Time of EEG',len(x1)/sf_eeg/60)
print('Time of ACC',len(x2)/sf_acc/60)

# plt.subplot(211)
EEG = ButterFilter(EEG, order=2,cutoff=0.3,fs = 500,model = 'highpass')
EEG = ButterFilter(EEG, order=2,cutoff=45,fs = 250,model = 'lowpass')
plt.plot(x1,EEG)
plt.show()
# #
# # plt.subplot(212)
# # plt.plot(x2,ACC)
#
# stft_specgram(EEG, 250,co = 0.005)