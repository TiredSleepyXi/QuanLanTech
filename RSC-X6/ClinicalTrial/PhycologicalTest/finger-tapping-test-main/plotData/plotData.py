import matplotlib.pyplot as plt


dados = #table vinda do arquivo fingerTappingtest.py


dado = [[e1/3, e2] for e1, e2 in dados]
plt.plot(*zip(*dados))
# plt.scatter(*zip(*dados))
plt.title('Finger Tapping Test')
plt.xlabel('time (s)')
plt.ylabel('Angle')
plt.show()