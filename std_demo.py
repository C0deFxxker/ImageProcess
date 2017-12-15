# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np

minx, maxx = -100, 100

X = np.arange(minx, maxx)
Y = np.arange(minx, maxx)

plt.plot(X, Y)
plt.show()