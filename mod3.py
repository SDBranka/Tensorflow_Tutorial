import matplotlib.pyplot as plt
import numpy as np

# # basic show graph matplotlib
# # X axis parameter:
# xaxis = np.array([2, 8])

# # Y axis parameter:
# yaxis = np.array([4, 9])

# plt.plot(xaxis, yaxis)
# plt.show()



x = [1, 2, 2.5, 3, 4]
y = [1, 4, 7, 9, 15]
plt.plot(x, y, "ro")
plt.axis([0, 6, 0, 20])
# plt.show()