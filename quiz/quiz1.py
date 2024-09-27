import numpy as np

# Q2
x  = np.array([1,1,2,2,3,3])
y = np.array([1,2,2,3,2,3])
up = 0
down = 0
for i in range(len(x)):
    up += x[i]*y[i]
    down += x[i]**2
w = up/down
print('q2:')
print('w:', w)

loss1 = 1/2*(y[0]-w*x[0])**2
loss2 = 1/2*(y[1]-w*x[1])**2
print(loss1, loss2)

x3 = np.array([5,0,8])
y3 = np.array([10,-7,21])
up3= 0
down3 = 0
for i in range(len(x3)):
    up3 += x3[i]*y3[i]
    down3 += x3[i]**2
w3 = up3/down3
print('w3: ', w3)