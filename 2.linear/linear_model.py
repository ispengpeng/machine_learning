import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tqdm
import matplotlib.pyplot as plt


df = pd.read_csv('SOCR-HeightWeight.csv')

#获取数据长度
length = len(df)

# 读取前100个数据
X = df.iloc[range(200),1]
Y = df.iloc[range(200),2]

#将其分成70%和30%两部分，分别作为训练集和数据集
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=0,shuffle=False)

#计算均方误差值
def compute_loss (w,b):
    loss = 0
    for ii in range(len(x_train)):
        loss = loss + (y_train[ii] - (x_train[ii]*w+b))**2
    loss = loss / len(x_train)
    return loss


#对数据进行训练
def train():

    loss_init = 10000
    w_step = 0.01
    b_step = 1
    loss_array = []
    loss_all = []

    for epoch in tqdm.tqdm(np.arange(0,6,w_step)):
        for ii in np.arange(-170,-130,b_step):
            loss = compute_loss(epoch, ii)
            if(loss<loss_init):
                loss_init = loss
                loss_array.append(loss_init)
                w = epoch
                b = ii
            else:
                pass
        loss_all.append(loss)


    print(f"loss = {loss_init}, w = {w}, b = {b}")

    r_m, r_b = np.polyfit(x_train, y_train, 1)
    print(r_m,r_b)

    plt.subplot(1,3,1)
    plt.scatter(range(len(loss_all)), loss_all, color='blue', label='Data points')
    plt.xlabel('times')
    plt.ylabel('loss')
    plt.title('loss')

    plt.subplot(1,3,2)
    plt.scatter(range(len(loss_array)), loss_array, color='blue', label='Data points')
    plt.xlabel('times')
    plt.ylabel('loss')
    plt.title('loss')

    plt.subplot(1,3,3)
    plt.scatter(x_train, y_train, color='blue', label='Data points')
    plt.plot(x_train, r_m*x_train + r_b, color='red', label='real line')
    plt.plot(x_train, w*x_train + b, color='green', label='predict line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    return w,b

def test(w,b):
    error_array = (y_test - (x_test*w + b))
    accuracy = (abs(error_array - y_test)/y_test)

    plt.scatter(range(len(error_array)), accuracy, color='blue', label='Data points')
    plt.xlabel('times')
    plt.ylabel('error')

    plt.title('error')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train()
    test(4.23,-159)

