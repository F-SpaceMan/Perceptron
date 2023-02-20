import numpy as np
import random as rd
import pandas as pd


def fatia_arr(array):
    
    newArray1 = []
    newArray2 = []

    for i in range(round(len(array)*0.7)):
        newArray1.append(array[i])

    for j in range(round(len(array)*(0.3)-1), -1, -1):
        newArray2.append(array[j])
    
    return [newArray1, newArray2]



def threshold(u):
    y = [0, 0, 0]

    for o in range(len(u)):
        if u[o] >= 0:
            y[o] = 1
    return y
    

def sigmoid(u):
    y = [0, 0, 0]

    maxi = u[0]
    maxIndex = 0
    for o in range(len(u)):
        if u[o] > maxi:
            maxIndex = o

    y[maxIndex] = 1
    return y


class Perceptron:
    def __init__(self):
        self.XTrain = []
        self.DTrain = []
        
        self.XTest = []
        self.DTest = []
            
    def perceptron_train(self, XD, W, b, taxa_treino, max_it):
        
        rd.shuffle(XD)

        for i in range(len(XD)):
            for j in range(6):
                XD[i][j] = float(XD[i][j])

        arrTrainTest = fatia_arr(XD)
        arrTrain = arrTrainTest[0]
        arrTest = arrTrainTest[1]

        for linha in arrTrain:
            self.XTrain.append(linha[:6])
            self.DTrain.append(linha[6])
        for linha in arrTest:
            self.XTest.append(linha[:6])
            self.DTest.append(linha[6])
    
        t = 0
        E = 1
        Erro_Epocas = []
        
        while (t < max_it) & (E>0):
            E = 0

            y = []
            e = []

            for i in range(len(self.XTrain)):
                x = np.array([self.XTrain[i]])

                u = np.dot(W, x.T) + b.T

                y.append(activationFunction(u))

    #             e = D[i] - y[i]
                e.append((np.array(self.DTrain[i]) - np.array(y[i])).tolist())


    #             print('erro ', e[i],'\n')

                W = W + (taxa_treino * (np.dot(np.array([e[i]]).T, x)))
                b = b + (taxa_treino * np.array([e[i]]))

                ee = np.dot(np.array([e[i]]),np.array([e[i]]).T)

    #             E é o erro por época, E = sum(e[i]^2)
                E += ee

                i += 1

            Erro_Epocas.append(E)
            t += 1


        for ind in range(len(Erro_Epocas)):
            Erro_Epocas[ind] = Erro_Epocas[ind][0][0]


        errosDf = pd.DataFrame(Erro_Epocas, columns=['Erro Treino'])

        errosDf.plot()
        
        self.W = W
        self.b = b
    
    def perceptron_test(self):

        E = []
        hit = 0

        for i in range(len(self.XTest)):
            x = np.array([self.XTest[i]])

            u = np.dot(self.W, x.T) + self.b.T

            y = activationFunction(u)
            e = (np.array(self.DTest[i]) - np.array(y)).tolist()

            ee = np.dot(np.array([e]),np.array([e]).T)

            if(ee[0][0] == 0):
                hit += 1

            i += 1

        hitRate = (hit/len(self.XTest)) * 100 
        print(f'Hit Rate:\n{hitRate}')

        

        
        
DTrad = {'DH':[0,0,1], 'SL':[0,1,0], 'NO':[1,0,0]}  

XTrain = []
DTrain = []
XTest = []
DTest = []

linhas = []

with open("column_3C.dat") as file:
    for line in file.readlines():
        aux = line.split()[:6]
        aux.append(DTrad[line.split()[6]])
        
        linhas.append(aux)


W = np.array([
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
])

b = np.array([[0.1,0.1,0.1]])

taxa_treino = 0.8
max_it = 50


print('Please, choose activation function: \n 1 for step \n 2 for sigmoid')
userInput = input()


if userInput == 1:
    activationFunction = threshold
elif userInput == 2:
    activationFunction = sigmoid


perceptrao1 = Perceptron()

perceptrao1.perceptron_train(linhas, W, b, taxa_treino, max_it)
perceptrao1.perceptron_test()
