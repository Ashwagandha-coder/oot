import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv

import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import random

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Считывание данных с файла
def GetCSV(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        columns = next(reader)
        colmap = dict(zip(columns, range(len(columns))))

    return np.matrix(np.loadtxt(filename, delimiter=",", skiprows=1))

# Нормализация строчек
def normalizeRow(matrix, row):
    length = matrix.shape[1]
    min = np.amin(matrix[row])
    max = np.amax(matrix[row])
    
    for i in range(length):
        matrix[row,i] = (matrix[row,i] - min) / (max - min)

# Нормализация тренировочных и тестовых данных
def normalizeData(trainData, testData, row):
    _min = min(np.amin(trainData[row]), np.amin(testData[row]))
    _max = max(np.amax(trainData[row]), np.amax(testData[row]))
    
    for i in range(trainData.shape[1]):
        trainData[row,i] = (trainData[row,i] - _min) / (_max - _min)
    
    for i in range(testData.shape[1]):
        testData[row,i] = (testData[row,i] - _min) / (_max - _min)

# Генерация случайных весов
def getRandomWeights(inLayerSize, outLayerSize):
    return np.matrix(np.random.rand(outLayerSize, inLayerSize))

# Генерация случайных смещений
def getRandomOffsets(layerSize):
    return (np.random.rand(layerSize,1)-0.5) * 0.1

# Посчитать функцию по каждому элементу матрицы
def matrixElementWiseFunction(matrix, function):
    result = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            result[i, j] = function(matrix[i, j])
    return result


# Сместить матрицу (добавить к каждому эл-ту число)
def offsetMatrix(matrix, offset):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i, j] += offset[i, 0]


# Neural network
global functionsCache
functionsCache = None

# Сгенерировать случайную нейронку
def getRandomNetwork(x, y, layer_infos):
    layer_count = len(layer_infos)
    last_neuron_count = x.shape[0]
    weights = []
    offsets = []
    
    for layerInfo in layer_infos:
        weights.append(getRandomWeights(last_neuron_count, layerInfo[0]))
        offsets.append(getRandomOffsets(layerInfo[0]))
        last_neuron_count = layerInfo[0]
    
    weights.append(getRandomWeights(last_neuron_count, y.shape[0]))
    offsets.append(getRandomOffsets(y.shape[0]))

    return [weights, offsets]

# Для удобства работы с функциями            
def getFunctionsArray(layer_infos, lastActivationFunc):
    global functionsCache
    if functionsCache is not None:
        return functionsCache

    functions = []
    
    for layerInfo in layer_infos:
        functions.append(layerInfo[1])
    
    functions.append(lastActivationFunc)
    functionsCache = functions

    return functions

# Вычисление функции стоимости (среднеквадратичное отклонение)
def calculateCost(currentOutput, y, setSize):
    cost = 0
    #print(y.shape)
    for i in range(setSize):
        for j in range(y.shape[0]):
            cost += (currentOutput[j,i]-y[j,i])**2
    
    return cost / 2

# Прямой проход по нейронке
def feedForward(x, weights, offsets, layer_infos, lastActivationFunc):
    functions = getFunctionsArray(layer_infos, lastActivationFunc)
    layer_count = len(layer_infos)

    z = []
    a = []
    a.append(x)

    # Calculation
    currentOutput = x
    for i in range(layer_count + 1):
        #print(i)
        #print("O: " + str(offsets[i].shape))
        #print("W: " + str(weights[i].shape))
        #print("H: " + str(currentOutput.shape))
        
        currentOutput = np.matmul(weights[i], currentOutput)
        offsetMatrix(currentOutput, offsets[i])
        
        z.append(currentOutput.copy())
        
        currentOutput = matrixElementWiseFunction(currentOutput, functions[i][0])
        a.append(currentOutput.copy())
        
        #print("A: " + str(currentOutput.shape))
    
    return [currentOutput,a ,z]

# Обратный проход по нейронке (Обучение нейронки, воозвращает новые веса и смещения)
def backPropagate(currentOutput, x, y, z, a, layer_infos, lastActivationFunc,weights, offsets, mu = 0.01):
    functions = getFunctionsArray(layer_infos, lastActivationFunc)
    layer_count = len(layer_infos)
    deltas = []
    
    currentDelta = np.multiply((currentOutput - y), matrixElementWiseFunction(z[layer_count], functions[layer_count][1]))
    deltas.append(currentDelta)
    
    for l in reversed(range(layer_count)):
        currentDelta = np.multiply(np.matmul(weights[l + 1].transpose(), currentDelta), matrixElementWiseFunction(z[l],functions[l][1]))
        deltas.append(currentDelta.copy())
    
    newWeights = []
    newOffsets = []

    for i in range(layer_count + 1):
        deltaL = deltas[layer_count - i]
        newOffsets.append(offsets[i] - np.dot(mu, deltaL))
        temp = np.matmul(deltaL, a[i].transpose())
        newWeights.append(weights[i] - np.dot(mu, temp))

    return [newWeights, newOffsets]

# Сама программа
def main(layer_infos, lastActivationFunc):
    #Считывание данных
    trainFilename = '/kaggle/input/modern-computer-technologies-laboratory/train.csv'
    testFilename = '/kaggle/input/modern-computer-technologies-laboratory/test.csv'
    train_csv = GetCSV(trainFilename)
    test_csv = GetCSV(testFilename)

    def GetInOutData(csv_data, shuffle = False):
        size = int(csv_data.shape[0])

        x = np.zeros((2, size))
        y = np.zeros((2, size))
        
        indexArray = list(range(size))
        if(shuffle):
            random.shuffle(indexArray)

        for i in indexArray:
            x[0,i] = csv_data[i,2]
            x[1,i] = csv_data[i,1]
            
            y[0,i] = csv_data[i,4]
            y[1,i] = csv_data[i,3]

        return {'x' : x, 'y' : y}
    
    trainSetSize = int(train_csv.shape[0])
    testSetSize = int(test_csv.shape[0])
    
    trainData = GetInOutData(train_csv)
    x = trainData['x']
    y = trainData['y']
    
    testData = GetInOutData(test_csv, True)
    xTest = testData['x']
    yTest = testData['y']
    
    # Нормализация данных
    normalizeData(x, xTest, 0)
    normalizeData(x, xTest, 1)

    normalizeData(y, yTest, 0)
    normalizeData(y, yTest, 1)
    
    # Генерация нчальной нейронки
    randomNetwork = getRandomNetwork(x,y,layer_infos)
    weights = randomNetwork[0]
    offsets = randomNetwork[1]
    
    # График тестовых данных
    sns.lineplot(data = pd.DataFrame({"Set Y[0]" : yTest[0]}), palette=['r'])
    sns.lineplot(data = pd.DataFrame({"Set Y[1]" : yTest[1]}), palette=['g'])
    
    iterations = 20
    period = 1
    speed = 0.01
    
    for i in range(iterations * period + 1):
        if (i / period) % 1 == 0:
            testOutput = feedForward(xTest, weights, offsets, layer_infos, lastActivationFunc)
            print("\nTEST COST: " + str(calculateCost(testOutput[0], yTest, testSetSize)) + "\n")
            #sns.lineplot(data = pd.DataFrame({"Set" + str(i) : yTest[0], "Output" : testOutput[0][0]}))
            #sns.lineplot(data = pd.DataFrame({"Output" + str(i) : testOutput[0][0]}))
        
        # Обучение (берет входные и выходные данные одной строки и делает прямой и обратный проход)
        for j in range(x.shape[1]):
            xDash = np.matrix(x[:,j]).transpose()
            yDash = np.matrix(y[:,j]).transpose()

            feedOutput = feedForward(xDash, weights, offsets, layer_infos, lastActivationFunc)
            currentOutput = feedOutput[0]
            a = feedOutput[1]
            z = feedOutput[2]
            #print(str(i) + " Train cost: " + str(calculateCost(currentOutput, yDash, 1)))
            
            newNetwork = backPropagate(currentOutput, xDash, yDash, z, a, layer_infos, lastActivationFunc,weights, offsets, speed)
            weights = newNetwork[0]
            offsets = newNetwork[1]

    # График результатов обучения
    testOutput = feedForward(xTest, weights, offsets, layer_infos, lastActivationFunc)
    print("\nTEST COST: " + str(calculateCost(testOutput[0], yTest, testSetSize)) + "\n")
    #sns.lineplot(data = pd.DataFrame({"Set" + str(i) : yTest[0], "Output" : testOutput[0][0]}))
    sns.lineplot(data = pd.DataFrame({"Output [0]" : testOutput[0][0]}), palette=['b'])
    sns.lineplot(data = pd.DataFrame({"Output [1]" : testOutput[0][1]}), palette=['y'])


    
def ReLu(x):
    return x if x > 0 else 0

def ReLuDerivative(x):
    return 0 if x < 0 else 1

def Sigmoid(x):
    return 1/(1+np.exp(-x))

def SigmoidDerivative(x):
    return Sigmoid(x) * (1 - Sigmoid(x))
    
ReLuFunctions = [ReLu, ReLuDerivative]
SigmoidFunctions = [Sigmoid, SigmoidDerivative]

# Запуск программы, задаются размеры слоев и функции
main(layer_infos=[[10, ReLuFunctions],[15, ReLuFunctions],[10, ReLuFunctions]], lastActivationFunc = SigmoidFunctions)