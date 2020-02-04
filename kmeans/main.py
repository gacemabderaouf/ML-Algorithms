from random import randint
import kmeansAPI as km
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

def loadData(proportion) :
    data = pd.DataFrame(pd.read_csv("dataset.txt",names=range(0,5)))
    dataLimit = int(data.shape[0] * proportion)
    dataSet = np.random.permutation(data)
    return dataSet[:dataLimit], dataSet[dataLimit:]

proportion = 2./3.
trainingDataSet, testDataSet = loadData(proportion)
classesNumber = 3
dataDim = len(trainingDataSet[0])-1 #without the last attribute (iris type)

classes, centers, errors = km.kmeans(trainingDataSet, dataDim, classesNumber)
colors = ["r","b","g","y","black"] 

for cl in classes :
    elements = pd.DataFrame(cl["elements"],index=range(0,len(cl["elements"])),columns=range(0,len(cl["elements"][0])))
    plt.scatter(elements.loc[:,0],elements.loc[:,1],c=colors[cl["classNumber"]])
    plt.scatter(centers[cl["classNumber"]][0],centers[cl["classNumber"]][1],s=150,c=colors[cl["classNumber"]],marker="+")

print("Training Data Set Errors :")
print(errors)

errors = pd.DataFrame(np.zeros((classesNumber,classesNumber)), columns=pd.DataFrame(trainingDataSet)[4].unique(), index=pd.DataFrame(trainingDataSet)[4].unique())

for elem in testDataSet :
    affectation = km.nearestElements(centers, elem[:-1])
    errors[elem[4]][classes[affectation]["name"]] += 1

print(errors)
plt.show()