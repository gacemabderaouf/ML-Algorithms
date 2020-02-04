from random import randint
import numpy as np
import pandas as pd



def initCenters(dataSet, classesNumber) :
    toReturn = []
    for _ in range(0, classesNumber) :
        toReturn.append(dataSet[randint(0, len(dataSet)-1)][:-1])
    return toReturn

def nearestElements(centers, element) :
    def distance(x, y) :
        return np.sum(np.power((np.array(x) - y), 2))
    distances = list(map(lambda x : distance(x , element), centers))
    return np.argmin(distances)


def kmeans(dataSet, dataDim, classesNumber) :
    centers = initCenters(dataSet, classesNumber)
    classes = []
    classesCenters = np.array([[0.0] * dataDim] * classesNumber)
    for i in range(0, classesNumber) :
        classes.append({"classNumber" : i, "elements" : [], "name" : ""})

    for element in dataSet :
        affectation = nearestElements(centers, element[:-1])
        classes[affectation]["elements"].append(list(element))
        classesCenters[affectation] = classesCenters[affectation] + element[:-1]
    
    for i in range(0, classesNumber) :
        centers[i] = classesCenters[i] / float(len(classes[i]["elements"]))

    dirtyBit = True
    while dirtyBit :
        dirtyBit = False
        for cl in classes :
            for elem in cl["elements"] :
                affectation = nearestElements(centers, elem[:-1])
                if affectation != cl["classNumber"] :
                    dirtyBit = True
                    classesCenters[affectation] = classesCenters[affectation] + elem[:-1]
                    classesCenters[cl["classNumber"]] = classesCenters[cl["classNumber"]] - elem[:-1]
                    classes[affectation]["elements"].append(list(elem))
                    classes[cl["classNumber"]]["elements"].remove(elem)
        if dirtyBit :
            for i in range(0, classesNumber) :
                centers[i] = classesCenters[i] / float(len(classes[i]["elements"]))
    
    errors = pd.DataFrame(np.zeros((classesNumber,classesNumber)), columns=pd.DataFrame(dataSet)[4].unique(), index=pd.DataFrame(dataSet)[4].unique())
    
    for cl in classes :
        classification = pd.DataFrame(cl["elements"])[4]
        cl["name"] = (classification.value_counts().index[0])
        for elem in cl["elements"] :
            errors[elem[4]][cl["name"]] += 1 
        

    return classes, centers, errors

