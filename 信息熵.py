from math import log

def calcshannonent(dataset):
    numentries = len(dataset)
    labelcounts ={}
    for vect in dataset:
        currentlabel = vect[-1]
        if currentlabel not in labelcounts:
            labelcounts[currentlabel] = 0
        labelcounts[currentlabel] += 1
    shannonent = 0.0
    print(labelcounts)
    for key in labelcounts:
        prob = float(labelcounts[key])/numentries
        shannonent -= prob * log(prob,2)
    return shannonent

dataset = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]

# res = calcshannonent(dataset)
# print(res)

def splitdata(dataset,axis,value):
    retdataset = []
    for vect in dataset:
        if vect[axis] == value:
            reduced_vect = vect[:axis]
            reduced_vect.extend(vect[axis+1:])
            retdataset.append(reduced_vect)
    return retdataset

# res = splitdata(dataset,0,1)
# print(res)

def choosebest(dataset):
    numfeatures = len(dataset[0])-1
    baseent = calcshannonent(dataset)
    bestinfogain = 0.0
    bestfeature = -1
    for i in range(numfeatures):
        featList = [example[i] for example in dataset]
        uniquevals = set(featList)
        newent = 0.0
        for value in uniquevals:
            subdataset = splitdata(dataset,i,value)
            prob = len(subdataset)/float(len(dataset))
            newent += prob * calcshannonent(subdataset)
        infogain = baseent - newent
        if(infogain>bestinfogain):
            bestinfogain = infogain
            bestfeature = i
        return bestfeature,bestinfogain

res = choosebest(dataset)
print(res)

