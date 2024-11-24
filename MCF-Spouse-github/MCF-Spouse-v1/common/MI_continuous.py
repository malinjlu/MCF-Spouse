from sklearn.feature_selection import mutual_info_regression
from scipy.stats import entropy
import numpy as np

def mi(firstVector, secondVector):
    iXY = mutual_info_regression(firstVector.reshape(-1, 1), secondVector)
    return iXY

def h(vector):
    _, bins = np.histogram(vector, bins='auto')
    probabilities, _ = np.histogram(vector, bins=bins, density=True)
    return entropy(probabilities, base=2)

def SU(firstVector, secondVector):
    hX = h(firstVector)
    hY = h(secondVector)
    iXY = mi(firstVector, secondVector)
    score = (2 * iXY) / (hX + hY)
    return score

def CFS_MI_Z(Data, target, alpha, p):
    ntest = 0
    mb = []

    train_data = Data[:, np.setdiff1d(range(p), target)]
    featureMatrix = train_data

    train_label = Data[:, target]
    classColumn = train_label

    numFeatures = featureMatrix.shape[1]
    classScore = np.zeros(numFeatures)
    vis = np.zeros(p)

    for i in range(numFeatures):
        ntest += 1
        classScore[i] = SU(featureMatrix[:, i], classColumn)

    classScore, indexScore = zip(*sorted(zip(classScore, range(numFeatures)), reverse=True))

    threshold = 0.05
    th3 = 0.15
    t = [index for index, score in zip(indexScore, classScore) if score < threshold]
    u = [score for score in classScore if score < threshold]

    indexScore = [index for index, score in zip(indexScore, classScore) if score > threshold]
    classScore = [score for score in classScore if score > threshold]

    if len(indexScore) == 0:
        selectedFeatures = []
        return mb, ntest, 0

    curPosition = 0
    mii = -1
    while curPosition < len(indexScore):
        mb_tmp = []
        j = curPosition + 1
        curFeature = indexScore[curPosition]
        while j < len(indexScore):
            scoreij = SU(featureMatrix[:, curFeature], featureMatrix[:, indexScore[j]])
            ntest += 1
            if scoreij > classScore[j]:
                indexScore.pop(j)
                classScore.pop(j)
            else:
                j += 1
        curPosition += 1

    selectedFeatures = indexScore
    pc = selectedFeatures
    last = selectedFeatures[-1]
    mb = yingshe2(pc, target)

    for feature in selectedFeatures:
        vis[feature] = 1

    len1 = len(selectedFeatures)
    for i in range(len1):
        mb_tmp = []
        a = selectedFeatures.index(last) + 1
        len2 = len(t)
        while a < len2:
            if vis[t[a]] == 1:
                a += 1
                continue
            scoreij = SU(featureMatrix[:, selectedFeatures[i]], featureMatrix[:, t[a]])
            ntest += 1
            if scoreij > u[a] + 0.13:
                iXYZ = cmi(train_label, featureMatrix[:, t[a]], featureMatrix[:, selectedFeatures[i]])
                iXY0 = mi(train_label, featureMatrix[:, t[a]])
                ntest += 2
                if t[a] >= target:
                    ttt = t[a] + 1
                else:
                    ttt = t[a]
                if iXYZ.all() > iXY0:
                    mb_tmp.append(t[a])
                    mb.append(ttt)
                    vis[t[a]] = 1
            a += 1

    time = 0
    return mb, ntest, time

def cmi(train_label, feature1, feature2):
    # Calculate joint entropy
    joint_entropy = entropy(np.column_stack((train_label, feature1, feature2)).T)

    # Calculate conditional entropy
    conditional_entropy = entropy(np.column_stack((train_label, feature1)).T)

    # Calculate conditional mutual information
    iXYZ = joint_entropy - conditional_entropy

    return iXYZ

def yingshe2(pc, target):
    pc = [p if p < target else p + 1 for p in pc]
    return pc
