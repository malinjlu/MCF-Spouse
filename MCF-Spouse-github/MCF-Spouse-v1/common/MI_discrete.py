import numpy as np
from scipy import stats
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

def cmi(train_label, feature1, feature2):

    train_label = train_label.astype(int)
    feature1 = feature1.astype(int)
    feature2 = feature2.astype(int)

    # Calculate joint entropy
    joint_entropy = stats.entropy(np.ravel_multi_index((train_label, feature1, feature2), (train_label.max()+1, feature1.max()+1, feature2.max()+1)))

    # Calculate conditional entropy
    conditional_entropy = stats.entropy(np.ravel_multi_index((train_label, feature1), (train_label.max()+1, feature1.max()+1)))

    # Calculate conditional mutual information
    iXYZ = joint_entropy - conditional_entropy

    return iXYZ

def mi(firstVector, secondVector):
    iXY = mutual_info_score(firstVector, secondVector)
    return iXY


def MI_G2(Data, target, alpha, p):
    # 初始化测试计数器和候选 MB 列表
    ntest = 0
    mb = []

    # 提取训练数据和特征矩阵
    train_data = Data[:, np.setdiff1d(range(p), target)]
    featureMatrix = train_data

    # 提取训练标签和类别列
    train_label = Data[:, target]
    classColumn = train_label

    # 计算特征数量和初始化类别得分和可见性列表
    numFeatures = featureMatrix.shape[1]
    classScore = np.zeros(numFeatures)
    vis = np.zeros(p)

    # 计算每个特征与类别的 SU 得分
    for i in range(numFeatures):
        ntest += 1
        classScore[i] = SU(featureMatrix[:, i], classColumn)

    # 对得分进行排序，并根据阈值筛选保留候选特征
    classScore, indexScore = zip(*sorted(zip(classScore, range(numFeatures)), reverse=True))
    threshold = 0.05
    t = [index for index, score in zip(indexScore, classScore) if score < threshold]
    u = [score for score in classScore if score < threshold]
    indexScore = [index for index, score in zip(indexScore, classScore) if score > threshold]
    classScore = [score for score in classScore if score > threshold]

    # 如果没有候选特征，返回空的候选 MB 列表
    if len(indexScore) == 0:
        selectedFeatures = []
        return mb, ntest, 0

    # 初始化循环变量和临时 MB 列表
    curPosition = 0
    mii = -1

    # 根据 SU 得分筛选保留候选特征
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

    # 将保留的候选特征添加到 MB 中，并更新可见性列表
    selectedFeatures = indexScore
    pc = selectedFeatures
    last = selectedFeatures[-1]
    mb = yingshe2(pc, target)
    for feature in selectedFeatures:
        vis[feature] = 1

    # 迭代检查候选特征是否满足条件，并更新 MB 列表和可见性列表
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
                # 计算条件互信息和互信息
                iXYZ = cmi(train_label, featureMatrix[:, t[a]], featureMatrix[:, selectedFeatures[i]])
                iXY0 = mi(train_label, featureMatrix[:, t[a]])
                ntest += 2
                if t[a] >= target:
                    ttt = t[a] + 1
                else:
                    ttt = t[a]
                # 判断是否满足条件 I(X;Y|Z) > I(X;Y)
                if iXYZ > iXY0:
                    mb_tmp.append(t[a])
                    mb.append(ttt)
                    vis[t[a]] = 1
                if t[a] in indexScore:
                    # 如果不满足条件，从候选 MB 集合中移除特征 Y
                    indexScore.remove(t[a])
                    classScore.remove(u[a])
            a += 1

    # 返回候选 MB 列表、测试计数器和时间
    time = 0
    return mb, ntest, time



def SU(firstVector, secondVector):
    hX = h(firstVector)
    hY = h(secondVector)
    iXY = mi(firstVector, secondVector)
    score = (2 * iXY) / (hX + hY)
    return score


def h(vector):
    _, counts = np.unique(vector, return_counts=True)
    probabilities = counts / len(vector)
    return entropy(probabilities, base=2)


def yingshe2(pc, target):
    pc = [p if p < target else p + 1 for p in pc]
    return pc


from scipy.stats import entropy


def mutual_information(firstVector, secondVector):
    """
    Computes the mutual information between two vectors.

    Args:
        firstVector (numpy.ndarray): The first vector.
        secondVector (numpy.ndarray): The second vector.

    Returns:
        mi (float): The mutual information.
    """
    joint_entropy = entropy(np.vstack((firstVector, secondVector)))
    marginal_entropy_x = entropy(firstVector)
    marginal_entropy_y = entropy(secondVector)
    mi1 = marginal_entropy_x + marginal_entropy_y - joint_entropy
    return mi1


def conditional_mutual_information(data_matrix, x, y, s):
    """
    Computes the conditional mutual information between x and y given s.

    Args:
        data_matrix (numpy.ndarray): The data matrix.
        x (int): The first node.
        y (int): The second node.
        s (list): The set of conditioning nodes.

    Returns:
        cmi (float): The conditional mutual information.
    """
    # Concatenate x and s for the first variable and y and s for the second variable
    first_variable = np.concatenate((data_matrix[:, x].reshape(-1, 1), data_matrix[:, s]), axis=1)
    second_variable = np.concatenate((data_matrix[:, y].reshape(-1, 1), data_matrix[:, s]), axis=1)

    # Compute the conditional mutual information
    cmi1 = mutual_information(first_variable, second_variable)
    return cmi1


def mutual_information_test(data, x, y, s):
    """
    Performs a test of mutual information for conditional independence.

    Args:
        data (numpy.ndarray): The data matrix.
        x (int): The index of the first node.
        y (int): The index of the second node.
        s (list): The indices of the set of conditioning nodes.
        alpha (float): The significance level.

    Returns:
        p_value (float): The p-value of the test.
        dependency_measure (float): The measure of dependency.
    """
    # 计算 x 和 y 的互信息
    mi_xy = mi(data[:, x], data[:, y])

    # 如果 s 为空，则直接返回互信息作为依赖度量，且 p 值为 1
    if not s:
        return 1, mi_xy

    # 计算在给定 s 的条件下 x 和 y 的条件互信息
    mi_xys = conditional_mutual_information(data, x, y, s)

    # 根据互信息计算 p 值
    p_value = 1 if (mi_xy >= mi_xys).any() else 0

    return p_value, abs(mi_xy - mi_xys)


'''
def conditional_independence_test_MI(data_matrix, x, y, s, alpha):
    """
    Conducts conditional independence test using mutual information.

    Args:
        data_matrix (numpy.ndarray): The data matrix.
        x (int): The first node.
        y (int): The second node.
        s (list): The set of conditioning nodes.
        alpha (float): The significance level.

    Returns:
        p_value (float): The p-value of the test.
        dependency_measure (float): The measure of dependency.
    """
    # 计算给定节点 x 和 y 的互信息
    mi_xy = mi(data_matrix[:, x], data_matrix[:, y])

    # 如果条件节点集合为空，则返回 x 和 y 的互信息作为依赖度量，且 p 值为 1
    if not s:
        return 1, mi_xy

    # 否则，计算给定节点 x 和 y 在给定条件节点集合 s 下的条件互信息
    mi_xys = conditional_mutual_information(data_matrix, x, y, s)

    # 根据互信息计算 p 值
    p_value = mutual_information_test(mi_xy, mi_xys, alpha)

    return p_value, mi_xys

def conditional_mutual_information(data_matrix, x, y, s):
    """
    Computes the conditional mutual information between x and y given s.

    Args:
        data_matrix (numpy.ndarray): The data matrix.
        x (int): The first node.
        y (int): The second node.
        s (list): The set of conditioning nodes.

    Returns:
        mi_xys (float): The conditional mutual information.
    """
    # 将数据矩阵的列作为参数传递给 mi 函数，而不是整个数据矩阵
    # 转换为一维数组
    mi_xys = mi(np.ravel(data_matrix[:, [x] + s]), np.ravel(data_matrix[:, [y] + s]))
    return mi_xys

def mutual_information_test(mi_xy, mi_xys, alpha):
    """
    Conducts a test of mutual information for conditional independence.

    Args:
        mi_xy (float): The mutual information between x and y.
        mi_xys (float): The conditional mutual information between x and y given s.
        alpha (float): The significance level.

    Returns:
        p_value (float): The p-value of the test.
    """
    # 根据互信息计算 p 值
    p_value = 1 if mi_xy == 0 else min(1, mi_xys / mi_xy)
    return p_value
'''