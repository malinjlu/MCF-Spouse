function iXY = mi(firstVector, secondVector)
    % 导入 Python 库
    py.importlib.import_module('sklearn.feature_selection');
    py.importlib.import_module('sklearn.feature_selection.mutual_info_');

    % 调用 Python 函数
    iXY = py.sklearn.feature_selection.mutual_info_regression(firstVector, secondVector);
    iXY = double(iXY);
end

function ent = h(vector)
    % 计算直方图
    [counts, bins] = histcounts(vector, 'BinMethod', 'auto');

    % 计算概率分布
    probabilities = counts / sum(counts);

    % 计算熵
    ent = entropy(probabilities);
end

function result = cmi(train_label, feature1, feature2, alpha)
    % 导入 Python 库
    py.importlib.import_module('scipy.stats');

    % 将 MATLAB 数组转换为 Python 列表
    train_label = train_label(:);
    feature1 = feature1(:);
    feature2 = feature2(:);

    % 调用 Python 函数
    joint_entropy = py.scipy.stats.entropy({train_label, feature1, feature2});
    conditional_entropy = py.scipy.stats.entropy({train_label, feature1});
    iXYZ = double(joint_entropy - conditional_entropy);

    % 判断是否大于 alpha
    if iXYZ > alpha
        result = 0;
    else
        result = 1;
    end
end