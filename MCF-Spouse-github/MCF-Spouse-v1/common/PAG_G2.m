function [pc, ntest, sepset, time] = PAG_G2(Data, target, alpha, ns, p, k)
    start = tic;
    cpc = [];
    pc = [];
    dep = zeros(1, p);
    sepset = cell(1, p);
    ntest = 0;

    % 遍历所有变量，寻找 PC 集合
    for i = 1:p
        if i ~= target
            % 计算变量 i 与目标变量 target 之间的条件独立性
            [pval, dep(i)] = my_g2test(i, target, [], Data, ns);

            % 根据显著性水平 alpha 判断条件独立性
            if isnan(pval)
                CI = 1;
            else
                CI = (pval > alpha);
            end

            % 如果条件独立，则将变量 i 加入到 cpc 集合中
            if CI
                continue;
            else
                cpc = [cpc, i];
            end
        end
    end

    % 根据 dep 数组对 cpc 集合进行排序，依次构建 PC 集合
    [dep_sort, var_index] = sort(dep(cpc), 'descend');

    for i = 1:length(cpc)
        Y = cpc(var_index(i));
        pc = [pc, Y];
        pc_length = length(pc);
        pc_tmp = pc;
        last_break_flag = 0;

        % 从 PC 集合中逐步剔除变量，构建部分祖先图或最大祖先图
        for j = pc_length:-1:1
            X = pc(j);
            CanPC = setdiff(pc_tmp, X);
            break_flag = 0;
            cutSetSize = 0;

            while length(CanPC) >= cutSetSize && cutSetSize <= k
                SS = subsets1(CanPC, cutSetSize); % 获取所有大小为 cutSetSize 的子集

                for si = 1:length(SS)
                    Z = SS{si};

                    % 如果当前节点 X 与目标节点 target 之间条件独立，则记录分离集合
                    [pval] = my_g2test(X, target, Z, Data, ns);
                    if isnan(pval)
                        CI = 0;
                    else
                        CI = (pval > alpha);
                    end

                    if CI
                        pc_tmp = CanPC;
                        sepset{1, X} = Z; % 记录分离集合
                        break_flag = 1;

                        if X == Y
                            last_break_flag = 1;
                        end

                        break;
                    end
                end

                if break_flag
                    break;
                end
                if last_break_flag
                    break;
                end

                cutSetSize = cutSetSize + 1;
            end

            if last_break_flag
                break;
            end
        end

        pc = pc_tmp;
    end

    time = toc(start);
end
