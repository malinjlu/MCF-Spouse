function [pc] = getpc_new_g2(data,target,alpha)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
data = data +1; 
[m,p] = size(data);
ns=max(data);
pc = HITONPC_G2(data,target,alpha,ns,p,3);
%pc = HITONMB_G2(data,target,alpha,ns,p,3);
%pc = PAG_G2(data,target,alpha,ns,p,3);
%pc = HITONPC_chi(data,target,alpha,ns,p,3);
for i = 1:length(pc)
    pc(i) = pc(i)-1;
end




