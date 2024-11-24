function [pc] = getpc_new_z(data,target,alpha)
[m,p] = size(data);
%pc = pc_z(data,target,alpha);
pc = HITONPC_Z(data,target,alpha,m,p,3);
for i = 1:length(pc)
    pc(i) = pc(i)-1;
end