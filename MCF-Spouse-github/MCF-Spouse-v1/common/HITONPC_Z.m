function   [pc,ntest,sepset,time]=HITONPC_Z(data,target,alpha, m,p, maxK)

start=tic;


cpc=[];
pc=[];


dep=zeros(1,p);
sepset=cell(1,p);
ntest=0;


for i=1:p
    if i~=target

        %[CI,dep(i)]=my_fisherz_test(data,i,target,[],m,alpha);#（原）
        %[CI,dep(i)]=cmi(target,i,[],alpha);
        print(CI)

        ntest=ntest+1;
        if CI==1 || isnan(dep(i))
            continue;
        end
        if CI==0
            cpc=[cpc,i];
        end
    end
end

%cpc

[~,var_index]=sort(dep(cpc),'descend');


for i=1:length(cpc)
    

    Y=cpc(var_index(i));
    pc=[pc Y];
    
    
    pc_length=length(pc);
    pc_tmp=pc;
    
    last_break_flag=0;
    
    for j=pc_length:-1:1
        
        X=pc(j);
        CanPC=mysetdiff(pc_tmp, X);
        
        break_flag=0;
        cutSetSize = 0;
                
        while length(CanPC) >= cutSetSize &&cutSetSize<=maxK
            
            SS = subsets1(CanPC, cutSetSize);    % all subsets of size cutSetSize
            
            for si=1:length(SS)
                Z = SS{si};
                
                
                if X~=Y             % idea from MMHC by YU
                    if isempty(find(Z==Y, 1))
                        continue;
                    end
                end   
                
                
                ntest=ntest+1;
                %CI=my_fisherz_test(data,X,target,Z,m,alpha);
                #CI=cmi(target,X,Z,alpha);

                if CI==1
                    pc_tmp=CanPC;
                    sepset{1,X}=Z;
                    break_flag=1;
                    
                    if X==Y
                        last_break_flag=1;
                    end
                    
                    break;
                end
            end
            
            if( break_flag==1 )
                break;
            end
            if( last_break_flag==1 )
                break;
            end
            
            cutSetSize = cutSetSize + 1;
        end

        if( last_break_flag==1 )
            break;
        end
    end
    
    pc=pc_tmp;
end

% pc = sort(pc);

time=toc(start);





