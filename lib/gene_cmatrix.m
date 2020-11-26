function [M,Sb,Sw] = gene_cmatrix(data,label)

class = unique(label);
cn = length(class);
dm = size(data,2);



% compute total mean values of data
u = mean(data,1);
% compute mean value of each class and within-class variation
Sw = zeros(dm,dm);
Sb = zeros(dm,dm);
for ci=1:cn
    id   = find(label == class(ci));
    ni   = length(id);
    
    % mean value and variation of class ci
    ui   = mean(data(id,:),1);
    var  = data(id,:) - repmat(ui,ni,1);
    Sw   = Sw + var' * var;                  % within-class variation    
    Sb   = Sb + ni*(ui - u)' * (ui - u);     % between-class variation
end

M = 0.5*Sw-1*Sb;
