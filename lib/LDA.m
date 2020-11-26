function [ projMatrix ] = LDA( data, label, d )
% LDA implement linear discriminant analysis to discriminant multivarite
% class of data
% Usage:
% [w] = LDA(data, label)
% Inputs:
%        -data: data points, n-by-m matrix. Each colum represent a type of
%        feature, each row is a sample;
%        -label: class label of data, n-by-1 matrix, each row represents 
%        the label of corresponding data;
%        -class: elements in labels, each element represents a class of
%        data;
%        -d: data dimension after projection, d must be smaller than
%        dimension of origin data, default d=1
% Outputs:
%         -W: Projection matrix, the first colum of W is the projection
%         vector that can maximize the discrimination;
%         -D: eigvalues, diagonal matrix;
%         -Gmd: struct contains mean value and convariance matrix of gaussian
%         distribution of projected data for each class
%
% @Author Hammer Zhang
% @Time: 2016.7.23, 9:57
% ========================================================================


class = unique(label);
cn = length(class);
dm = size(data,2);

if d >= dm
    error('Projection dimension must be smaller than origin data dimension...\n');
end

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

% train the projection matrix
     [W,D]    = eig(Sb,Sw);
     [~,I]    = sort(diag(D),'descend');
projMatrix    = W(:,I(1:d));






end