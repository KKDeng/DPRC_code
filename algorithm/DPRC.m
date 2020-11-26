function accuracy = DPRC(TrainSet, TestSet, train_num, test_num, class_num)
% Discriminative projection and representation-based classification framework 
%
% Inputs:
%       TrainSet            train sets of size dxn, where d is dimension and n is number of sets 
%       TestSet             test sets of size dxn, where d is dimension and n is number of sets
%       train_num           the size of train sets
%       test_num            the size of test sets
%       class_num           the size of class
%       lambda              regularization parameter of RC methods
%       options             options
%
% Output:
%       accuracy            classification accurary
%

% parameters for approximated augmented Lagrangian method
options.rho = 10; options.epso = 0.9; options.tau = 0.99; 
options.mu = 1.1;options.decrease= 0.9;
options.max_iter = 100; options.inn_max_iter = 50; 
options.error = 1e-6; options.reduce_dimension=200;

%% learning a discriminative projection matrix P
fprintf('discriminative projection...\n')
%[H,P] = PADMM(TrainSet,options); partial ADMM
[H,P] = AALM(TrainSet,options); %  Approximated augmented Lagrangian method


%% project training samples into discrimination space
TrainSet.X = P'*TrainSet.X; TestSet.X = P'*TestSet.X;


%% classification with various RC methods
fprintf('classify...\n')
%accuracy = esrc(TrainSet, TestSet, train_num, test_num, class_num, 0.1, options);
accuracy = src(TrainSet, TestSet, train_num, test_num, class_num);




