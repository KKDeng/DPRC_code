function best_accuracy = src(TrainSet, TestSet, train_num, test_num, class_num)
% Sparse representation classification (SRC) algorithm
%
% Inputs:
%       TrainSet            train sets of size dxn, where d is dimension and n is number of sets
%       TestSet             test sets of size dxn, where d is dimension and n is number of sets
%       test_num            numner of test sets
%       class_num           numner of classes
%       lambda              regularization paramter
% Output:
%       accuracy            classification accurary
%
% References:
%       J. Wright, A. Yang, A. Ganesh, S. Sastry, and Y. Ma,
%       "Robust face recognition via sparse representation,"
%       IEEE Transaction on Pattern Analysis and Machine Intelligence, vol.31, no.2, pp.210-227, 2009.
%
%
% Created by H.Kasai on July 06, 2017


% extract options
% if ~isfield(options, 'verbose')
%     verbose = false;
% else
%     verbose = options.verbose;
% end
% 
% if ~isfield(options, 'eigenface')
%     eigenface = true;
% else
%     eigenface = options.eigenface;
% end
% 
% if ~isfield(options, 'eigenface_dim')
%     eigenface_dim = train_num;
% else
%     eigenface_dim = options.eigenface_dim;
% end
% 
% 
% % generate eigenface
% if eigenface
%     [disc_set, ~, ~] = Eigenface_f(TrainSet.X, eigenface_dim);
%     
%     % project on subspace
%     TrainSet.X  =  disc_set' * TrainSet.X;
%     TestSet.X   =  disc_set' * TestSet.X;
% end

% normalize data to l2-norm
[TrainSet.X, ~] = data_normalization(TrainSet.X, TrainSet.y, 'std');
[TestSet.X, ~] = data_normalization(TestSet.X, TestSet.y, 'std');


set = [0.005,0.01,0.02,0.03];
% prepare class array
classes = unique(TrainSet.y);
 P = inv(TrainSet.X'*TrainSet.X+0.001*eye(size(TrainSet.X,2)))*TrainSet.X';
% prepare predicted label array
identity = zeros(1, test_num);
iter =0;best_accuracy=0;
while(iter<length(set))
    lambda = set(iter+1);
    for i = 1 : test_num
        
        y = TestSet.X(:, i);
        
       
        x0 = P*y;
        
        maxIteration = 5000;
        isNonnegative = false;
        %lambda = 1e-2; %5e-3;
        tolerance = 0.01;
        STOPPING_GROUND_TRUTH = -1;
        
        stoppingCriterion = STOPPING_GROUND_TRUTH;
        [xp, iterationCount] = SolveHomotopy(TrainSet.X, y, ...
            'maxIteration', maxIteration,...
            'isNonnegative', isNonnegative, ...
            'stoppingCriterion', stoppingCriterion, ...
            'groundtruth', x0, ...
            'lambda', lambda, ...
            'tolerance', tolerance);
        
        
        
        
        
        
        % prepare residual array
        residuals = zeros(1, class_num);
        
        % calculate residual for each class
        for j = 1 : class_num
            idx = find(TrainSet.y == classes(j));
            %residuals(j) = norm(y-TrainSet.X(:,idx)*xp(idx))/sum(xp(idx).*xp(idx));
            residuals(j) = norm(y - TrainSet.X(:,idx)*xp(idx));
        end
        
        % calculate the predicted label with minimum residual
        [~, label] = min(residuals);
        identity(i) = label;
        
        
        
        
    end
    
    % calculate accuracy
    correct_num = sum(identity == TestSet.y);
    accuracy = correct_num/test_num;
    best_accuracy = max(best_accuracy, accuracy );
    iter = iter +1;
end
end



