clear
addpath('lib');
addpath('algorithm');
load('dataset\Yale_32x32.mat');

X=fea';y=gnd;


num=50;



for j=3:6
    sum_accuracy_DPRC = 0; sum_accuracy_src = 0; 
    for k=1:num
        fprintf('         number of samples per class:%d---the %d times\n',j,k)
        load(['dataset\Yalesplit\',num2str(j),'train\',num2str(k),'.mat']);
        testIdx = int16(testIdx); trainIdx = int16(trainIdx);
        
        TestSet.X=X(:,testIdx);TestSet.y=y(testIdx)';
        TrainSet.X=X(:,trainIdx);TrainSet.y=y(trainIdx)';
        train_num=length(TrainSet.y);
        test_num=length(TestSet.y);
        class_num=length(unique(TrainSet.y));
        
        
        
        accuracy_DPRC =  DPRC(TrainSet, TestSet, train_num, test_num, class_num);
        accuracy_src =  src(TrainSet, TestSet, train_num, test_num, class_num);
        
        
        
        
        
        fprintf('The accuracy of DPRC  is %8.5f      \n', accuracy_DPRC);
        fprintf('The accuracy of SRC   is %8.5f      \n', accuracy_src);
        
        
        sum_accuracy_DPRC = sum_accuracy_DPRC + accuracy_DPRC;
        sum_accuracy_src = sum_accuracy_src + accuracy_src;
        
        
    end
    fprintf('\n\n*******number of samples per class:%d *******\n',j)
    fprintf('The averaged accuracy of DPRC  is %8.5f      \n', sum_accuracy_DPRC/num);
    fprintf('The averaged accuracy of SRC   is %8.5f      \n', sum_accuracy_src/num);
    
    
end
