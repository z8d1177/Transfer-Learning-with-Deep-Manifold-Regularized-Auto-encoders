global params;

global numM;                % input data feature dimensions  m
global numK;                                 % number of hidden units k
global numC;
global beta;
global numX;


%% ======================================================================
%  STEP 1: Load data
%
load('corel.mat');
numExamples = size(data, 2);

result =[];
iCnt = 1;
for sourceFlowerIndex = 1:4
    for sourceTrafficeIndex = 1:4
        if(sourceFlowerIndex == 1)
            trainData = [data(:,1:offset(1)) data(:,offset(4+sourceTrafficeIndex-1)+1:offset(4+sourceTrafficeIndex))];
            trainLabels = [ones(1,offset(1)) zeros(1,offset(4+sourceTrafficeIndex)-offset(4+sourceTrafficeIndex-1))] + 1;
            numX = [offset(1), offset(4+sourceTrafficeIndex)-offset(4+sourceTrafficeIndex-1)];
        else
            trainData = [data(:,offset(sourceFlowerIndex-1)+1:offset(sourceFlowerIndex)) data(:,offset(4+sourceTrafficeIndex-1)+1:offset(4+sourceTrafficeIndex))];
            trainLabels = [ones(1,offset(sourceFlowerIndex)-offset(sourceFlowerIndex-1)) zeros(1,offset(4+sourceTrafficeIndex)-offset(4+sourceTrafficeIndex-1))] + 1;
            numX = [offset(sourceFlowerIndex)-offset(sourceFlowerIndex-1), offset(4+sourceTrafficeIndex)-offset(4+sourceTrafficeIndex-1)];
        end;
        
        for targetFlowerIndex = 1:4
            if targetFlowerIndex == sourceFlowerIndex
                continue
            end
            for targetTrafficIndex = 1:4
                if targetTrafficIndex == sourceTrafficeIndex
                    continue
                end
                
                %% ======================================================================
                %  STEP 1: Load data
                if(targetFlowerIndex == 1)
                    testData = [data(:,1:offset(1)) data(:,offset(4+targetTrafficIndex-1)+1:offset(4+targetTrafficIndex))];
                    testLabels = [ones(1,offset(1)) zeros(1,offset(4+targetTrafficIndex)-offset(4+targetTrafficIndex-1))] + 1;
                else
                    testData = [data(:,offset(targetFlowerIndex-1)+1:offset(targetFlowerIndex)) data(:,offset(4+targetTrafficIndex-1)+1:offset(4+targetTrafficIndex))];
                    testLabels = [ones(1,offset(targetFlowerIndex)-offset(targetFlowerIndex-1)) zeros(1,offset(4+targetTrafficIndex)-offset(4+targetTrafficIndex-1))] + 1;
                end;
                
                layers = 2;
                for layer = 1:layers
                    
                    imgSize = 16;
                    params.patchWidth=9;           % width of a patch
                    params.n=params.patchWidth^2;   % dimensionality of input to RICA
                    % params.lambda = 0.0005;   % sparsity cost
                    params.lambda = 1e-2;
                    params.numFeatures = 64; % number of filter banks to learn
                    % params.epsilon = 1e-2;
                    params.epsilon = 1e-6;
                    params.m=20000; % num patches
                    
                    params.alpha = 0.5;%1;
                    params.beta = 0.00001;%0.0001;
                    params.numK = 10;   %80;%10;
                    params.numC = 2;   %80;%10;
                    params.numM = size(trainData,1);
                    params.numX = numX;
                    
                    %%  STEP 2: zca whitening
                    xx = [trainData testData];
                    xZCAWhite = ZCA_Gen(xx);
                    
                    %% STEP 3: stacked denoising autoencoder with softmax regression
                    outputs = sda_softmax(xZCAWhite,trainLabels,params);
                    
                    
                    %% STEP 4: manifold
                    beta = 0.1;
                    manilayers = 2;
                    total = [outputs(:,1:sum(numX,2)),outputs(:,sum(numX,2)+1:size(xx,2))];
                    [maniparameters]= LaplacianMatrix(outputs(:,1:sum(numX,2)),outputs(:,sum(numX,2)+1:size(xx,2)),2);
                    [allhx, D_cell, W_cell] = ManiRepresentationLearning(total, beta, manilayers, maniparameters);
                    
                    trainData = allhx(:,1:sum(numX,2));
                    testData = allhx(:,sum(numX,2)+1:size(xx,2));
                    
                end
                
                svmStruct = svmtrain(trainData',trainLabels','showplot',true);
                C = svmclassify(svmStruct,testData','showplot',true);
                errRate = sum(testLabels'~= C)/size(testData,2)  %mis-classification rate
                result = [result;1 - errRate];
                
                iCnt = iCnt + 1;
            end
        end
    end
end
% xlswrite('C:\Users\pc\Documents\MATLAB\icbk\result\results.xlsx',result);