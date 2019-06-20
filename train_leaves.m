global params;

global numM;                % input data feature dimensions  m
global numK;                                 % number of hidden units k
global numC;
global beta;
global numX;


%% ======================================================================
%  STEP 1: Load data
%
load('C:\Users\pc\Documents\MATLAB\icbk\dataset\leaves.mat');
numExamples = size(data, 2);

result =[];
iCnt = 1;

for sourceIndex = 1:4
    for targetIndex = 1:4
        if sourceIndex == targetIndex
            continue
        end
        %% ======================================================================
        %  STEP 1: Load data
        if(sourceIndex == 1)
            trainDataOri = [lithocarpusData eucalyptusData];
            trainLabels = [ones(1,size(lithocarpusData,2)) zeros(1,size(eucalyptusData,2))];
            numX = [size(lithocarpusData,2) size(eucalyptusData,2)];
        end;
        if(sourceIndex == 2)
            trainDataOri = [prunusData eucalyptusData];
            trainLabels = [ones(1,size(prunusData,2)) zeros(1,size(eucalyptusData,2))];
            numX = [size(prunusData,2) size(eucalyptusData,2)];
        end;
        if(sourceIndex == 3)
            trainDataOri = [salixData eucalyptusData];
            trainLabels = [ones(1,size(salixData,2)) zeros(1,size(eucalyptusData,2))];
            numX = [size(salixData,2) size(eucalyptusData,2)];
        end;
        if(sourceIndex == 4)
            trainDataOri = [viburnumData eucalyptusData];
            trainLabels = [ones(1,size(viburnumData,2)) zeros(1,size(eucalyptusData,2))];
            numX = [size(viburnumData,2) size(eucalyptusData,2)];
        end;
        
        if(targetIndex == 1)
            testData = [lithocarpusData eucalyptusData];
            testLabels = [ones(1,size(lithocarpusData,2)) zeros(1,size(eucalyptusData,2))];
        end;
        if(targetIndex == 2)
            testData = [prunusData eucalyptusData];
            testLabels = [ones(1,size(prunusData,2)) zeros(1,size(eucalyptusData,2))];
        end;
        if(targetIndex == 3)
            testData = [salixData eucalyptusData];
            testLabels = [ones(1,size(salixData,2)) zeros(1,size(eucalyptusData,2))];
        end;
        if(targetIndex == 4)
            testData = [viburnumData eucalyptusData];
            testLabels = [ones(1,size(viburnumData,2)) zeros(1,size(eucalyptusData,2))];
        end;
        
        trainData = trainDataOri;
        
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
    end
end

iCnt = iCnt + 1;
xlswrite('C:\Users\pc\Documents\MATLAB\icbk\result\leaves_results.xlsx',result);
