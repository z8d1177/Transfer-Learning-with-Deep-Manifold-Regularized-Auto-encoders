function result = test_LR(traindata, testdata, trainlabel, testlabel)
    trainlabel = trainlabel -1;
    testlabel = testlabel - 1;


    %% use the model test the target domain data
    result = zeros(2,1);
    trainlabel(trainlabel==0)=-1;
    testlabel(testlabel==0)=-1;
    label = trainlabel;
    label1 = testlabel;
    tempTrainXY = scale_cols(traindata, trainlabel);
    
    % train the classifier
    c00 = zeros(size(tempTrainXY,1),1);
    lambdaLG = exp(linspace(-0.5,6,20));
    wbest=c00;
    f1max = -inf;
    for j = 1 : length(lambdaLG)
        c_0 = train_cg(tempTrainXY,c00,lambdaLG(j));
        f1 = logProb(tempTrainXY,c_0);
        if f1 > f1max
            f1max = f1;
            wbest = c_0;
        end
    end
    C = wbest;

    % test the test data   
    probability = 1./(1+1./(exp(C'*traindata)));
    probability(probability >= 0.5) = 1;
    probability(probability < 0.5) = -1;
    result = mean(probability(:) == label(:));