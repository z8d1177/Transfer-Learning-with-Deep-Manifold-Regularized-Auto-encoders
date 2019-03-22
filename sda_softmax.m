function [ outputs ] = sda_softmax( xx,TrainLabel,parameters )

numM = parameters.numM;
numK = parameters.numK;
numC = parameters.numC;
alpha = parameters.alpha;
beta = parameters.beta;
numX = parameters.numX;
TrainData = xx(:,1:sum(numX,2));
TestData = xx(:,sum(numX,2)+1:size(xx,2));

%%Initialize the parameter
theta = initialize_img1(numK, numM, numC, TrainData, TestData);    % Randomly initialize the parameters

%%Use minFunc to minimize the function
options.Method = 'lbfgs';
options.maxIter = 300;	  % Maximum number of iterations of L-BFGS to run
options.display = 'on';
options.TolFun  = 1e-6;
options.TolX = 1e-1119;
options.maxFunEvals = 4000;
label1 = [TrainLabel;ones(1,size(TrainLabel,2))-TrainLabel];
[opttheta, cost] = minFunc( @(p) computeObjectAndGradiend(p, numM, numK,...
    numC, numX, alpha, beta, TrainData, TestData, label1), theta, options);

W1 = reshape(opttheta(1:numK*numM), numK, numM);
W2 = reshape(opttheta(numK*numM+1:numK*numM+numK*numC), numC, numK);
W22 = reshape(opttheta(numK*numM+numK*numC+1:numK*numM+2*numK*numC), numK, numC);
W11 = reshape(opttheta(numK*numM+2*numK*numC+1:2*numK*numM+2*numK*numC), numM, numK);
b1 = opttheta(2*numK*numM+2*numK*numC+1:2*numK*numM+2*numK*numC+numK);
b2 = opttheta(2*numK*numM+2*numK*numC+numK+1:2*numK*numM+2*numK*numC+numK+numC);
b22 = opttheta(2*numK*numM+2*numK*numC+numK+numC+1:2*numK*numM+2*numK*numC+2*numK+numC);
b11 = opttheta(2*numK*numM+2*numK*numC+2*numK+numC+1:2*numK*numM+2*numK*numC+2*numK+numC+numM);

hiddeninputs = sigmoid(W1 * xx + b1 * ones(1, size(xx,2)));
probability = sigmoid(W2 * hiddeninputs + b2 * ones(1, size(hiddeninputs,2)));
hiddenvalues1 = sigmoid(W22 * probability + b22 * ones(1, size(probability,2)));
outputs = sigmoid(W11 * hiddenvalues1 + b11 * ones(1, size(hiddenvalues1,2)));

end

