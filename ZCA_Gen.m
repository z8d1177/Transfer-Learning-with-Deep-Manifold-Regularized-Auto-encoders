function [ xZCAWhite ] = ZCA_Gen( x )

%% Step 0: Zero-mean the data (by row)
avg = mean(x,1);
x = x - repmat(avg,size(x,1),1);

%% Step 1: Implement PCA to obtain xRot
sigma = x * x' / size(x,2);
[U,S,V] = svd(sigma);
xRot = U' * x;

%% Step 2: Find k, the number of components to retain
sum_lambda = sum(S(:));
k = size(S,1);
for i = size(S,1):-1:1
    rate = (sum_lambda - S(i,i)) / sum_lambda;
    if rate < 0.99
        k = min(k,i+1);
        break;
    end;
end

%% Step 3: Implement PCA with dimension reduction
xTilde = U(:,1:k)' * x;
xHat = U(:,1:k) * xTilde;

%% Step 4: Implement PCA with whitening and regularisation
epsilon = 1e-1; 
xPCAWhite = diag(1./sqrt(diag(S) + epsilon)) * xRot;

%% Step 5: Implement ZCA whitening
xZCAWhite = U * xPCAWhite;
end

