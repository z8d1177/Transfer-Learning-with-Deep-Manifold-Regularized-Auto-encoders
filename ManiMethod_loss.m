function [hx, W, D] = ManiMethod_loss(xx, parameter)
alpha = parameter.alpha;
lambda = parameter.lambda;

xx = full(xx);
W = [];
[d, n] = size(xx);
t = var(xx');

% index = t > 0.000001;
% xx = xx(index,:);

H = eye(n) - ones(n,n)/n;
xxc = xx*H;

s = sum(abs(xxc),2);
index = ( s > 1e-9 );
xxc = xxc(index,:);
xx = xx(index,:);

C = xxc*xxc';

[W, D] = get_W_loss(xxc',diag(C),lambda,parameter);

W = full(W);
hx = W'*xx;

hx = tanh(hx*alpha);
end

function [W, D] = get_W_loss(X,A,lambda,parameter)

[n, d] = size(X);
D = ones(n,1);
max_iter = 10 % ususllay converge in less than 10 iterations

obj_old = 1e9;
for i = 1:max_iter
  
    W = update_W(X,A, D, lambda,parameter);  
    D = update_D(X-X*W);
end
end

function W = update_W(X,A, D, lambda,parameter)
C = X'* bsxfun(@times,X,D);
W = inv(C + lambda*diag(A)+ parameter.beta*X'*parameter.L*X) * C;
end

function D = update_D(R)
s = sqrt(sum(R.*R,2));
s = s + 1e-7;
index = (s>0);
D = zeros(length(s),1);
D(index) = 1./s(index);
D = D/2;
end
