function [hx, W, D] = myMethod_loss(xx, beta, parameter)
% xx : dxn input
xx = full(xx);

% c  = var(xx');index = (c>0.000001);xx = xx(index,:); disp(['deleted ' num2str(sum(~index)) ' features ']);
W = [];

[d, n] = size(xx);
t = var(xx');
index = t > 0.000001;
xx = xx(index,:);

H = eye(n) - ones(n,n)/n;
xxc = xx*H;
% xxc = xx;

s = sum(abs(xxc),2);
index = ( s > 1e-9 );
xxc = xxc(index,:);
xx = xx(index,:);

C = xxc*xxc';  %%%%%%
% C = xx*xx';

% clear H;
% C0 = C;
% c = getC(xxc);
% C = C + lambda*diag(C);
% % % % % % % % % % % % C = C + lambda*diag(diag(C));
[W, D] = get_W_loss(xxc',diag(C),beta,parameter); % l21 loss
% C = C + lambda*eye(size(C)); %%%%%%%%%%%%%%%%%%%%%%%%%

% a = sort(abs(sum(C0)));
% disp(['C0 minimum : ' num2str(a(1)) ' ' num2str(a(2))]);

% W = inv(C)*C0;
% disp('compute W with adapt prior');
% W = learning_prior_from_data(xx', lambda);

% figure;hist(W(:),1000);
% W = C\C0;

if sum(sum(isnan(W))) > 0
    error('W is wrong' );
end
W = full(W);
hx = W'*xx;
% disp(['max values: ' num2str(max(max(hx)))]);
% hx = my_nonlinear(hx,alpha, 1);
% hx = tanh(hx*alpha);
% hx = tanh(hx*alpha;
% hx = W'*xx*alpha;

% b = (W'*xx - xx)*ones(n,1)/n;
% hx = W'*xx + repmat(b, 1, n); % not better
% hx = W'*xx;
% figure;hist(hx(:),1000);
% [alpha] = compute_alpha(hx, 2);
% hx = hx*alpha;
% hx = hard_tanh(hx*alpha);
hx = tanh(hx*2);
% figure;hist(hx(:),1000);title('my');  
% hx = my_nonlinear(hx,alpha, 1); % works well, and t does not effect the result
% hx = alpha*my_nonlinear1(hx, 0.02);
end

function [W, D] = get_W_loss(X,A,beta,parameter)
% min |XW - X|_{2,1} + lambda trace(W'*A*W)
[n, d] = size(X);
D = ones(n,1);
max_iter = 10 % ususllay converge in less than 10 iterations
% W = update_W(X,A, D, lambda);
% W = eye(d);
% D = update_D(X-X*W);
% obj_old = get_obj(X, A,W,lambda)
obj_old = 1e9;
for i = 1:max_iter
%     R = X-X*W;
%     o1 = sum(sum( R.*bsxfun(@times,R,D))) + sum(1./D)/4 + lambda*sum(sum( W.*bsxfun(@times,W,A)))
%     d1 = sum(1./D)/4
    
    W = update_W(X,A, D, beta,parameter);
    
%     R = X-X*W;
%     o2 = sum(sum( R.*bsxfun(@times,R,D)))+sum(1./D)/4 + lambda*sum(sum( W.*bsxfun(@times,W,A)))
    
%     d2 = sum(1./D)/4
    
    D = update_D(X-X*W);
%     min(D)
    
%     o3 = sum(sum( R.*bsxfun(@times,R,D))) +sum(1./D)/4 + lambda*sum(sum( W.*bsxfun(@times,W,A)))
%     d3 = sum(1./D)/4
    
%     obj_new = get_obj(X,A,W,lambda,parameter);
%     assert(obj_new <= obj_old, ['old: ' num2str(obj_old) ', new: ' num2str(obj_new)]);
%     if obj_old - obj_new < obj_new * 1e-5
%         disp(['converged in ' num2str(i) ' iterations']);
%         break;
%     end
%     obj_old = obj_new;
end
end

function obj = get_obj(X,A,W,beta,parameter)
% A is vector
R = X - X*W;
obj1 = sum(sqrt(sum(R.*R, 2)));
% obj2 = sum(sum( W.*bsxfun(@times,W,A)));
% obj2 = sum(sqrt(sum(W.*W,2)));
obj3 = trace(W'*X'*parameter*X*W);
obj = obj1 + beta*obj3;
end

function W = update_W(X,A, D, beta,parameter)
% C = X'*X;
C = X'* bsxfun(@times,X,D);
W = inv(C + 0.5*diag(A)+ beta*X'*parameter*X) * C;
end

function D = update_D(R)
s = sqrt(sum(R.*R,2));
s = s + 1e-7;
index = (s>0);
% disp(['num of zero: ' num2str( length(index) -  sum(index))]);
D = zeros(length(s),1);
D(index) = 1./s(index);
D = D/2;
end
