function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% h = g(z) = g(X*theta) -- vectorized implementation of h(X)
z = X * theta;
h = (1+exp(-z)).^-1;

% J(theta) = (1/m) * ( (-yT log(h)) - (1-y)T log(1-h) ) + 
%     + lambda/(2m) * sum (theta ^2) ==> added term

J = (1/m) * ...
    (   ( transpose(-y) * log(h) ) - ...
        ( transpose(1-y) * log(1-h) ) ...
    ) + ...
    (lambda/(2*m)) * sum(theta(2:length(theta)) .^2);  % new term

% grad_J = (1/m) * XT * (h - y) /// for theta 0
tempgrad = (1/m) * ...
    transpose(X) * ...
    (h - y);
    
% gradient for theta 0
grad(1) = tempgrad(1);

tempgrad = (1/m) * ...
    transpose(X) * ...
    (h - y) + ...
    (lambda/m) * theta;   % new term

    % gradient for rest of theta terms
grad(2:length(theta)) = tempgrad(2:size(theta));
   

    
    





% =============================================================

end
