function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
% ============================================================

% h = Theta * x
h = X * theta;

% J(theta) = (1/2*m) * (h - y)^2 + 
%     + lambda/(2m) * sum (theta ^2) ==> reguarization term

J = (1/(2*m)) * ((X * theta) - y)' * ((X * theta) - y) + ...
    (lambda/(2*m)) * sum(theta(2:length(theta)) .^2);  % new term

% grad_J = (1/m) * XT * (h - y) /// for theta 0
tempgrad = (1/m) * ( transpose(X) * (h - y) );
    
% gradient for theta 0
grad(1) = tempgrad(1);

% gradient for rest of theta terms
tempgrad = (1/m) * ( transpose(X) * (h - y) ) + ...
    (lambda/m) * theta;   % regularization term

grad(2:length(theta)) = tempgrad(2:size(theta));

% =========================================================================

grad = grad(:);

end
