function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
s=0;
i=0;
for i=1:m
% calculate h_theta
h = X(i,:) * theta;
s = s + (h - y(i))^2;

% theta'
% fprintf('i= %6.1f , h= %6.1f , y(i)= %6.1f, sum= %6.1f \n', i, h, y(i), s)

end

J = s/(2*m);

% =========================================================================

end
