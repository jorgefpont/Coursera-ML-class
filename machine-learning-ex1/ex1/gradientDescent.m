function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
s0=0;
s1=0;
i=0;
for i=1:m

  h = X(i,:) * theta;       % calculate h_theta
  s0 = s0 + (h - y(i));     % calculate error sum term for theta-0
  s1 = s1 + (h - y(i))*X(i,2); % calculate error sum term for theta-1

end

s0 = (alpha/m)*s0;
s1 = (alpha/m)*s1;
S = [s0 ; s1];
theta = theta - S;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
