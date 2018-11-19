function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
% =====
%
% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network // BECAUSE nn_params is the vector with the 
% unrolled versions of Theta1 and Theta2

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables -- # of rows in X matrix
m = size(X, 1);

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%                   <<< YOUR CODE HERE >>>

% do some check just in case
% fprintf('# rows in X == length of y ? (1=yes, 0=no \n')
% m == size(y,1)

% Add ones to the X data matrix
X = [ones(m, 1) X];

% check sizes
%fprintf('Size y: '), size(y)
%fprintf('Size X: '), size(X)
%fprintf('Size Theta1: '), size(Theta1)
%fprintf('Size Theta2: '), size(Theta2)

% =============================================================
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
% =============================================================

% calculate z(2) = theta1 * a(1), then a(2) = g(z(2))
% a(1) is X
% from part 1,  a(2) = g(X*theta1) -- vectorized implementation
z2 = X * transpose(Theta1);
a2 = sigmoid(z2);

% add ones to the a(2) matrix
ma2 = size(a2, 1);
a2 = [ones(ma2, 1) a2];
  
% calculate z(3) = theta2 * a(2), then a(3) = g(z(3))
% a(3) = h(x) 
% in this case a3 is a 5000 * 10, so vectors y(i) are rows
z3 = a2 * transpose(Theta2);
a3 = sigmoid(z3);

% check sizes
%fprintf('\n*****Beginning run ...\n\n')
%fprintf('Size z2: '), size(z2)
%fprintf('Size a2: '), size(a2)
%fprintf('Size a2: '), size(a2)
%fprintf('Size z3: '), size(z3)
%fprintf('Size a3: '), size(a3)

% construct y_matrix
y_matrix = zeros(m,num_labels);

for i = 1:m,
  temp = y(i);
  y_matrix(i, temp) = 1;
end;
 
% Calculate the unregularized cost function J
% J(theta) = (1/m) * ( (-yT log(h)) - (1-y)T log(1-h) )

J = (1/m) * ( ...
    (-y_matrix .* (log(a3))) - ...
     ((1-y_matrix) .* (log(1-a3))));
     
J = sum(J(:));

% Calculate the regularized cost function J
% Square Theta matrices minus the first columns
t_Theta1 = Theta1(:,2:size(Theta1,2));
t_Theta2 = Theta2(:,2:size(Theta2,2));

t_Theta1 = t_Theta1 .^2;
t_Theta2 = t_Theta2 .^2;

ss_Theta1 = sum(t_Theta1(:));
ss_Theta2 = sum(t_Theta2(:));

reg_term = (lambda/(2*m)) * (ss_Theta1 + ss_Theta2);

J = J + reg_term;

% =============================================================
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
% =============================================================






% =============================================================
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
% =============================================================






% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
