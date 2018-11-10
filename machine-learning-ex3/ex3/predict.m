function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
m = size(X, 1);
% num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% calculate z(2) = theta1 * a(1), then a(2) = g(z(2))
% a(1) is X
% from part 1,  a(2) = g(X*theta1) -- vectorized implementation
a2 = sigmoid(X * transpose(Theta1));

% add ones to the a(2) matrix
ma2 = size(a2, 1);
a2 = [ones(ma2, 1) a2];

% calculate z(3) = theta2 * a(2), then a(3) = g(z(3))
% a(3) = h(x)
a3 = sigmoid(a2 * transpose(Theta2));

% for each row of p, pick the largest value
% p represents the index
[highest_prob, p] = max(a3,[],2);

% =========================================================================


end
