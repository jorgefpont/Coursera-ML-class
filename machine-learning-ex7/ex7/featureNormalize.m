function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% X is 2 x m
% mu is a 2 x 1 
% where the 1st element is the avg of all x1's
% and the second element is the average of the x2's
% mean(matrix) does a mean of the rows
mu = mean(X);

% bsxfun >> Apply element-wise operation to two arrays

% X_norm subracts the x1(i) avg from each x1(i) and ditto for x2(i)s
X_norm = bsxfun(@minus, X, mu);

% sigma is 2 x 1 (like mu) and the values are the std's of the columns
sigma = std(X_norm);

X_norm = bsxfun(@rdivide, X_norm, sigma);

% ============================================================

end
