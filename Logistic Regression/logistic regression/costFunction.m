function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

J = 0; % initialize cost to zero 

grad = zeros(size(theta)); % initialize gradient to zeros vector of size theta

power_term = -X * theta; % calculate the power term of sigmoid function

hypothesis = 1 ./ (1 + exp(power_term)); % calculate the hypothesis sigmoid function

J = (1/m) .* sum(( - y .* log(hypothesis)) - ((1 - y) .* log(1 - hypothesis))); % calculate the cost

grad = (1/m) .* X' * (hypothesis - y); % calculate the gradient

end
