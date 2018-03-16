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

z = X * theta; % calculate the power term of sigmoid function for theta non zeros

g = sigmoid(z); % calculate the hypothesis sigmoid function

J = sum((- y).*log(g) - ( 1 - y ).*log( 1 - g))/m + ( theta' * theta * lambda ) / ( 2 * m )... % calculate the cost with regularization
    - ( theta(1) * theta(1) * lambda ) / ( 2 * m ) ; % minus theta zero regularization term


grad(1) = ( sum(( g - y ).*X( :,1 )))/m ; % calculate gradient for first term without regularization

for j = 2:length( theta )   
 
    grad(j) =   ( sum(( g - y ).*X( :,j ))) / m + (lambda*theta(j))/m; % calculate gradient for rest terms without regularization
 
end


end