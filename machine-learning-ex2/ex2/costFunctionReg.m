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

h = sigmoid(X*theta);
I=ones(size(y,1),1);

J = (1/m)*(-y'*log(h)-(I-y)'*log(1-h))+(lambda/(2*m))*(sumsqr(theta)-(theta(1)^2));

theta_x = theta;
theta_x(1) = 0;% since theta_o is treated seperately in regularization
grad = (1/m)*X'*(h-y)+(lambda/m)*theta_x;

% =============================================================

end
