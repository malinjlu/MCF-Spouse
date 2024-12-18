function [CI, r, p] = my_fisherz_test(data,X, Y, S,N, alpha)
% COND_INDEP_FISHER_Z Test if X indep Y given Z using Fisher's Z test
% CI = cond_indep_fisher_z(X, Y, S, C, N, alpha)
%
% C is the covariance (or correlation) matrix
% N is the sample size
% alpha is the significance level (default: 0.05)
%
% See p133 of T. Anderson, "An Intro. to Multivariate Statistical Analysis", 1984

if nargin < 6, alpha = 0.05; end

C=cov((data(:,[X,Y,S])));
size_C=size(C,2);
X1=1;
Y1=2;
S1=[3:size_C];


r = partial_corr_coef(C, X1, Y1, S1);
z = 0.5*log( (1+r)/(1-r) );
z0 = 0;
W = sqrt(N - length(S1) - 3)*(z-z0); % W ~ N(0,1)
cutoff = norminv(1 - 0.5*alpha); % P(|W| <= cutoff) = 0.95
%cutoff = mynorminv(1 - 0.5*alpha); % P(|W| <= cutoff) = 0.95
if abs(W) < cutoff
  CI = 1;
else % reject the null hypothesis that rho = 0
  CI = 0;
end
p = normcdf(W);
r=abs(r);
%p = mynormcdf(W);

%%%%%%%%%

function p = normcdf(x,mu,sigma)
%NORMCDF Normal cumulative distribution function (cdf).
%   P = NORMCDF(X,MU,SIGMA) computes the normal cdf with mean MU and
%   standard deviation SIGMA at the values in X.
%
%   The size of P is the common size of X, MU and SIGMA. A scalar input  
%   functions as a constant matrix of the same size as the other inputs.    
%
%   Default values for MU and SIGMA are 0 and 1 respectively.

%   References:
%      [1]  M. Abramowitz and I. A. Stegun, "Handbook of Mathematical
%      Functions", Government Printing Office, 1964, 26.2.

%   Copyright (c) 1993-98 by The MathWorks, Inc.
%   $Revision: 1.1.1.1 $  $Date: 2005/04/26 02:29:18 $

if nargin < 3, 
    sigma = 1;
end

if nargin < 2;
    mu = 0;
end

[errorcode x mu sigma] = distchck(3,x,mu,sigma);

if errorcode > 0
    error('Requires non-scalar arguments to match in size.');
end

%   Initialize P to zero.
p = zeros(size(x));

% Return NaN if SIGMA is not positive.
k1 = find(sigma <= 0);
if any(k1)
    tmp   = NaN;
    p(k1) = tmp(ones(size(k1))); 
end

% Express normal CDF in terms of the error function.
k = find(sigma > 0);
if any(k)
    p(k) = 0.5 * erfc( - (x(k) - mu(k)) ./ (sigma(k) * sqrt(2)));
end

% Make sure that round-off errors never make P greater than 1.
k2 = find(p > 1);
if any(k2)
    p(k2) = ones(size(k2));
end

%%%%%%%%

function x = norminv(p,mu,sigma);
%NORMINV Inverse of the normal cumulative distribution function (cdf).
%   X = NORMINV(P,MU,SIGMA) finds the inverse of the normal cdf with
%   mean, MU, and standard deviation, SIGMA.
%
%   The size of X is the common size of the input arguments. A scalar input  
%   functions as a constant matrix of the same size as the other inputs.    
%
%   Default values for MU and SIGMA are 0 and 1 respectively.

%   References:
%      [1]  M. Abramowitz and I. A. Stegun, "Handbook of Mathematical
%      Functions", Government Printing Office, 1964, 7.1.1 and 26.2.2

%   Copyright (c) 1993-98 by The MathWorks, Inc.
%   $Revision: 1.1.1.1 $  $Date: 2005/04/26 02:29:18 $

if nargin < 3, 
    sigma = 1;
end

if nargin < 2;
    mu = 0;
end

[errorcode p mu sigma] = distchck(3,p,mu,sigma);

if errorcode > 0
    error('Requires non-scalar arguments to match in size.');
end

% Allocate space for x.
x = zeros(size(p));

% Return NaN if the arguments are outside their respective limits.
k = find(sigma <= 0 | p < 0 | p > 1);
if any(k)
    tmp  = NaN;
    x(k) = tmp(ones(size(k))); 
end

% Put in the correct values when P is either 0 or 1.
k = find(p == 0);
if any(k)
    tmp  = Inf;
    x(k) = -tmp(ones(size(k)));
end

k = find(p == 1);
if any(k)
    tmp  = Inf;
    x(k) = tmp(ones(size(k))); 
end

% Compute the inverse function for the intermediate values.
k = find(p > 0  &  p < 1 & sigma > 0);
if any(k),
    x(k) = sqrt(2) * sigma(k) .* erfinv(2 * p(k) - 1) + mu(k);
end






function [r, c] = partial_corr_coef(S, i, j, Y)
% PARTIAL_CORR_COEF Compute a partial correlation coefficient
% [r, c] = partial_corr_coef(S, i, j, Y)
%
% S is the covariance (or correlation) matrix for X, Y, Z
% where X=[i j], Y is conditioned on, and Z is marginalized out.
% Let S2 = Cov[X | Y] be the partial covariance matrix.
% Then c = S2(i,j) and r = c / sqrt( S2(i,i) * S2(j,j) ) 
%

% Example: Anderson (1984) p129
% S = [1.0 0.8 -0.4;
%     0.8 1.0 -0.56;
%     -0.4 -0.56 1.0];
% r(1,3 | 2) = 0.0966 
%
% Example: Van de Geer (1971) p111
%S = [1     0.453 0.322;
%     0.453 1.0   0.596;
%     0.322 0.596 1];
% r(2,3 | 1) = 0.533

X = [i j];
i2 = 1; % find_equiv_posns(i, X);
j2 = 2; % find_equiv_posns(j, X);
S2 = S(X,X) - S(X,Y)*inv(S(Y,Y))*S(Y,X);
c = S2(i2,j2);
r = c / sqrt(S2(i2,i2) * S2(j2,j2));

