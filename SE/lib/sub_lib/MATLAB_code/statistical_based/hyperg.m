function z=hyperg(a,b,c,x,n)
% HYPERGEOMETRIC2F1 Computes the hypergeometric function 
% using a series expansion:
%
%    f(a,b;c;x)=
%
%    1 + [ab/1!c]x + [a(a+1)b(b+1)/2!c(c+1)]x^2 +
%    [a(a+1)(a+2)b(b+1)(b+2)/3!c(c+1)(c+2)]x^3 + ...
%
% The series is expanded to n terms
%
% This function solves the Gaussian Hypergeometric Differential Equation:
%
%     x(1-x)y'' + {c-(a+b+1)x}y' - aby = 0
%
% The Hypergeometric function converges only for:
% |x| < 1
% c != 0, -1, -2, -3, ...
%
%
% Comments to:
% Diego Garcia   - d.garcia@ieee.org
% Chuck Mongiovi - mongiovi@fast.net
% June 14, 2002

if nargin ~= 5
    error('Usage: hypergeometric2f1(a,b,c,x,n) --> Wrong number of arguments')
end

if (n <= 0 | n ~= floor(n))
    error('Usage: hypergeometric2f1(a,b,c,x,n) --> n has to be a positive integer')
end

% if (abs(x) > 1)
%    z=min(0.99,x);
%    return;
%    error('Usage: hypergeometric2f1(a,b,c,x,n) --> |x| has to be less than 1')
% end

if (c <= 0 & c == floor(c))
    error('Usage: hypergeometric2f1(a,b,c,x,n) --> c != 0, -1, -2, -3, ...')
end

z = 0;
m = 0;
while (m<n)
    if (m == 0)
        delta = 1;
    else
        delta = delta .* x .* (a + (m - 1)) .* (b + (m-1)) ./ m ./ (c + (m-1));
   end
   z = z + delta;
   m = m + 1;
end