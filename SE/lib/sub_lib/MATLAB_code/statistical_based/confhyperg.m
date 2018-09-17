function z=confhyperg(a,b,x,n)
% 
% Computes the confluent hypergeometric function 
% using a series expansion:
%
%    f(a,b;x)=
%
%    1 + [ab/1!c]x + [a(a+1)/2!b(b+1)]x^2 +
%    [a(a+1)(a+2)/3!b(b+1)(b+2)]x^3 + ...
%
% The above series is expanded to n terms
%
% 
%
% Philipos C. Loizou

if nargin ~= 4
    error('Usage: confhyperg(a,b,x,n) - Incorrect number of arguments')
end

if (n <= 0 | n ~= floor(n))
    error('Usage: confhyperg (a,b,c,x,n) - n has to be a positive integer')
end

NEG=0;
if x<0
    x=abs(x);
    a=b-a;
    NEG=1;   
end

z = 0;
m = 0;
while (m<n)
    if (m == 0)
        delta = 1;
    else
        delta = delta .* x .* (a + (m - 1))  ./ (m .* (b + (m-1)));  
    end
   
   z = z + delta;
   m = m + 1;
end

if NEG==1  % if x<0
    z=exp(-x).*z;
end;