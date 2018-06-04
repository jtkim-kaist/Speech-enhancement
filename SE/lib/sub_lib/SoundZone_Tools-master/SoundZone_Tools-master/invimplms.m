function inv=invimplms(den,n,d)
% Inverse impulse using the Levinson-Durbin algorithm
% syntax inv=invimplms(den,n,d)
%         den - denominator impulse
%         n   - length of result
%         d   - delay of result
%         inv - inverse impulse response of length n with delay d
%
% Levinson-Durbin algorithm from Proakis and Manolokis p.865
%
% Author: Bob Cain, May 1, 2001 arcane[AT]arcanemethods[DOT]com

    m=xcorr(den,n-1);
    m=m(n:end);
    b=[den(d+1:-1:1);zeros(n-d-1,1)];
    inv=Tools.toepsolveMEX(m,b);
end


function quo=divimplms(num,den,n,d)
%Syntax quo=divimplms(num,den,n,d)
%       num - numerator impulse
%       den - denominator impulse
%       n   - length of result
%       d   - delay of result
%        quo - quotient impulse response of length n delayed by d
%
% Levinson-Durbin algorithm from Proakis and Manolokis p.865
%
% Author: Bob Cain, May 1, 2001 arcane@arcanemethods.com

    m=xcorr(den,n-1);
    m=m(n:end);
    b=xcorr([zeros(d,1);num],den,n-1);
    b=b(n:-1:1);
    quo=Tools.toepsolveMEX(m,b);
end

function hinv=toepsolve(r,q)
% Solve Toeplitz system of equations.
%    Solves R*hinv = q, where R is the symmetric Toeplitz matrix
%    whos first column is r
%    Assumes all inputs are real
%    Inputs:  
%       r - first column of Toeplitz matrix, length n
%       q - rhs vector, length n
%    Outputs:
%       hinv - length n solution
%
%   Algorithm from Roberts & Mullis, p.233
%
%   Author: T. Krauss, Sept 10, 1997
%
%   Modified: R. Cain, Dec 16, 2004 to remove a pair of transposes
%             that caused errors.

    n=length(q);
    a=zeros(n+1,2);
    a(1,1)=1;

    hinv=zeros(n,1);
    hinv(1)=q(1)/r(1);

    alpha=r(1);
    c=1;
    d=2;

    for k=1:n-1,
       a(k+1,c)=0;
       a(1,d)=1;
       beta=0;
       j=1:k;
       beta=sum(r(k+2-j).*a(j,c))/alpha;
       a(j+1,d)=a(j+1,c)-beta*a(k+1-j,c);
       alpha=alpha*(1-beta^2);
       hinv(k+1,1)=(q(k+1)-sum(r(k+2-j).*hinv(j,1)))/alpha;
       hinv(j)=hinv(j)+a(k+2-j,d)*hinv(k+1);
       temp=c;
       c=d;
       d=temp;
    end
end

