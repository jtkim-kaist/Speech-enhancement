function [y]=fconv(x, h)
% Fast Parallelised Convolution
%   [y] = FCONV(x, h) convolves x and h in the frequency domain
%         to +-1.
%
%      x = input vector
%      h = input vector
% 
%      See also CONV
%

Ly=size(x,1)+size(h,1)-1;  % 
Ly2=pow2(nextpow2(Ly));    % Find smallest power of 2 that is > Ly

if isa(x, 'gpuArray')
Ly  = gpuArray(Ly);
Ly2 = gpuArray(Ly2);
end
    
X=fft(x, Ly2);             % Fast Fourier transform
H=fft(h, Ly2);	           % Fast Fourier transform
Y=X.*H;        	           % 
y=real(ifft(Y, Ly2));      % Inverse fast Fourier transform
y=y(1:1:Ly,:);             % Take just the first N elements