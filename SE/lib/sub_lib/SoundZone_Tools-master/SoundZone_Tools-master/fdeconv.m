function [h]=fdeconv(x, y)
% Fast Parallelised Deconvolution
%   [h] = FDECONV(x, y) deconvolves x and y in the frequency domain
%
%      x = input vector (original)
%      y = input vector (altered)
% 
%      See also DECONV
%

Lh=size(y,1)-size(x,1)+1;  % 
Lh2=pow2(nextpow2(Lh));    % Find smallest power of 2 that is > Ly

if isa(x, 'gpuArray')
Lh  = gpuArray(Lh);
Lh2 = gpuArray(Lh2);
end
    
X=fft(x, Lh2);             % Fast Fourier transform
Y=fft(y, Lh2);	           % Fast Fourier transform
H=Y./X;        	           % 
h=real(ifft(H, Lh2));      % Inverse fast Fourier transform
h=h(1:1:Lh,:);             % Take just the first N elements