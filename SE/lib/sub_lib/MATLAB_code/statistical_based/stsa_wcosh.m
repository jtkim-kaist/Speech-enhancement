function stsa_wcosh(filename,outfile,p)

%
%  Implements the Bayesian estimator based on the weighted cosh
%  distortion measure [1, Eq. 34].
% 
%  Usage:  stsa_wcosh(noisyFile, outputFile, p)
%           
%         infile - noisy speech file in .wav format
%         outputFile - enhanced output file in .wav format
%         p  - power exponent used in the weighted cosh measure.
%              Valid values for p: p>-1
%  
%
%  Example call:  stsa_wcosh('sp04_babble_sn10.wav','out_wcosh.wav',-0.5);
%
%  References:
%   [1] Loizou, P. (2005). Speech enhancement based on perceptually motivated 
%       Bayesian estimators of the speech magnitude spectrum. IEEE Trans. on Speech 
%       and Audio Processing, 13(5), 857-869.
%   
% Author: Philipos C. Loizou
%
% Copyright (c) 2006 by Philipos C. Loizou
% $Revision: 0.0 $  $Date: 10/09/2006 $
%-------------------------------------------------------------------------

if nargin<3
    fprintf('Usage: stsa_wcosh(infile.wav,outfile.wav,p) \n');
    fprintf('       where p>-1 \n\n');
    return;
end;

if p<-1
    error('ERROR!  p needs to be larger than -1.\n\n');
end

[x, Srate, bits]= wavread( filename);	



% =============== Initialize variables ===============
%

len=floor(20*Srate/1000); % Frame size in samples
if rem(len,2)==1, len=len+1; end;
PERC=50; % window overlap in percent of frame size
len1=floor(len*PERC/100);
len2=len-len1; 


win=hanning(len);  % define window
win = win*len2/sum(win);  % normalize window for equal level output 

% Noise magnitude calculations - assuming that the first 6 frames is noise/silence
%
nFFT=2*len;
nFFT2=len/2;
noise_mean=zeros(nFFT,1);
j=1;
for k=1:5
   noise_mean=noise_mean+abs(fft(win.*x(j:j+len-1),nFFT));
   j=j+len;
end
noise_mu=noise_mean/5;
noise_mu2=noise_mu.^2;

%--- allocate memory and initialize various variables
   
x_old=zeros(len1,1);
Nframes=floor(length(x)/len2)-1;
xfinal=zeros(Nframes*len2,1);


%===============================  Start Processing =======================================================
%
k=1;
aa=0.98;
CC2=sqrt(gamma((p+3)/2)/gamma((p+1)/2));

for n=1:Nframes 
   
  
   insign=win.*x(k:k+len-1);
    
   %--- Take fourier transform of  frame
   
   spec=fft(insign,nFFT);   
   sig=abs(spec); % compute the magnitude
   sig2=sig.^2;
   
       gammak=min(sig2./noise_mu2,40);  % post SNR
       if n==1
           ksi=aa+(1-aa)*max(gammak-1,0);
       else
       
           ksi=aa*Xk_prev./noise_mu2 + (1-aa)*max(gammak-1,0);     % a priori SNR   
          
       end
     
       vk=ksi.*gammak./(1+ksi);
       
         
       % --- for the weighted cosh measure
       
       numer=CC2*sqrt(vk.*confhyperg(-(p+1)/2,1,-vk,100));
       denom=gammak.*sqrt(confhyperg(-(p-1)/2,1,-vk,100));
       hw=numer./denom;
       
       
       sig=sig.*hw;
       Xk_prev=sig.^2;
       
       xi_w= ifft( hw .* spec, nFFT); 
	   xi_w= real( xi_w);
	
          
	% --- Overlap and add ---------------
    %
    xfinal(k:k+ len2-1)= x_old+ xi_w(1:len1);
	x_old= xi_w(len1+ 1: len);
   
 
 k=k+len2;
end
%========================================================================================


wavwrite(xfinal,Srate,16,outfile);

