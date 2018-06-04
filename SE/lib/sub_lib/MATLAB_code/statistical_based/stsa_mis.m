function stsa_mis(filename,outfile)

%
%  Implements the Bayesian estimator based on the modified Itakura-Saito 
%  distortion measure [1, Eq. 43].
% 
%  Usage:  stsa_mis(noisyFile, outputFile)
%           
%         infile - noisy speech file in .wav format
%         outputFile - enhanced output file in .wav format
%  
%
%  Example call:  stsa_mis('sp04_babble_sn10.wav','out_mis.wav');
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

if nargin<2
   fprintf('Usage: stsa_mis inFile outFile.wav \n\n');
   return;
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
nFFT=len;
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
   

img=sqrt(-1);
x_old=zeros(len1,1);
Nframes=floor(length(x)/len2)-1;
xfinal=zeros(Nframes*len2,1);

%===============================  Start Processing =======================================================
%
k=1;
aa=0.98;
fprintf('\nThis might take some time ...\n');
for n=1:Nframes 
   
  
   insign=win.*x(k:k+len-1);
    
   %--- Take fourier transform of  frame ----
   
   spec=fft(insign,nFFT);   
   sig=abs(spec); % compute the magnitude
   sig2=sig.^2;
   
       gammak=min(sig2./noise_mu2,40);  % post SNR. Limit it to avoid overflows
       if n==1
           ksi=aa+(1-aa)*max(gammak-1,0);
       else
           ksi=aa*Xk_prev./noise_mu2 + (1-aa)*max(gammak-1,0);     % a priori SNR   
       end
     
       vk=ksi.*gammak./(1+ksi);
      
       sig_hat=log(comp_int(vk,gammak,sig)); % Eq. 41
       
       Xk_prev=sig_hat.^2;
       
       xi_w= ifft( sig_hat.* exp(img*angle(spec))); 
	   xi_w= real( xi_w);
	  
      
	% --- Overlap and add ---------------
    %
    xfinal(k:k+ len2-1)= x_old+ xi_w(1:len1);
	x_old= xi_w(len1+ 1: len);
   
    if rem(n,20)==0, fprintf('Frame: %d Percent completed:%4.2f\n',n,n*100/Nframes); end;
 
 k=k+len2;
end
%========================================================================================




wavwrite(xfinal,Srate,16,outfile);

%------------------------------E N D  -----------------------------------
function xhat=comp_int(vk,gammak,Yk)

% -- Evaluates Eq. 43 in [1]
%

Yk2=Yk.*Yk;
G2=gammak.^2;
EV=exp(-vk);

N=40; % number of terms to keep in infinite sum (Eq. 43)
L=length(vk)/2+1;
J1=zeros(L,1);
J2=zeros(L,1);

for j=1:L
  sum=0;  sum_b=0;
  for m=0:N
     F=factorial(m);
     d1=(vk(j))^m;
     d2=hyperg(-m,-m,0.5,Yk2(j)/(4*G2(j)),10);
     d2_b=hyperg(-m,-m,1.5,Yk2(j)/(4*G2(j)),10);
     sum=sum+d1*d2/F;
     sum_b=sum_b+gamma(m+1.5)*d1*d2_b/(F*gamma(m+1));
end
 J1(j)=sum;
 J2(j)=sum_b;
end
 

J1=J1.*EV(1:L);
J2=J2.*EV(1:L).*sqrt(vk(1:L)).*Yk(1:L)./gammak(1:L);


xhat2=max(real(J1+J2),0.00001);
xhat = [xhat2; flipud(xhat2(2:L-1))];
