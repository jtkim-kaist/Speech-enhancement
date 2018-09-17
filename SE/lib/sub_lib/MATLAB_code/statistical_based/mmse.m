function mmse(filename,outfile,SPU)

%
%  Implements the MMSE algorithm [1].
% 
%  Usage:  mmse(noisyFile, outputFile, SPU)
%           
%         infile - noisy speech file in .wav format
%         outputFile - enhanced output file in .wav format
%         SPU  - if 1, includes speech-presence uncertainty
%                if 0, doesnt include speech-presence uncertainty
%  
%
%  Example call:  mmse('sp04_babble_sn10.wav','out_mmse.wav',1);
%
%  References:
%   [1] Ephraim, Y. and Malah, D. (1985). Speech enhancement using a minimum 
%       mean-square error log-spectral amplitude estimator. IEEE Trans. Acoust., 
%       Speech, Signal Process., ASSP-23(2), 443-445.
%   
% Authors: Philipos C. Loizou
%
% Copyright (c) 2006 by Philipos C. Loizou
% $Revision: 0.0 $  $Date: 10/09/2006 $
%-------------------------------------------------------------------------

if nargin<3
    fprintf('Usage: mmse(infile.wav,outfile.wav,SPU) \n');
    fprintf('where SPU=1 - includes speech presence uncertainty\n');
    fprintf('      SPU=0 - does not includes speech presence uncertainty\n\n');
    return;
end;

if SPU~=1 & SPU~=0
    error('ERROR: SPU needs to be either 1 or 0.');
end

[x, Srate, bits]= wavread( filename);	


% =============== Initialize variables ===============

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
j=1;
noise_mean=zeros(nFFT,1);
for k=1:6
    noise_mean=noise_mean+abs(fft(win.*x(j:j+len-1),nFFT));
    j=j+len;
end
noise_mu=noise_mean/6;
noise_mu2=noise_mu.^2;

%--- allocate memory and initialize various variables

k=1;
img=sqrt(-1);
x_old=zeros(len1,1);
Nframes=floor(length(x)/len2)-1;
xfinal=zeros(Nframes*len2,1);

% --------------- Initialize parameters ------------
%
k=1;
aa=0.98;
eta= 0.15;
mu=0.98;
c=sqrt(pi)/2;
qk=0.3;
qkr=(1-qk)/qk;
ksi_min=10^(-25/10); % note that in Chap. 7, ref. [17], ksi_min (dB)=-15 dB is recommended

%===============================  Start Processing =======================================================
%
for n=1:Nframes

    insign=win.*x(k:k+len-1);

    %--- Take fourier transform of  frame
    %
    spec=fft(insign,nFFT);
    sig=abs(spec); % compute the magnitude
    sig2=sig.^2;

    gammak=min(sig2./noise_mu2,40);  % posteriori SNR
    if n==1
        ksi=aa+(1-aa)*max(gammak-1,0);
    else
        ksi=aa*Xk_prev./noise_mu2 + (1-aa)*max(gammak-1,0);     
        % decision-direct estimate of a priori SNR
        ksi=max(ksi_min,ksi);  % limit ksi to -25 dB
    end

    log_sigma_k= gammak.* ksi./ (1+ ksi)- log(1+ ksi); 
    vad_decision= sum( log_sigma_k)/nFFT;    
    if (vad_decision< eta) % noise only frame found
        noise_mu2= mu* noise_mu2+ (1- mu)* sig2;
    end
    % ===end of vad===

    vk=ksi.*gammak./(1+ksi);
    [j0,err]=besseli(0,vk/2);
    [j1,err2]=besseli(1,vk/2);
    if any(err) | any(err2)
        fprintf('ERROR! Overflow in Bessel calculation in frame: %d \n',n);
    else
        C=exp(-0.5*vk);
        A=((c*(vk.^0.5)).*C)./gammak;
        B=(1+vk).*j0+vk.*j1;
        hw=A.*B;
    end


    % --- estimate speech presence probability
    %
    if SPU==1
        evk=exp(vk);
        Lambda=qkr*evk./(1+ksi);
        pSAP=Lambda./(1+Lambda);
       sig=sig.*hw.*pSAP;
    else
        sig=sig.*hw;
    end
    
    Xk_prev=sig.^2;  % save for estimation of a priori SNR in next frame

    xi_w= ifft( sig .* exp(img*angle(spec)),nFFT);

    xi_w= real( xi_w);

    xfinal(k:k+ len2-1)= x_old+ xi_w(1:len1);
    x_old= xi_w(len1+ 1: len);

    k=k+len2;
    
end
%========================================================================================


wavwrite(xfinal,Srate,16,outfile);

