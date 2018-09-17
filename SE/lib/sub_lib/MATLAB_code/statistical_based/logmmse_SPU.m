function logmmse_SPU(filename,outfile,option)

%
%  Implements the logMMSE algorithm with signal-presence uncertainty (SPU) [1].
%  Four different methods for estimating the a priori probability of speech absence
%  (P(H0)) are implemented.
% 
%  Usage:  logmmse_SPU(noisyFile, outputFile, option)
%           
%         infile - noisy speech file in .wav format
%         outputFile - enhanced output file in .wav format
%         option  - method used to estimate the a priori probability of speech
%                   absence, P(Ho):
%                  1  - hard decision (Soon et al. [2])
%                  2  - soft decision (Soon et al. [2])
%                  3  - Malah et al.(1999) - ICASSP
%                  4  - Cohen (2002) [1]
%
%
%  Example call:  logmmse_SPU('sp04_babble_sn10.wav','out_logSPU.wav',1);
%
%  References:
%   [1] Cohen, I. (2002). Optimal speech enhancement under signal presence 
%       uncertainty using log-spectra amplitude estimator. IEEE Signal Processing 
%       Letters, 9(4), 113-116.
%   [2] Soon, I., Koh, S., and Yeo, C. (1999). Improved noise suppression
%       filter using self-adaptive estimator of probability of speech absence. 
%       Signal Processing, 75, 151-159.
%   
% Author: Philipos C. Loizou
%
% Copyright (c) 2006 by Philipos C. Loizou
% $Revision: 0.0 $  $Date: 10/09/2006 $
%-------------------------------------------------------------------------

if nargin<3
    fprintf('Usage: logmmse_SPU(infile.wav,outfile.wav,option) \n');
    fprintf('where option = \n');
    fprintf(' 1  - hard decision ( Soon et al)\n');
    fprintf(' 2  - soft decision (Soon et al.)\n');
    fprintf(' 3  - Malah et al.(1999) \n');
    fprintf(' 4  - Cohen (2002) \n');
    return;
end;

if option<1 | option>4 | rem(option,1)~=0
    error('ERROR!  option needs to be an integer between 1 and 4.\n\n');
end

[x, Srate, bits]= wavread( filename);	



% =============== Initialize variables ===============
%

len=floor(20*Srate/1000); % Frame size in samples
if rem(len,2)==1, len=len+1; end;
PERC=50; % window overlap in percent of frame size
len1=floor(len*PERC/100);
len2=len-len1;

win=hamming(len);  % define window

% Noise magnitude calculations - assuming that the first 6 frames is
% noise/silence 
%
nFFT=len;
nFFT2=floor(len/2);
noise_mean=zeros(nFFT,1);
j=1;
for k=1:6
    noise_mean=noise_mean+abs(fft(win.*x(j:j+len-1),nFFT));
    j=j+len;
end
noise_mu=noise_mean/6;
noise_mu2=noise_mu.^2;

%--- allocate memory and initialize various variables


aa=0.98;
mu=0.98;
eta=0.15;
img=sqrt(-1);
x_old=zeros(len1,1);
Nframes=floor(length(x)/len2)-floor(len/len2);
xfinal=zeros(Nframes*len2,1);


if option==4 % Cohen's method  
global zetak zeta_fr_old z_peak

      len2a=len/2+1;
      zetak=zeros(len2a,1); 
      zeta_fr_old=1000;  
      z_peak=0;     
end;

%===============================  Start Processing ======================================================= 
%

qk=0.5*ones(len,1);
ksi_old=zeros(len,1);
ksi_min=10^(-25/10);
%qkr=(1-qk)/qk;
%qk2=1/(1-qk);

Gmin=10^(-20/10);  % needed for Cohen's implementation
k=1;

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
        ksi=aa*Xk_prev./noise_mu2 + (1-aa)*max(gammak-1,0);     
        % a priori SNR
        ksi=max(ksi_min,ksi);  % limit ksi to -25 dB
    end

    log_sigma_k= gammak.* ksi./ (1+ ksi)- log(1+ ksi);    
    vad_decision= sum( log_sigma_k)/ len;    
    if (vad_decision< eta) 
        % noise only frame found
        noise_mu2= mu* noise_mu2+ (1- mu)* sig2;
    end
    % ===end of vad===

    %ksi=qk2*ksi;  
    A=ksi./(1+ksi);
    vk=A.*gammak;
    ei_vk=0.5*expint(vk);
    hw=A.*exp(ei_vk);

    % --- estimate conditional speech-presence probability ---------------
    %
    [qk]=est_sap(qk,ksi,ksi_old,gammak,option);   % estimate P(Ho)- a priori speech absence prob.  
    pSAP = (1-qk)./(1-qk+qk.*(1+ksi).*exp(-vk)); % P(H1 | Yk)


    % ---- Cohen's 2002 ------
    %
    Gmin2=Gmin.^(1-pSAP); % Cohen's (2002) - Eq 8
    Gcohen=(hw.^pSAP).*Gmin2;
    sig = sig.*Gcohen;
    %----------------------------
 
    Xk_prev=sig.^2;
    ksi_old=ksi; % needed for Cohen's method for estimating q

    xi_w= ifft( sig .* exp(img*angle(spec)));
    xi_w= real( xi_w);

    % --------- Overlap and add ---------------
    %
    xfinal(k:k+ len2-1)= x_old+ xi_w(1:len1);
    x_old= xi_w(len1+ 1: len);

    k=k+len2;
end
%========================================================================================


wavwrite(xfinal,Srate,16,outfile);

%--------------------------- E N D -----------------------------------------


function [qk]=est_sap(qk,xsi,xsi_old,gammak,type)

% function returns a priori probability of speech absence, P(Ho)
%

global zetak zeta_fr_old z_peak

if type ==1  % hard-decision: Soon et al.
    beta=0.1;
    dk=ones(length(xsi),1);
    i0=besseli(0,2*(gammak.*xsi).^0.5);
    temp=exp(-xsi).*i0;
    indx=find(temp>1);
    dk(indx)=0;
    
    qk=beta*dk + (1-beta)*qk;

    
    
elseif type==2  % soft-decision: Soon et al.
    beta=0.1;
    i0=besseli(0,2*(gammak.*xsi).^0.5);
    
    temp=exp(-xsi).*i0;
    P_Ho=1./(1+temp);
    P_Ho=min(1,P_Ho);
    
    qk=beta*P_Ho + (1-beta)*qk;
    
    
elseif type==3  % Malah et al. (1999)
    
    if mean(gammak(1:floor(length(gammak)/2)))> 2.4 % VAD detector
        
      beta=0.95;
      gamma_th=0.8;
      dk=ones(length(xsi),1);
      indx=find(gammak>gamma_th);
      dk(indx)=0;
    
      qk=beta*qk+(1-beta)*dk;
    end
    
elseif type==4  % Cohen (2002)
    beta=0.7;
    len=length(qk);
    len2=len/2+1;
    
    zetak=beta*zetak+(1-beta)*xsi_old(1:len2);
    
    
    z_min=0.1; z_max=0.3162;
    C=log10(z_max/z_min);
    zp_min=1; zp_max=10;
    zeta_local=smoothing(zetak,1);
    zeta_global=smoothing(zetak,15);
    
    Plocal=zeros(len2,1);   % estimate P_local
    imax=find(zeta_local>z_max);
    Plocal(imax)=1;
    ibet=find(zeta_local>z_min & zeta_local<z_max);
    Plocal(ibet)=log10(zeta_local(ibet)/z_min)/C;
    
    
    Pglob=zeros(len2,1);   % estimate P_global
    imax=find(zeta_global>z_max);
    Pglob(imax)=1;
    ibet=find(zeta_global>z_min & zeta_global<z_max);
    Pglob(ibet)=log10(zeta_global(ibet)/z_min)/C;
    
    zeta_fr=mean(zetak);  % estimate Pframe
    if  zeta_fr>z_min
        if zeta_fr>zeta_fr_old
            Pframe=1;
            z_peak=min(max(zeta_fr,zp_min),zp_max);
        else
            if zeta_fr <=z_peak*z_min, Pframe=0;
            elseif zeta_fr>= z_peak*z_max, Pframe=1;
            else, Pframe=log10(zeta_fr/z_peak/z_min)/C;
            end
        end
    else
        Pframe=0;
    end
    zeta_fr_old=zeta_fr;
    qk2 = 1- Plocal.*Pglob*Pframe;  % estimate prob of speech absence
    qk2= min(0.95,qk2);
    qk = [qk2; flipud(qk2(2:len2-1))];
    
      
end
    
%----------------------------------------------
function y=smoothing (x,N);

len=length(x);
win=hanning(2*N+1);
win1=win(1:N+1);
win2=win(N+2:2*N+1);

y1=filter(flipud(win1),[1],x);

x2=zeros(len,1);
x2(1:len-N)=x(N+1:len);

y2=filter(flipud(win2),[1],x2);

y=(y1+y2)/norm(win,2);

