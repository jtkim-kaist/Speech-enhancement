function specsub_ns(filename,method,outfile)
%
%
%  Implements the basic spectral subtraction algorithm [1] with different
%  noise estimation algorithms specified by 'method'.
%  
%
%  Usage:  specsub_ns(noisyFile, method, outputFile)
%           
%         infile - noisy speech file in .wav format
%         outputFile - enhanced output file in .wav format
%         method - noise estimation algorithm:
%                  'martin'    = Martin''s minimum tracking algorithm [2]
%                  'mcra'      = Minimum controlled recursive average algorithm (Cohen) [3] 
%                  'mcra2'     = variant of Minimum controlled recursive
%                               average algorithm (Rangachari & Loizou) [4]
%                  'imcra'     = improved Minimum controlled recursive
%                               average algorithm (Cohen) [5]
%                  'doblinger' = continuous spectral minimum tracking(Doblinger) [6] 
%                  'hirsch'    = weighted spectral average (Hirsch &
%                               Ehrilcher) [7]
%                  'conn_freq' = connected frequency regions (Sorensen &
%                               Andersen) [8]
%   
%
%  Example call:  specsub_ns('sp04_babble_sn10.wav','mcra2','out_ss_mcra2.wav');
%
%  References:
%   [1] Berouti, M., Schwartz, M., and Makhoul, J. (1979). Enhancement of speech 
%       corrupted by acoustic noise. Proc. IEEE Int. Conf. Acoust., Speech, 
%       Signal Processing, 208-211.
%   [2] Martin, R. (2001). Noise power spectral density estimation based on optimal
%       smoothing and minimum statistics. IEEE Transactions on Speech and Audio 
%       Processing, 9(5), 504-512.
%   [3] Cohen, I. (2002). Noise estimation by minima controlled recursive averaging 
%       for robust speech enhancement. IEEE Signal Processing Letters,
%       9(1), 12-15.
%   [4] Rangachari, S. and Loizou, P. (2006). A noise estimation algorithm  for 
%       highly nonstationary environments. Speech Communication, 28,
%       220-231.
%   [5] Cohen, I. (2003). Noise spectrum estimation in adverse environments: 
%       Improved minima controlled recursive averaging. IEEE Transactions on Speech 
%       and Audio Processing, 11(5), 466-475.
%   [6] Doblinger, G. (1995). Computationally efficient speech enhancement by 
%       spectral minima tracking in subbands. Proc. Eurospeech, 2, 1513-1516.
%   [7] Hirsch, H. and Ehrlicher, C. (1995). Noise estimation techniques for robust 
%       speech recognition. Proc. IEEE Int. Conf. Acoust. , Speech, Signal 
%       Processing, 153-156.
%   [8] Sorensen, K. and Andersen, S. (2005). Speech enhancement with natural 
%       sounding residual noise based on connected time-frequency speech presence 
%       regions. EURASIP J. Appl. Signal Process., 18, 2954-2964.
%   
% Authors: Sundar Rangachari, Yang Lu and Philipos C. Loizou
%
% Copyright (c) 2006 by Philipos C. Loizou
% $Revision: 0.0 $  $Date: 10/09/2006 $
%-------------------------------------------------------------------------


if nargin<3
    fprintf('Usage: specsub_ns(infile.wav,method,outfile.wav) \n');
    fprintf('  where ''method'' is one of the following:\n');
    fprintf('   martin    = Martin''s minimum tracking algorithm\n');
    fprintf('   mcra      = Minimum controlled recursive average algorithm (Cohen) \n')
    fprintf('   mcra2     = variant of Minimum controlled recursive average algorithm (Rangachari & Loizou)\n')
    fprintf('   imcra     = improved Minimum controlled recursive average algorithm (Cohen) \n');
    fprintf('   doblinger = continuous spectral minimum tracking (Doblinger) \n');
    fprintf('   hirsch    = weighted spectral average (Hirsch & Ehrilcher) )\n'); 
    fprintf('   conn_freq = connected frequency regions (Sorensen & Andersen)\n');
    fprintf('\n For more help, type: help specsub_ns\n\n');
    return;
end


[x,Srate,nbits]=wavread(filename);


% =============== Initialize variables ===============
%

len=floor(20*Srate/1000); % Frame size in samples
if rem(len,2)==1, len=len+1; end;
PERC=50; % window overlap in percent of frame size
len1=floor(len*PERC/100);
len2=len-len1; 


alpha=2.0; % power exponent
FLOOR=0.002;
win=hamming(len); %tukey(len,PERC);  % define window


%--- allocate memory and initialize various variables
   
k=1;
nFFT=2*len;
img=sqrt(-1);
x_old=zeros(len1,1);
Nframes=floor(length(x)/len2)-1;
xfinal=zeros(Nframes*len2,1);

%===============================  Start Processing =======================================================
%
for n=1:Nframes 
   
   insign=win.*x(k:k+len-1);     %Windowing  
   spec=fft(insign,nFFT);     %compute fourier transform of a frame
   sig=abs(spec); % compute the magnitude
   ns_ps=sig.^2;
   
   % ----------------- estimate/update noise psd --------------
   if n == 1
         parameters = initialise_parameters(ns_ps,Srate,method);   
    else
        parameters = noise_estimation(ns_ps,method,parameters);
   end
    
   noise_ps = parameters.noise_ps;
   noise_mu=sqrt(noise_ps);  % magnitude spectrum
   % ---------------------------------------------------------
   
    %save the phase information for each frame.
    theta=angle(spec);  
   
   SNRseg=10*log10(norm(sig,2)^2/norm(noise_mu,2)^2);
   
   if alpha==1.0
      beta=berouti1(SNRseg);
   else
     beta=berouti(SNRseg);
  end
   
   
   %&&&&&&&&&
   sub_speech=sig.^alpha - beta*noise_mu.^alpha;
   diffw = sub_speech-FLOOR*noise_mu.^alpha;
   
   % Floor negative components
   z=find(diffw <0);  
   if~isempty(z)
      sub_speech(z)=FLOOR*noise_mu(z).^alpha;
   end
   
    
   sub_speech(nFFT/2+2:nFFT)=flipud(sub_speech(2:nFFT/2));  % to ensure conjugate symmetry for real reconstruction
   %multiply the whole frame fft with the phase information
   x_phase=(sub_speech.^(1/alpha)).*(cos(theta)+img*(sin(theta)));
  
   
   % take the IFFT 
   xi=real(ifft(x_phase));         
 
  % --- Overlap and add ---------------
  % 
  xfinal(k:k+len2-1)=x_old+xi(1:len1);
  x_old=xi(1+len1:len);
  
 k=k+len2;
end
%========================================================================================

wavwrite(xfinal,Srate,16,outfile);

%-------------------------------- E N D --------------------------------------
function a=berouti1(SNR)

if SNR>=-5.0 & SNR<=20
   a=3-SNR*2/20;
else
   
  if SNR<-5.0
   a=4;
  end

  if SNR>20
    a=1;
  end
  
end

function a=berouti(SNR)

if SNR>=-5.0 & SNR<=20
   a=4-SNR*3/20; 
else
   
  if SNR<-5.0
   a=5;
  end

  if SNR>20
    a=1;
  end
  
end
