function  ics_masker_rule(filename, clfile, outfile)

% filename - noisy speech filename (mixture)
% clfile - clean speech filename
% outilfe - name of output file
%
% Implements the algorithm presented in:
% Kim, G. and Loizou, P. (2010). "A new binary mask based on noise
%   constraints for improved speech intelligibility," Proc. INTERSPEECH, 
%   Makuhari, Japan, pp. 1632-1635.
%
%  Copyright (c) 2011 by Philipos C. Loizou


if nargin<3
    fprintf('ERROR Usage: ics_masker_rule(MixtureFile,CleanFile,OutputFile)\n');
    return;
end

method = 'mcra2'; % noise-estimation algorithm (Rangachari & LOizou, Speech Comm., 2006)

[x, Srate, nb] = wavread(filename);  % noisy speech
[cl,Srate2,nb] = wavread(clfile);    % clean speech


% =============== Initialize variables ===============
%
len = floor(20*Srate/1000); % Frame size in samples
if rem(len,2)==1, len=len+1; end;
PERC = 50; % window overlap in percent of frame size
len1 = floor(len*PERC/100);
len2 = len-len1; 


win = hanning(len); 

% Noise magnitude calculations - assuming that the first 4 frames is
% noise/silence
nFFT = len;
%% Initialize noise estimate

noise_mean=zeros(nFFT,1);
j=1;
for k=1:4
   magn=abs(fft(win.*x(j:j+len-1),nFFT)).^2;
   noise_mean=noise_mean+magn;
   j=j+len;
end
noise_initial=noise_mean/4;
%%

%--- allocate memory and initialize various variables
   
x_old = zeros(len1,1);
Nframes = floor(length(x)/len2)-1;
xfinal = zeros(Nframes*len2,1);

 
%% ===========  initialize parameters   ============
k = 1;
aa = 0.98; % Decisioni-Directed
m=1; 
nFFT2=floor(nFFT/2);

%% Start processing
%
for n = 1:Nframes 
    insign = x(k:k+len-1).*win;  % noisy speech
    cl_sign =cl(k:k+len-1).*win;  % clean speech
    tn = insign - cl_sign;   % noise signal
    
    
    %--- Take fourier transform of  signals
   
    spec = fft(insign,nFFT);   
    sig = abs(spec); % compute the magnitude of noisy speech
    sig2 = sig.^2;
       
    
    cl_spec = fft(cl_sign,nFFT);   
    cl_sig = abs(cl_spec); % compute the magnitude of clean signal
    cl_sig2 = cl_sig.^2;
  
    
    tn_spec = fft(tn,nFFT);   
    tn_sig = abs(tn_spec); % compute the magnitude of masker
    tn_sig2 = tn_sig.^2;
   
     
    
%% Estimate noise spectrum
%
 % use noise estimation algorithm
    if n==1
        parameters = initialise_parameters(noise_initial,Srate,method);
        parameters = noise_estimation(noise_initial, method, parameters); 
    else
       parameters = noise_estimation(sig2, method, parameters); 
    end
    noise_mu2 = parameters.noise_ps;
   
    
%%       
    gammak = min(sig2./noise_mu2,20);
    if n==1
        
        ksi2 = aa+(1-aa)*max(gammak-1,0);
        ksi2 = max(ksi2,0.0025);
        
    else   
        ksi2 = aa*Xk_prev./noise_mu2 + (1-aa)*max(gammak-1,0); % Traditional a priori SNR in MMSE                    
        ksi2 = max(ksi2,0.0025);
    end
 
   
%% COMPUTE GAIN FUNCTION FOR NOISE SPECTRUM and target spectrum
% 
     x_hw_m = 1./sqrt(ksi2+1);   % Eq. 5 in Kim & Loizou, Proc. Interspeech, 2010
     
     x_hw = sqrt(ksi2./(ksi2+1));  %Eq 2   - square-root Wiener gain function
    
%% SAVE Enhanced magnitude spectrum needed for ksi estimation in next frame

    x_sig = sig.*x_hw;   
    Xk_prev = x_sig.^2;
    
       
%%  Estimate noise magnitude spectrum (Eq. 4)
% 
    mask_spec = sig .* x_hw_m;
    
    %% ========== Apply masker rule =====================
    %
    rpp = find(mask_spec> tn_sig);  % Eq 6 - keep over-estimates of noise spectrum
    hw_residual= zeros(len,1);
    hw_residual(rpp)=1;
    hw_residual(1)=1; hw_residual(nFFT2+1)=1;
     

%%    Synthesize signal
%

    x_hw_new = x_hw .* hw_residual;
    % x_hw_new = hw_residual;
    %x_hw_new = x_hw;   % use Wiener gain function (all channels)
    xi_w = ifft(x_hw_new .* spec);    % synthesize signal by keeping components satisfying Eq 6 
    xi_w = real( xi_w);
	
    % --- Overlap and add ---------------
    %
    xfinal(k:k+ len1-1) = x_old+ xi_w(1:len1);
    x_old = xi_w(len2+ 1: len);

   
    k = k+len2;
end

%%  =======================================================================
 



if max(abs(xfinal))>1
   xfinal = xfinal*0.6/max(abs(xfinal));
   fprintf('Max amplitude exceeded 1 for file %s\n',filename);   
end

wavwrite(xfinal,Srate,16,outfile);

