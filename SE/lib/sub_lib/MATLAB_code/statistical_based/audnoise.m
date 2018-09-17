function audnoise(ns_file,outfile)

%
%  Implements the audible-noise suppression algorithm [1].
% 
%  Usage:  audnoise(noisyFile, outputFile)
%           
%         infile - noisy speech file in .wav format
%         outputFile - enhanced output file in .wav format
%
%  It runs 2 iterations, but one could change the number of iterations by
%  modifying accordingly the variable iter_num on line 33.
%
%  Example call:  audnoise('sp04_babble_sn10.wav','out_aud.wav');
%
%  References:
%   [1] Tsoukalas, D. E., Mourjopoulos, J. N., and Kokkinakis, G. (1997). Speech 
%       enhancement based on audible noise suppression. IEEE Trans. on Speech and 
%       Audio Processing, 5(6), 497-514.
%   
% Authors: Yi Hu and Philipos C. Loizou
%
% Copyright (c) 2006 by Philipos C. Loizou
% $Revision: 0.0 $  $Date: 10/09/2006 $
%-------------------------------------------------------------------------

if nargin<2
   fprintf('Usage: audnoise(noisyfile.wav,outFile.wav) \n\n');
   return;
end


iter_num=2;  % number of iterations
NF_SABSENT= 6;
%this is the number of speech-absent frames to estimate the initial
%noise power spectrum

[nsdata, Fs, bits]= wavread( ns_file);	%nsdata is a column vector

aa=0.98;
mu=0.98;
eta=0.15; 

nwind= floor( 20* Fs/ 1000);	%this corresponds to 20ms window
if rem( nwind, 2)~= 0 nwind= nwind+ 1; end	%made window length even
noverlap= nwind/ 2;
w= hamming( nwind);
rowindex= ( 1: nwind)';

%we assume the first NF_SABSENT frames are speech absent, we use them to estimate the noise power spectrum
noisedata= nsdata( 1: nwind* NF_SABSENT);	noise_colindex= 1+ ( 0: NF_SABSENT- 1)* nwind;
noisematrixdata = zeros( nwind, NF_SABSENT);
noisematrixdata( :)= noisedata( ...
    rowindex( :, ones(1, NF_SABSENT))+ noise_colindex( ones( nwind, 1), :)- 1);
noisematrixdata= noisematrixdata.* w( :, ones( 1, NF_SABSENT)) ;	%WINDOWING NOISE DATA
noise_ps= mean( (abs( fft( noisematrixdata))).^ 2, 2); %NOTE!!!! it is a column vector

% ----- estimate noise in CBs ------------------
%
noise_b=zeros(nwind/2+1,1);
[CB_FREQ_INDICES]=find_CB_FREQ_INDICES(Fs,nwind,16,nwind/2);

for i = 1:length(CB_FREQ_INDICES)
    noise_b(CB_FREQ_INDICES{i})=ones(size(CB_FREQ_INDICES{i},2),1)*mean(noise_ps(CB_FREQ_INDICES{i}));
end
noise_b1=[noise_b; fliplr(noise_b(2:nwind/2))];

nslide= nwind- noverlap;

x= nsdata;
nx= length( x);	ncol= fix(( nx- noverlap)/ nslide);
colindex = 1 + (0: (ncol- 1))* nslide;
if nx< (nwind + colindex(ncol) - 1)
    x(nx+ 1: nwind+ colindex(ncol) - 1) = ...
        rand( nwind+ colindex( ncol)- 1- nx, 1)* (2^ (-15));   % zero-padding
end

es_old= zeros( noverlap, 1);
%es_old is actually the second half of the previous enhanced speech frame,
%it is used for overlap-add

for k= 1: ncol

    y= x( colindex( k): colindex( k)+ nwind- 1);
    y= y.* w;	%WINDOWING NOISY SPEECH DATA

    y_spec= fft( y);	y_specmag= abs( y_spec);	y_specang= angle( y_spec);
    %they are the frequency spectrum, spectrum magnitude and spectrum phase, respectively

    y_ps= y_specmag.^ 2;	%power spectrum of noisy speech
    y_ps1=y_ps(1:nwind/2+1);
    
    % ====start of vad ===    
    gammak=min(y_ps./noise_ps,40);  % post SNR
    if k==1
        ksi=aa+(1-aa)*max(gammak-1,0);
    else
        ksi=aa*Xk_prev./noise_ps + (1-aa)*max(gammak-1,0);     % a priori SNR
    end

    log_sigma_k= gammak.* ksi./ (1+ ksi)- log(1+ ksi);    
    vad_decision= sum( log_sigma_k)/ nwind;    
    if (vad_decision < eta) 
        % noise only frame found
        noise_ps= mu* noise_ps+ (1- mu)* y_ps;
    end
    
    
    for i = 1:length(CB_FREQ_INDICES)
        noise_b(CB_FREQ_INDICES{i})=...
            ones(size(CB_FREQ_INDICES{i},2),1)*mean(noise_ps(CB_FREQ_INDICES{i}));
    end
    
    % ===end of vad===

    x_cons1=max(y_ps-noise_ps,0.001);  
    % conservative estimate of x from power spectral subtraction
    x_cons = x_cons1(1:nwind/2+1);

    % --- Estimate masking thresholds iteratively (as per page 505) ----
    %
    Tk0=mask(x_cons,nwind,Fs,16);
    Xp=y_ps1;
    for j=1:iter_num
        ab = noise_b+(noise_b.^2)./Tk0;  % Eq. 41
        Xp=(Xp.^2)./(ab+Xp);             % Eq. 40
        Tk0=mask(Xp,nwind,Fs,16);
    end

    % --- Estimate alpha ------
    %
    alpha = (noise_b+Tk0).*(noise_b./Tk0);  
    % eq. 26 for Threshold (T) method with ni(b)=1

    % ---- Apply suppression rule --------------
    %
    H0 = (Xp./(alpha+Xp));
    H=[H0(1:nwind/2+1); flipud(H0(2:nwind/2))];

    x_hat = H.*y_spec;
    Xk_prev= abs( x_hat).^ 2;
    
    es_tmp=real(ifft(x_hat));

    % ---- Overlap and add ---------------

    es_data( colindex( k): colindex( k)+ nwind- 1)= [es_tmp( 1: noverlap)+ es_old;...
        es_tmp( noverlap+ 1: nwind)];
    %overlap-add
    es_old= es_tmp( nwind- noverlap+ 1: nwind);
end

wavwrite( es_data, Fs, bits, outfile);

%------------------------------------------------------

function [CB_FREQ_INDICES]=find_CB_FREQ_INDICES(Fs,dft_length,nbits,frame_overlap)
% This function is from Matlab STSA Toolbox for Audio Signal Noise Reduction
% Copyright (C) 2001  Patrick J. Wolfe 

freq_val = (0:Fs/dft_length:Fs/2)';
freq=freq_val;
crit_band_ends = [0;100;200;300;400;510;630;770;920;1080;1270;1480;1720;2000;2320;2700;3150;3700;4400;5300;6400;7700;9500;12000;15500;Inf];
imax = max(find(crit_band_ends < freq(end)));
num_bins = length(freq);
LIN_TO_BARK = zeros(imax,num_bins);
i = 1;
for j = 1:num_bins
    while ~((freq(j) >= crit_band_ends(i)) & (freq(j) < crit_band_ends(i+1))),i = i+1;end
    LIN_TO_BARK(i,j) = 1;
end
% Calculation of critical band frequency indices--i.e., which bins are in which critical band for i = 1:imax
for i=1:imax,
    CB_FREQ_INDICES{i} = find(LIN_TO_BARK(i,:));
end

