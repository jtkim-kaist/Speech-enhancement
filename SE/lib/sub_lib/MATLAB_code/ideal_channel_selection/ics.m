function ics(noisefile, clfile, outfile, nsnr, thrd)

% noisefile - name of masker file
% clfile - name of clean sentence file
% outfile - name of output (processed) file
% nsnr is the overall input SNR (in dB) for noisy file
% thrd is the threshold for binary masking (in dB)
%
%  Copyright (c) 2011 by Philipos C. Loizou


if nargin<5
    fprintf('ERROR Usage: ics (MaskerFile,CleanFile,OutputFile,SNR,Threshold)\n');
    return;
end


[cl,fs,nb] = wavread(clfile);
cl = cl -mean(cl);

[n0,fs2,nb] = wavread(noisefile); 
if fs2~=fs
    error('Sampling frequency of masker file does not match that of target file');
end

n = n0(1:length(cl));
n = n - mean(n);


%--

%----scale the noise file to get required SNR------------
%
se=norm(cl,2)^2; %... signal energy
nsc=se/(10^(nsnr/10));
ne=norm(n,2)^2;  % noise energy
n=sqrt(nsc/ne)*n; % scale noise energy to get required SNR
ne=norm(n,2)^2;
%fprintf('Estimated SNR=%f\n',10*log10(se/ne));
x = cl + n;    % the noisy signal

ind1 = find(clfile == '.');
wavwrite(x,fs,16,[clfile(1:ind1-1) '-noisy.wav']); % save noisy signal

% =============== Initialize variables ===============
%
len = floor(20*fs/1000); % Frame size in samples
if rem(len,2)==1, len=len+1; end;
PERC = 50; % window overlap in percent of frame size
len1 = floor(len*PERC/100);
len2 = len-len1;

win = hanning(len);
nFFT = len;
nFFT2=round(nFFT/2);
%--- allocate memory and initialize various variables

Nframes = floor(length(x)/len2)-1;
xfinal = zeros(Nframes*len2,1);

%  masking threshold

thrd = 10^(thrd/10);

%===========  Start Processing    ============
k = 1;
x_old = zeros(len1,1);

BMask=zeros(Nframes,nFFT2);

for i = 1:Nframes

    insign = win.*x(k:k+len-1);  % noisy signal
    cl_sign = win.*cl(k:k+len-1);  % clean signal
    tn = win.*n(k:k+len-1);      % noise (masker)
   
    %--- Take fourier transform of  frame

    spec = fft(insign,nFFT);  % noisy speech spectrum
    sig = abs(spec); % compute the magnitude
    sig2 = sig.^2;

    cl_spec = fft(cl_sign,nFFT);  % clean speech spectrum
    cl_sig = abs(cl_spec); % compute the magnitude
    cl_sig2 = cl_sig.^2;

    tn_spec = fft(tn,nFFT);
    tn_sig = abs(tn_spec); % compute the magnitude
    tn_sig2 = tn_sig.^2;


    ksi_IEC = cl_sig2./tn_sig2;  % True instantaneous SNR
    %ksi_IEC = cl_sig2./sig2;  % True instantaneous SNR

    % binary masking
    hw = ones(len1,1);
    ind = find(ksi_IEC(1:len1)<thrd);
    hw(ind) = 0;
    BMask(i,1:nFFT2)=1-hw(1:nFFT2); 

    hw = [hw ;hw(end:-1:1)];
    xi_w = ifft( hw .* spec);   % synthesize binary-masked signal
    %xi_w = ifft( hw .* tn_spec);   % synthesize  signal by modulating mask by noise 
    xi_w = real( xi_w);

    % --- Overlap and add ---------------
    %
    xfinal(k:k+ len2-1) = x_old+ xi_w(1:len1);
    x_old = xi_w(len1+ 1: len);
    k=k+len2;
end
%========================================================================================


%xfinal = xfinal/32768;
if max(abs(xfinal))>1.0; %32768
    xfinal = xfinal*0.6/max(abs(xfinal));
    fprintf('Max amplitude exceeded 1 for file %s\n',clfile);
end


wavwrite(xfinal,fs,16,outfile);

imagesc(BMask)

handle = imagesc([0 length(cl)/fs],[0 fs/2],BMask'); 
axis('xy');
colormap('gray')
xlabel('Time (secs)'); ylabel ('Freq. (Hz)');
axis([0 length(cl)/fs  0 fs/2]);

    

return;

