function ics_competing_talker(filename, clfile, t_outfile, m_outfile,thrd)

% filename - mixture filename
% clfile - clean target filename
% t_outfile - output file: Target
% m_outfile - output file: Competing talker
% thrd - SNR threshold in dB
%
%   Copyright (c) 2011 by Philipos C. Loizou


[x,Srate,nb] = wavread(filename); % mixture signal
cl = wavread(clfile); % clean speech signal

% =============== Initialize variables ===============
%
len = floor(20*Srate/1000); % Frame size in samples
if rem(len,2)==1, len=len+1; end;
PERC = 50; % window overlap in percent of frame size
len1 = floor(len*PERC/100);
len2 = len-len1;
win = hanning(len);
nFFT = len;

%--- allocate memory and initialize various variables

x_old = zeros(len1,1);
Nframes = floor(length(x)/len2)-1;
xfinal = zeros(Nframes*len2,1); % for target signal
x_old_m = zeros(len1,1);
xfinal_m = zeros(Nframes*len2,1); % for masker signal


%% ===========  Start Processing    ============
k = 1;
thrd = 10^(thrd/10);
m=1;
nFFT2=floor(nFFT/2);

for n = 1:Nframes
    insign = win.*x(k:k+len-1);  % mixture
    cl_sign = win.*cl(k:k+len-1);  % clean target
    tn = insign - cl_sign;   % masker signal
    
    %--- Take fourier transform of  frames

    spec = fft(insign,nFFT);
    sig = abs(spec); % compute the magnitude
    sig2 = sig.^2;
  
    cl_spec = fft(cl_sign,nFFT);
    cl_sig = abs(cl_spec); % compute the magnitude
    cl_sig2 = cl_sig.^2;

    tn_spec = fft(tn,nFFT);
    tn_sig = abs(tn_spec); % compute the magnitude
    tn_sig2 = tn_sig.^2;

    %% Compute IDEAL BINARY MASK
    %
    ksi_IEC = cl_sig2./tn_sig2;  % Ideal ksi

    hw_true = zeros(len,1);
    indp = find(ksi_IEC>= thrd); % ideal IBM
    hw_true(indp)=1;  % true ksi
    % ideal IBM to recover the target signal 

    hw_true_m = zeros(len,1);
    indp = find(ksi_IEC< thrd); % ideal IBM
    hw_true_m(indp)= 1;  % true ksi
    % complementary ideal IBM to recover the mask signal 

    
    x_hw_m = hw_true_m;
    x_hw = hw_true;

    

    % Synthesize target signal
    xi_w = ifft( x_hw .* spec);    %
    xi_w = real( xi_w);
    xfinal(k:k+ len1-1) = x_old+ xi_w(1:len1);
    x_old = xi_w( len2+ 1: len);

    % Synthesize masker signal
    xi_w_m = ifft( x_hw_m .* spec);   
    xi_w_m = real( xi_w_m);
    xfinal_m( k: k+ len1-1) = x_old_m+ xi_w_m( 1:len1);
    x_old_m = xi_w_m( len2+ 1: len);

    k = k+len2;
end

%%  =======================================================================


if max(abs(xfinal))>1
    xfinal = xfinal*0.6/max(abs(xfinal));
    fprintf('Max amplitude exceeded 32768 for file %s\n',filename);
end

wavwrite(xfinal, Srate, 16, t_outfile);


if max(abs(xfinal_m))>1
    xfinal_m = xfinal_m* 0.6/max(abs(xfinal_m));
    fprintf('Max amplitude exceeded 32768 for file %s\n',filename);
end

wavwrite(xfinal_m, Srate, 16, m_outfile);

