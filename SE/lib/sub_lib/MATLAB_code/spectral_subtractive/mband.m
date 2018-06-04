function mband(infile,outfile,Nband,Freq_spacing); 

%  Implements the multi-band spectral subtraction algorithm [1].
% 
%  Usage:  mband(infile, outputfile,Nband,Freq_spacing)
%           
%         infile - noisy speech file in .wav format
%         outputFile - enhanced output file in .wav format
%         Nband - Number of frequency bands (recommended 4-8)
%         Freq_spacing - Type of frequency spacing for the bands, choices:
%                        'linear', 'log' and 'mel'
%
%  Example call:  mband('sp04_babble_sn10.wav','out_mband.wav',6,'linear');
%
%  References:
%   [1] Kamath, S. and Loizou, P. (2002). A multi-band spectral subtraction 
%       method for enhancing speech corrupted by colored noise. Proc. IEEE Int.
%       Conf. Acoust.,Speech, Signal Processing
%   
% Authors: Sunil Kamath and Philipos C. Loizou
%
% Copyright (c) 2006 by Philipos C. Loizou
% $Revision: 0.0 $  $Date: 10/09/2006 $
%-------------------------------------------------------------------------


AVRGING=1; FRMSZ=20; OVLP=50; Noisefr=6; FLOOR=0.002; VAD=1;
% VAD -> Use voice activity detector, choices: 1 -to use VAD and 0 -otherwise
% AVRGING -> Do pre-processing (smoothing & averaging), choice: 1 -for pre-processing and 0 -otherwise, default=1
% FRMSZ -> Frame length in milli-seconds, default=20
% OVLP -> Window overlap in percent of frame size, default=50
% Noisefr -> Number of noise frames at beginning of file for noise spectrum estimate, default=6
% FLOOR -> Spectral floor, default=0.002

if nargin<4
    fprintf('Usage: mband(inFile.wav, outFile.wav, NumberBands, Freq_spacing)\n');
    fprintf('Type "help mband" for more help.\n')
    return;
end



[in,fs]=wavread(infile);


frmelen=floor(FRMSZ*fs/1000);           % Frame size in samples 
ovlplen=floor(frmelen*OVLP/100);        % Number of overlap samples
cmmnlen = frmelen-ovlplen;              % Number of common samples between adjacent frames
fftl = 2;
while fftl<frmelen
    fftl=fftl*2;
end

switch Freq_spacing
case {'linear','LINEAR'}
    bandsz(1) = floor(fftl/(2*Nband));
    for i=1:Nband
        lobin(i)=(i-1)*bandsz(1)+1;
        hibin(i)=lobin(i)+bandsz(1)-1;
        bandsz(i)=bandsz(1);        
    end
case {'log','LOG'}
    [lof,midf,hif]=estfilt1(Nband,fs);
    lobin = round(lof*fftl/fs)+1;
    hibin = round(hif*fftl/fs)+1;
    bandsz = hibin-lobin+1;
case {'mel','MEL'}
    [lof,midf,hif]=mel(Nband,0,fs/2);
    lobin = round(lof*fftl/fs)+1;
    hibin = round(hif*fftl/fs)+1;
    lobin(1)=1;
    hibin(end)=fftl/2+1;
    bandsz = hibin-lobin+1;
otherwise
    fprintf('Error in selecting frequency spacing, type "help mbss" for help.\n');
    return;
end
    

img=sqrt(-1);
% Calculate Hamming window
win=sqrt(hamming(frmelen));

% Estimate noise magnitude for first 'Noisefr' frames
% Estimated noise spectrum is stored in noise_spect
noise_pow=zeros(fftl,1);
j=1;
for k=1:Noisefr
    n_fft = fft(in(j:j+frmelen-1).* win, fftl);
    n_mag = abs(n_fft);
    n_ph = angle(n_fft);
    n_magsq = n_mag.^2;
    noise_pow = noise_pow + n_magsq;
    j = j + frmelen;
end
n_spect = sqrt(noise_pow/Noisefr);

% input to noise reduction part
x = in;

% Framing the signal with Window = 20ms and overlap = 10ms, 
% the output is a matrix with each column representing a frame
framed_x = frame(x,win,ovlplen,0,0);
[tmp, nframes] = size(framed_x);

%====Start Processing====
x_win = framed_x;

x_fft = fft(x_win,fftl);
x_mag = abs(x_fft);
x_ph = angle(x_fft);

if AVRGING
    % smooth the input spectrum
    filtb = [0.9 0.1];
    x_magsm(:,1) = filter(filtb, 1, x_mag(:,1));
    for i=2:nframes
        x_tmp1 = [x_mag(frmelen-ovlplen,i-1); x_mag(:,i)];
        x_tmp2 = filter(filtb, 1, x_tmp1);
        x_magsm(:,i) = x_tmp2(2:end);
    end
    
    % weighted spectral estimate 
    Wn2=0.09; Wn1=0.25; W0=0.32; W1=0.25; W2=0.09;
    x_magsm(:,1) = (W0*x_magsm(:,1)+W1*x_magsm(:,2)+W2*x_magsm(:,3));
    x_magsm(:,2) = (Wn1*x_magsm(:,1)+W0*x_magsm(:,2)+W1*x_magsm(:,3)+W2*x_magsm(:,4));
    for i=3:nframes-2
        x_magsm(:,i) = (Wn2*x_magsm(:,i-2)+Wn1*x_magsm(:,i-1)+W0*x_magsm(:,i)+W1*x_magsm(:,i+1)+W2*x_magsm(:,i+2));
    end
    x_magsm(:,nframes-1) = (Wn2*x_magsm(:,nframes-1-2)+Wn1*x_magsm(:,nframes-1-1)+W0*x_magsm(:,nframes-1)+W1*x_magsm(:,nframes));
    x_magsm(:,nframes) = (Wn2*x_magsm(:,nframes-2)+Wn1*x_magsm(:,nframes-1)+W0*x_magsm(:,nframes));
else
    x_magsm = x_mag;
end

%NOISE UPDATE DURING SILENCE FRAMES
if VAD
    [n_spect,state]=noiseupdt(x_magsm,n_spect,cmmnlen,nframes);
else
    for i=2:nframes
        n_spect(:,i)=n_spect(:,1);
    end
end

% Calculte the segmental SNR in each band -------------
start = lobin(1);
stop = hibin(1);
k=0;
for i=1:Nband-1
    for j=1:nframes
        SNR_x(i,j) = 10*log10(norm(x_magsm(start:stop,j),2)^2/norm(n_spect(start:stop,j),2)^2);
    end      
    start = lobin(i+1);
    stop = hibin(i+1);
    k=k+1;
end

for j=1:nframes
    SNR_x(k+1,j) = 10*log10(norm(x_magsm(start:fftl/2+1,j),2)^2/norm(n_spect(start:fftl/2+1,j),2)^2);
end
beta_x = berouti(SNR_x);

% ---------- START SUBTRACTION PROCEDURE --------------------------

sub_speech_x = zeros(fftl/2+1,nframes);
k=0;
for i=1:Nband-1   % channels 1 to Nband-1
    sub_speech=zeros(bandsz(i),1);
    start = lobin(i);
    stop = hibin(i);
    switch i
    case 1,
        for j=1:nframes
            n_spec_sq = n_spect(start:stop,j).^2;
            sub_speech(:,j) = x_magsm(start:stop,j).^2 - beta_x(i,j)*n_spec_sq;
        end
    otherwise
        for j=1:nframes
            n_spec_sq = n_spect(start:stop,j).^2;
            sub_speech(:,j) = x_magsm(start:stop,j).^2 - beta_x(i,j)*n_spec_sq*2.5;
        end
        k=k+1;
    end
    z=find(sub_speech <0);
    x_tmp = x_magsm(start:stop,:);
    if~isempty(z)
        sub_speech(z) = FLOOR*x_tmp(z).^2;
    end
    sub_speech = sub_speech+0.05*x_magsm(start:stop,:).^2;
    sub_speech_x(lobin(i):hibin(i),:) = sub_speech_x(lobin(i):hibin(i),:)+ sub_speech; 
    
end

% ----- now process last band ---------------------------
%
start = lobin(Nband);
stop = fftl/2+1;
clear FLOOR_n_matrix;
clear sub_speech;
for j=1:nframes
    n_spec_sq = n_spect(start:stop,j).^2;
    sub_speech(:,j) = x_magsm(start:stop,j).^2 - beta_x(Nband,j)*n_spec_sq*1.5;
end

z=find(sub_speech <0);
x_tmp = x_magsm(start:stop,:);
if~isempty(z)
    sub_speech(z) = FLOOR*x_tmp(z).^2;
end

sub_speech = sub_speech+0.01*x_magsm(start:stop,:).^2;
sub_speech_x(start:stop,:) = sub_speech_x(start:stop,:)+ sub_speech; 


% Reconstruct whole spectrum
sub_speech_x(fftl/2+2:fftl,:)=flipud(sub_speech_x(2:fftl/2,:));

%multiply the whole frame fft with the phase information
y1_fft = sub_speech_x.^(1/2).*(cos(x_ph) + img*sin(x_ph));

% to ensure a real signal
y1_fft(1,:) = real(y1_fft(1,:));
y1_fft(fftl/2+1,:) = real(y1_fft(fftl/2+1,:)); 

% take the IFFT 
y1_ifft = ifft(y1_fft);
y1_r = real(y1_ifft);

% overlap and add
y1(1:frmelen)=y1_r(1:frmelen,1);
start=frmelen-ovlplen+1;
mid=start+ovlplen-1;
stop=start+frmelen-1;
for i=2:nframes
    y1(start:mid) = y1(start:mid)+y1_r(1:ovlplen,i)';
    y1(mid+1:stop) = y1_r(ovlplen+1:frmelen,i);
    start = mid+1;
    mid=start+ovlplen-1;
    stop=start+frmelen-1;
end
out=y1;

wavwrite(out(1:length(x)),fs,16,outfile);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%CALLED FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function a=berouti(SNR)
[nbands,nframes]=size(SNR);
for i=1:nbands
    for j=1:nframes
        if SNR(i,j)>=-5.0 & SNR(i,j)<=20
            a(i,j)=4-SNR(i,j)*3/20; 
        elseif SNR(i,j)<-5.0
            a(i,j)=4.75;
        else
            a(i,j)=1;
        end  
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [n_spect,state]=noiseupdt(x_magsm,n_spect,cmmnlen,nframes)
SPEECH=1;	
SILENCE=0;
i=1;
x_var= x_magsm(:,i).^ 2;	
n_var= n_spect(:,i).^ 2;
rti= x_var./n_var - log10(x_var./n_var)-1;
judgevalue= mean(rti,1);
judgevalue1((i-1)*cmmnlen+1 : i*cmmnlen)= judgevalue;
if (judgevalue> 0.4)
    state((i-1)*cmmnlen+1 : i*cmmnlen)= SPEECH;
else
    state((i-1)*cmmnlen+1 : i*cmmnlen)= SILENCE;
    n_spect(:,i)= sqrt(0.9*n_spect(:,i).^2 + (1-0.9)*x_magsm(:,i).^ 2);
end
for i=2:nframes;
    x_var= x_magsm(:,i).^ 2;	
    n_var= n_spect(:,i-1).^ 2;
    rti= x_var./n_var - log10(x_var./n_var)-1;
    judgevalue= mean(rti,1);
    judgevalue1((i-1)*cmmnlen+1 : i*cmmnlen)= judgevalue;
    if (judgevalue> 0.45)
        state((i-1)*cmmnlen+1 : i*cmmnlen)= SPEECH;
        n_spect(:,i)=n_spect(:,i-1);
    else
        state((i-1)*cmmnlen+1 : i*cmmnlen)= SILENCE;
        n_spect(:,i)= sqrt(0.9*n_spect(:,i-1).^2 + (1-0.9)*x_magsm(:,i).^ 2);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [lower1,center,upper1]=estfilt1(nChannels,Srate)
% Copyright (c) 1996-97 by Philipos C. Loizou
FS=Srate/2;
UpperFreq=FS; LowFreq=1;
range=log10(UpperFreq/LowFreq);
interval=range/nChannels;
center=zeros(1,nChannels);
for i=1:nChannels  % ----- Figure out the center frequencies for all channels
    upper1(i)=LowFreq*10^(interval*i);
    lower1(i)=LowFreq*10^(interval*(i-1));
    center(i)=0.5*(upper1(i)+lower1(i));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [lower,center,upper] = mel(N,low,high)
% This function returns the lower, center and upper freqs
% of the filters equally spaced in mel-scale
% Input: N - number of filters
% 	 low - (left-edge) 3dB frequency of the first filter
%	 high - (right-edge) 3dB frequency of the last filter
%
% Copyright (c) 1996-97 by Philipos C. Loizou

ac=1100; fc=800;

LOW =ac*log(1+low/fc);
HIGH=ac*log(1+high/fc);
N1=N+1;
e1=exp(1);
fmel(1:N1)=LOW+[1:N1]*(HIGH-LOW)/N1;
cen2 = fc*(e1.^(fmel/ac)-1);
lower=zeros(1,N); upper=zeros(1,N); center=zeros(1,N);

lower(1:N)=cen2(1:N);
upper(1:N)=cen2(2:N+1);
center(1:N) = 0.5*(lower+upper);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%END OF MBSS CODE


function [fdata, fstart] = frame( sdata, window, frmshift, offset, trunc )
%  frame(): Place vector data into a matrix of frames
%
%  fdata = frame( sdata, window, [frmshift], [offset], [trunc] )
%
%  This function places sampled data in vector sdata into a matrix of
%  frame data.  The input sampled data sdata must be a vector.  The
%  window is a windowing vector (eg, hamming) applied to each frame of
%  sampled data and must be specified, because it defines the length
%  of each frame in samples.  The optional frmshift parameter
%  specifies the number of samples to shift between frames, and if not
%  specified defaults to the window size (which implies no overlap).
%  The optional offset specifies the offset from the first sample to
%  be used for processing.  If not specified, it is set to 0, which
%  means that the first sample of the sdata is the first sample of the
%  frame data.  The value of offset can be negative, in which case
%  initial padding of 0 samples is done.  The optional argument trunc
%  is a flag that specifies that sample data at the end should be 
%  truncated so that the last frame contains only valid data from the
%  samples and no zero padding is done at the end of the sample data
%  to fill a frame.  This means some sample data at the end will be
%  lost.  The default is not to truncate, but to pad with zero
%  samples until all sample data is represented in a frame at the end.
  

if nargin < 2
  error('frame: must specify sdata and window');
end


% check inputs
if any( size(sdata) == 1 )
  sdata = sdata(:)';
  ndata = length(sdata);
else
  error( 'frame: sdata must be vector');
end
  
if any( size(window) == 1 )
  window = window(:)';
  nwind = length(window);
else
  error('frame: window must be vector');
end

if nargin < 3
  frmshift = nwind;
elseif frmshift <=0
  error('frame: shift must be positive');
end

% resize sdata based on offset 
if nargin >= 4
  if offset > 0
    sdata = sdata( 1+offset : max(1+offset,ndata) );
    ndata = size(sdata,2);
  elseif offset < 0 
    sdata = [ zeros(1,abs(offset)) sdata];
    ndata = size(sdata,2);
  end
end

if nargin < 5
  trunc = 0;
end


% frame the data
if trunc
  nframes = floor( (ndata-nwind)/frmshift + 1 );
else
  nframes = ceil(ndata/frmshift);
end

tdata = zeros(nwind,nframes);
dowind = any( window ~= 1 );
ixstrt = 1;

for frm=1:nframes
  ixend = min( ndata, ixstrt+nwind-1 );
  ixlen = ixend-ixstrt+1;
  tdata(1:ixlen,frm) = sdata(ixstrt:ixend)';
  fstart(frm) = ixstrt;
  ixstrt = ixstrt + frmshift;
end

if offset ~= 0 
  fstart = fstart + offset;
end

if dowind
  fdata = scale_mtx( tdata, window, 1 );
else
  fdata = tdata;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
function fdata = scale_mtx( tdata, window, flag)

[dros,dcol] = size(tdata);
[wros,wcol] = size(window);
if wcol>wros
   window = window';
end
for i=1:dcol
   fdata(:,i) = tdata(:,i).*window;
end


