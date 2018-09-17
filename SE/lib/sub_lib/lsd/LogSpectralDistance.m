function LSD=LogSpectralDistance(Clean,Noisy,fs,RS,Range)

%LSD=LOGSPECTRALDISTANCE(CLEAN,NOISY,FS,RS,RANGE)
% Calculates the average log-spectral distance between CLEAN and NOISY
% signals. Frames are 25ms long with 60 percent (15ms) overlap, hamming
% windowed. RS is the remove silence option (default: 0) if 1, the program
% uses a Voice Activity Detector to find non-speech frames and eliminates
% them from calculation of LSD. FS is the sampling frequency (Default
% 10KHz). RANGE is the frequency range used in the calculation of LSD
% (default: [0 FS/2]) it is a two-element vector the first element should
% be smaller than the second one and non of them should be greater than
% FS/2.

if nargin<3
    fs=10000;
end
if nargin<4
    RS=0;
end
if nargin<5
    Range=[0,fs/2];
end
if RS==1
    [Clean, Noisy]=RemoveSilence(Clean,Noisy,fs);
end

Len=min(length(Clean),length(Noisy));
Clean=Clean(1:Len);
Noisy=Noisy(1:Len);
Clean=Clean./sqrt(sum(Clean.^2));
Noisy=Noisy./sqrt(sum(Noisy.^2));

W=round(.025*fs);
SP=.4;
CL=abs(mySpecgram(Clean,W,SP));
NO=abs(mySpecgram(Noisy,W,SP));
nfft2=size(CL,1);
N=min(size(CL,2),size(NO,2)); %Number of frames
RangeBin=freq2bin(Range,fs/2,nfft2);
RangeBin=RangeBin(1):RangeBin(2);
LSD=mean(sqrt(mean((log(CL(RangeBin,1:N))-log(NO(RangeBin,1:N))).^2)));



function [x2, y2]=RemoveSilence(x1,y1,fs,IS)

% Remove Silence
% [X2, Y2]=REMOVESILENCE(X1,Y1,FS,IS)
% This function removes the silence parts from the signals X1 and Y1 and
% returns the corresponding wave forms. Y1 is normally a modified (noisy or
% enhanced) version of X1. The silence frames are detected using X1 which
% is supposed to be the clean signal. For this purpose a Voice Activity
% Detector (VAD) is used. The use of this function is for evaluation of the
% speech quality (e.g. SNR) at speech active frame only. FS is tha sampling
% frequency. IS is the initial silence duration (the defaul value) which is
% used to model the noise template. the default value of IS is 0.25 sec.
% Date: Feb-05
% Author: Esfandiar Zavarehei

if (nargin<4)
    IS=.25; %seconds
end
% Window size and overlap and other initialization values
W=.025*fs;
SP=.010*fs/W;
wnd=hamming(W);

NIS=fix((IS*fs-W)/(SP*W) +1);%number of initial silence segments

Y1=segment(y1,W,SP,wnd);
Y1=fft(Y1);
Y1=Y1(1:fix(end/2)+1,:);
Y1P=angle(Y1);
Y1=abs(Y1);
X1=segment(x1,W,SP,wnd);
X1=fft(X1);
X1=X1(1:fix(end/2)+1,:);
X1P=angle(X1);
X1=abs(X1);
NumOfFrames=min(size(X1,2),size(Y1,2));

NoiseLength=15;
N=mean(X1(:,1:NIS)')'; %initial Noise Power Spectrum mean

%Find the non-speech frames of X1
for i=1:NumOfFrames
    if i<=NIS
        SpeechFlag(i)=0;
        NoiseCounter=100;
        Dist=.1;
    else
        [NoiseFlag, SpeechFlag(i), NoiseCounter, Dist]=vad(X1(:,i),N,NoiseCounter,2.5,fix(.08*10000/(SP*W))); %Magnitude Spectrum Distance VAD
    end
    if SpeechFlag(i)==0 & i>NIS
        N=(NoiseLength*N+X1(:,i))/(NoiseLength+1); %Update and smooth noise mean
    end
end
SpeechIndx=find(SpeechFlag==1);

x2=OverlapAdd2(X1(:,SpeechIndx),X1P(:,SpeechIndx),W,SP*W);
y2=OverlapAdd2(Y1(:,SpeechIndx),Y1P(:,SpeechIndx),W,SP*W);


function [NoiseFlag, SpeechFlag, NoiseCounter, Dist]=vad(signal,noise,NoiseCounter,NoiseMargin,Hangover)

%[NOISEFLAG, SPEECHFLAG, NOISECOUNTER, DIST]=vad(SIGNAL,NOISE,NOISECOUNTER,NOISEMARGIN,HANGOVER)
%Spectral Distance Voice Activity Detector
%SIGNAL is the the current frames magnitude spectrum which is to labeld as
%noise or speech, NOISE is noise magnitude spectrum template (estimation),
%NOISECOUNTER is the number of imediate previous noise frames, NOISEMARGIN
%(default 3)is the spectral distance threshold. HANGOVER ( default 8 )is
%the number of noise segments after which the SPEECHFLAG is reset (goes to
%zero). NOISEFLAG is set to one if the the segment is labeld as noise
%NOISECOUNTER returns the number of previous noise segments, this value is
%reset (to zero) whenever a speech segment is detected. DIST is the
%spectral distance.
%Saeed Vaseghi
%edited by Esfandiar Zavarehei
%Sep-04

if nargin<4
    NoiseMargin=3;
end
if nargin<5
    Hangover=8;
end
if nargin<3
    NoiseCounter=0;
end

FreqResol=length(signal);

SpectralDist= 20*(log10(signal)-log10(noise));
SpectralDist(find(SpectralDist<0))=0;

Dist=mean((SpectralDist));
if (Dist < NoiseMargin)
    NoiseFlag=1;
    NoiseCounter=NoiseCounter+1;
else
    NoiseFlag=0;
    NoiseCounter=0;
end

% Detect noise only periods and attenuate the signal
if (NoiseCounter > Hangover)
    SpeechFlag=0;
else
    SpeechFlag=1;
end

function Seg=segment(signal,W,SP,Window)

% SEGMENT chops a signal to overlapping windowed segments
% A= SEGMENT(X,W,SP,WIN) returns a matrix which its columns are segmented
% and windowed frames of the input one dimentional signal, X. W is the
% number of samples per window, default value W=256. SP is the shift
% percentage, default value SP=0.4. WIN is the window that is multiplied by
% each segment and its length should be W. the default window is hamming
% window.
% 06-Sep-04
% Esfandiar Zavarehei

if nargin<3
    SP=.4;
end
if nargin<2
    W=256;
end
if nargin<4
    Window=hamming(W);
end
Window=Window(:); %make it a column vector

L=length(signal);
SP=fix(W.*SP);
N=fix((L-W)/SP +1); %number of segments

Index=(repmat(1:W,N,1)+repmat((0:(N-1))'*SP,1,W))';
hw=repmat(Window,1,N);
Seg=signal(Index).*hw;

function b=freq2bin(f,fmax,W,mode)
% B=FREQ2BIN(F,FMAX,NFFT)
% Returns the bin number, B, corresponding to the frequency F, when maximum
% frequency is FMAX and the number of fft samples is NFFT
% Use F=FREQ2BIN(B,FMAX,NFFT,'EX') to get the exact value (not round)
% Date: May 2005
% Author: Esfandiar Zavarehei
step=fmax/(W-1);
if (nargin>3) & (strcmp(upper(mode),'EX'))
    b=(f/step)+1;
else
    b=round(f/step)+1;
end

    
function X=mySpecgram(x,W,SP);

if nargin<3
    SP=.4;
end
if nargin<2
    W=250;
end
X=segment(x,W,SP);
X=fft(X);
X=X(1:fix(end/2)+1,:);
