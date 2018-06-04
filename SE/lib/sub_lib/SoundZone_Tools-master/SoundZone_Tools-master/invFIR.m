function [ih]=invFIR(type,h,Nfft,Noct,L,range,reg,window)
% Design inverse filter (FIR) from mono or stereo impulse response
% ------------------------------------------------------------------------------ 
% description: design inverse filter (FIR) from mono or stereo impulse response
% ------------------------------------------------------------------------------
% inputs overview
% ---------------
% type   - 1. 'linphase': symmetric two-sided response compensating magnitude while maintaining original phase information
%          2. 'minphase': one-sided response compensating magnitude with minimal possible group delay
%          3. 'complex': asymmetric two-sided response compensating magnitude and phase
%         
% h      - mono or stereo impulse response (column vector)
% 
% Nfft   - FFT length for calculating inverse FIR
% 
% Noct   - optional fractional octave smoothing (e.g. Noct=3 => 1/3 octave smooth, Noct=0 => no smoothing)
% 
% L      - length of inverse filter (truncates Nfft-length filter to L)
% 
% range  - frequency range to be inverted (e.g. [32 16000] => 32 Hz to 16 kHz)
% 
% reg    - amount of regularization (in dB) inside (reg(1)) and outside (reg(2)) the specified range
%         (example: reg=[20 -6] => inverts frequency components within 'range' with a max. gain of 20 dB,
%                   while dampening frequencies outside 'range' by 6 dB)
% 
% window - window=1 applies a hanning window to the inverse filter
% -----------------------------------------------------------------------------------------------
% 
% Background information: 
% complex inversion of non-minimum phase impulse responses
% --------------------------------------------------------
% If an acoustical impulse response contains reflections there will be repeated similar magnitude characteristics during sound propagation.
% This causes the impulse response to consist of a maximum-phase and a minimum phase component.
% Expressed in trms of z-transformation the min.-phase are within the unit circle while the max.-phase component are outside.
% Those components can be seen as numerator coefficients of an digital FIR filter.
% Inversion turnes the numerator coefficients into denumerator coefficients and those outside the unit circle will make the resulting
% filter (which is now an IIR filter) unstable.
% Making use of the DFT sets |z|=1 and that means taht the region of convergence now includes the unit circle.
% Now the inverse filter is stable but non-causal (left handed part of response towards negative times).
% To compensate this, the resulting response is shifted in time to make the non-causal part causal.
% But the "true" inverse is still an infinite one but is represented by an finite (Nfft-long) approximation.
% Due to this fact and due to the periodic nature of the DFT, the Nfft-long "snapshot" of the true invese also contains 
% overlapping components from adjacents periodic repetitions (=> "time aliasing").
% Windowing the resulting response helps to suppress aliasing at the edges but does not guarantee that the complete response is aliasing-free.
% In fact inverting non-minimum phase responses will always cause time aliasing - the question is not "if at all" but "to which amount".
% Time-aliasing "limiters":
%     - use of short impulse responses to be inverted (=> windowing prior to inverse filter design)
%     - use of longer inverse filters (=> increasing FFT length)
%     - avoide inversion of high-Q (narrow-band spectral drips/peaks with high amplitde) spectral components (=> regularization, smoothing)
% In addition the parameters should be choosen to minimize the left-sided part of the filter response to minimize perceptual disturbing pre-ringing.
% 
% ----------------------------------------------------------------
% References:
% - papers of the AES (e.g. from A. Farina, P. Nelson, O. Kirkeby)
% - Oppenheim, Schafer "Discrete-Time Signal Processing"
% ----------------------------------------------------------------


fs=44100;
f1=range(1);
f2=range(2);
reg_in=reg(1);
reg_out=reg(2);
if window==1
    win=0.5*(1-cos(2*pi*(1:L)'/(L+1)));
else
    win=1;
end
% regularization
%---------------

% calculate 1/3 octave edges of regularization
if f1 > 0 && f2 < fs/2
    freq=(0:fs/(Nfft-1):fs/2)'; 
    f1e=f1-f1/3;
    f2e=f2+f2/3;
    if f1e < freq(1)
        f1e=f1;
        f1=f1+1;
    end
    if f2e > freq(end)
        f2e=f2+1;
    end
    % regularization B with 1/3 octave interpolated transient edges 
    B=interp1([0 f1e f1 f2 f2e freq(end)],[reg_out reg_out reg_in reg_in reg_out reg_out],freq,'pchip');
    B=10.^(-B./20); % from dB to linear
    B=vertcat(B,B(end:-1:1)); 
    b=ifft(B,'symmetric');
    b=circshift(b,Nfft/2);
    b=0.5*(1-cos(2*pi*(1:Nfft)'/(Nfft+1))).*b;
    b=minph(b); % make regularization minimum phase
    B=fft(b,Nfft);
else
    B=0;
end
%----------------------

% calculate inverse filter
if strcmp(type,'complex')==1
    H=fft(h(:,1),Nfft);
elseif strcmp(type,'linphase')==1 || strcmp(type,'minphase')==1
    H=abs(fft(h(:,1),Nfft));
end
if Noct > 0
    [H]=cmplxsmooth(H,Noct); % fractional octave smoothing
end
iH=conj(H)./((conj(H).*H)+(conj(B).*B)); % calculating regulated spectral inverse
ih=circshift(ifft(iH,'symmetric'),Nfft/2);
ih=win.*ih(end/2-L/2+1:end/2+L/2); % truncation to length L


% 2-channel case
%-------------------------------------------------
if size(h,2)==2
    if strcmp(type,'complex')==1
        H=fft(h(:,2),Nfft);
    elseif strcmp(type,'linphase')==1 || strcmp(type,'minphase')==1
        H=abs(fft(h(:,2),Nfft));
    end
    if Noct > 0
        [H]=cmplxsmooth(H,Noct);
    end
    iH=conj(H)./((conj(H).*H)+(conj(B).*B));
    ihr=circshift(ifft(iH,'symmetric'),Nfft/2);
    ihr=win.*ihr(end/2-L/2+1:end/2+L/2);
    ih=[ih ihr];
end

if strcmp(type,'minphase')==1
    ih=minph(ih);
end

% calculate minimum phase component of impulse response
function [h_min] = minph(h)
n = length(h);
h_cep = real(ifft(log(abs(fft(h(:,1))))));
odd = fix(rem(n,2));
wn = [1; 2*ones((n+odd)/2-1,1) ; ones(1-rem(n,2),1); zeros((n+odd)/2-1,1)];
h_min = zeros(size(h(:,1)));
h_min(:) = real(ifft(exp(fft(wn.*h_cep(:)))));
if size(h,2)==2
    h_cep = real(ifft(log(abs(fft(h(:,2))))));
    odd = fix(rem(n,2));
    wn = [1; 2*ones((n+odd)/2-1,1) ; ones(1-rem(n,2),1); zeros((n+odd)/2-1,1)];
    h_minr = zeros(size(h(:,2)));
    h_minr(:) = real(ifft(exp(fft(wn.*h_cep(:)))));
    h_min=[h_min h_minr];
end
