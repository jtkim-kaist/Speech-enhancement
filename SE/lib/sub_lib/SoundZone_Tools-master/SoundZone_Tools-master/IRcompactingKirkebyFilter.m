function [ filt ] = IRcompactingKirkebyFilter( ir, ir_len, f_band, fs, reg )
% Compacting Kirkeby Filter

% Regularisation parameter
ereg = epsreg(ir_len*fs,f_band,fs,reg);

% Time-packing filtering
H = fft(ir, ir_len*fs);

C = conj(H) ./ (conj(H).*H + ereg);

filt = ifft(C);

end

function ereg = epsreg(Nfft, f_band, fs, reg)
f1=f_band(1);
f2=f_band(2);
reg_in=reg(1);
reg_out=reg(2);
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
    B=interp1([0 f1e f1 f2 f2e freq(end)],[reg_out reg_out reg_in reg_in reg_out reg_out],freq,'PCHIP');
    B = db2mag(-B); % from dB to linear
    B=vertcat(B,B(end:-1:1));
    b=ifft(B,'symmetric');
    b=circshift(b,Nfft/2);
    b=0.5*(1-cos(2*pi*(1:Nfft)'/(Nfft+1))).*b;
    b=minph(b); % make regularization minimum phase
    B=fft(b,Nfft);
else
    B=0;
end
ereg = (conj(B).*B);
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
end