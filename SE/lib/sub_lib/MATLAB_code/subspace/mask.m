% Author: Patrick J. Wolfe
%         Signal Processing Group
%         Cambridge University Engineering Department
%         p.wolfe@ieee.org
% Johnston perceptual model initialisation
function M= mask( Sx, dft_length, Fs, nbits)

frame_overlap= dft_length/ 2;    
freq_val = (0:Fs/dft_length:Fs/2)';
half_lsb = (1/(2^nbits-1))^2/dft_length;

freq= freq_val;
thresh= half_lsb;
crit_band_ends = [0;100;200;300;400;510;630;770;920;1080;1270;...
        1480;1720;2000;2320;2700;3150;3700;4400;5300;6400;7700;...
        9500;12000;15500;Inf];

% Maximum Bark frequency
%
imax = max(find(crit_band_ends < freq(end)));

% Normalised (to 0 dB) threshold of hearing values (Fletcher, 1929) 
% as used  by Johnston.  First and last thresholds are corresponding 
% critical band endpoint values, elsewhere means of interpolated 
% critical band endpoint threshold values are used.
%
abs_thr = 10.^([38;31;22;18.5;15.5;13;11;9.5;8.75;7.25;4.75;2.75;...
        1.5;0.5;0;0;0;0;2;7;12;15.5;18;24;29]./10);
ABSOLUTE_THRESH = thresh.*abs_thr(1:imax);

% Calculation of tone-masking-noise offset ratio in dB
%
OFFSET_RATIO_DB = 9+ (1:imax)';

% Initialisation of matrices for bark/linear frequency conversion
% (loop increments i to the proper critical band)
%
num_bins = length(freq);
LIN_TO_BARK = zeros(imax,num_bins);
i = 1;
for j = 1:num_bins
    while ~((freq(j) >= crit_band_ends(i)) & ...
            (freq(j) < crit_band_ends(i+1))),
        i = i+1;
    end
    LIN_TO_BARK(i,j) = 1;
end

% Calculation of spreading function (Schroeder et al., 82)

spreading_fcn = zeros(imax);
summ = 0.474:imax;
spread = 10.^((15.81+7.5.*summ-17.5.*sqrt(1+summ.^2))./10);
for i = 1:imax
    for j = 1:imax
        spreading_fcn(i,j) = spread(abs(j-i)+1);
    end
end

% Calculation of excitation pattern function

EX_PAT = spreading_fcn* LIN_TO_BARK;

% Calculation of DC gain due to spreading function

DC_GAIN = spreading_fcn* ones(imax,1);


%Sx = X.* conj(X);


C = EX_PAT* Sx;

% Calculation of spectral flatness measure SFM_dB
%
[num_bins num_frames] = size(Sx);
k = 1/num_bins;
SFM_dB = 10.*log10((prod(Sx).^k)./(k.*sum(Sx))+ eps);

% Calculation of tonality coefficient and masked threshold offset
%
alpha = min(1,SFM_dB./-60);
O_dB = OFFSET_RATIO_DB(:,ones(1,num_frames)).*...
    alpha(ones(length(OFFSET_RATIO_DB),1),:) + 5.5;

% Threshold calculation and renormalisation, accounting for absolute 
% thresholds

T = C./10.^(O_dB./10);
T = T./DC_GAIN(:,ones(1,num_frames));
T = max( T, ABSOLUTE_THRESH(:, ones(1, num_frames)));

% Reconversion to linear frequency scale 

%M = 1.* sqrt((LIN_TO_BARK')*T);
M= LIN_TO_BARK'* T;