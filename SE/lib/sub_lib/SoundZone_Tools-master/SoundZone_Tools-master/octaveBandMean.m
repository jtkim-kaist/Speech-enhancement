function [ magSpect_oct, freqVec_oct ] = octaveBandMean( magSpect, freqVec, octSpace, centerFreq )
% Given a magnitude spectrum this function will calculate the average (single, third, nth) octave band magnitude
% 
% Syntax:	[ MAGSPECT_OCT, FREQVEC_OCT ] = OCTAVEBANDMEAN( MAGSPECT, FREQVEC, OCTSPACE, CENTERFREQ )
% 
% Inputs: 
% 	magSpect - An arbitrary magnitude spectrum as a vector
% 	freqVec - The corresponding frequencies for each magnitude value
% 	octSpace - A single value between 0 and 1 specifying the octave spacing
% 	centerFreq - The center frequency for the octave band
% 
% Outputs: 
% 	magSpect_oct - The magnitude spectrum averaged per octave band
% 	freqVec_oct - The corresponding frequencies for the output magnitude vector
% 
% See also: 

% Author: Jacob Donley
% University of Wollongong
% Email: jrd089@uowmail.edu.au
% Copyright: Jacob Donley 2016, 2017
% Date: 8 July 2016
% Revision: 0.2 (29 March 2017)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 4
    centerFreq=1e3;
end
if nargin < 3
    octSpace = 1/3;
end

if size(magSpect,2) > size(magSpect,1), magSpect = magSpect.'; end
if size(freqVec,2) > size(freqVec,1), freqVec = freqVec.'; end

freqVec_ = nonzeros(freqVec); %remove zero values

n = floor(log(freqVec_( 1 )/centerFreq)/log(2)/octSpace): ...
    floor(log(freqVec_(end)/centerFreq)/log(2)/octSpace);

freqVec_oct = centerFreq * (2 .^ (n*octSpace));
if freqVec_oct(end) ~= freqVec_(end)
    % This may happen if there is no octave band that is centered at the
    % Nyquist frequency (fs/2)
    freqVec_oct(end+1) = freqVec_(end); % So we find an average at the 
    % Nyquist frequency that is of the same width as the other octave bands
    warning(['The sampling frequency used does not have a Nyquist ' ...
    'frequency that, after division by the centre frequency, is a ' ...
    'power of two. ' ...
    'i.e. (' num2str(freqVec_(end)*2) '/2)/' num2str(centerFreq) ' is ' ... 
    'not a power of two.']);
end
fd = 2^(octSpace/2);
fupper = freqVec_oct * fd;
flower = freqVec_oct / fd;

octBands = [flower' fupper'];

[~, oBandInds] = min(abs( ...
    repmat(freqVec,1,2,length(octBands)) ...
    - repmat(permute(octBands,[3 2 1]),length(freqVec),1,1)  ));

oBandInds = permute(oBandInds, [3 2 1]);

magSpect_oct = zeros(1,length(freqVec_oct));
for band = 1:length(freqVec_oct)
    magSpect_oct(band+1) = mean(  ...
        magSpect( oBandInds(band,1):oBandInds(band,2) ) );
end

magSpect_oct(1) = magSpect(1);
freqVec_oct  = [0 freqVec_oct];
end

