function [ y ] = sineSweepLin( f_start, f_end, dur, fs )
% Synthesize a linear sine sweep
%   Detailed explanation goes here
t=0:1/fs:dur-1/fs;
f=linspace(f_start,(f_end+f_start)/2,length(t));
y=sin(2*pi*f.*t);
end

