function wiener_iter(ns_file, es_file, iter_num)

%
%  Implements the basic iterative Wiener filtering algorithm [1].
% 
%  Usage:  wiener_iter(noisyFile, outputFile,NumberOfIterations)
%           
%         infile - noisy speech file in .wav format
%         outputFile - enhanced output file in .wav format
%         NumberOfIterations - number of iterations (recommended 2-4)
%         
%  Example call:  wiener_iter('sp04_babble_sn10.wav','out_wien.wav',3);
%
%  References:
%   [1] Lim, J. and Oppenheim, A. V. (1978). All-pole modeling of degraded speech. 
%       IEEE Trans. Acoust. , Speech, Signal Proc., ASSP-26(3), 197-210.
%   
% Authors: Yi Hu and Philipos C. Loizou
%
% Copyright (c) 2006 by Philipos C. Loizou
% $Revision: 0.0 $  $Date: 10/09/2006 $
%-------------------------------------------------------------------------

if nargin<3
   fprintf('Usage: wiener_iter(noisyfile.wav,outFile.wav,NumIter) \n\n');
   return;
end

NF_SABSENT= 6;
%this is the number of speech-absent frames to estimate the initial 
%noise power spectrum

[nsdata, Fs, bits]= wavread( ns_file);	%nsdata is a column vector

nwind= floor( 20* Fs/ 1000);	%this corresponds to 20ms window
if rem( nwind, 2)~= 0 nwind= nwind+ 1; end	%made window length even
noverlap= nwind/ 2;	w= hanning( nwind);	rowindex= ( 1: nwind)';
pred_order= 12;	%LPC order is set to 12
FFTlen=2*nwind;

%we assume the first NF_SABSENT frames are speech absent, we use them to estimate the noise power spectrum
noisedata= nsdata( 1: nwind* NF_SABSENT);	noise_colindex= 1+ ( 0: NF_SABSENT- 1)* nwind;
noisematrixdata = zeros( nwind, NF_SABSENT);
noisematrixdata( :)= noisedata( ...
   rowindex( :, ones(1, NF_SABSENT))+ noise_colindex( ones( nwind, 1), :)- 1);
noisematrixdata= noisematrixdata.* w( :, ones( 1, NF_SABSENT)) ;	%WINDOWING NOISE DATA
noise_r0= sum( sum( noisematrixdata.^ 2)/ nwind)/ NF_SABSENT;	%noise energy
noise_ps= mean( (abs( fft( noisematrixdata,FFTlen))).^ 2, 2);	
%NOTE!!!! it is a column vector

nslide= nwind- noverlap;

x= nsdata( nwind* NF_SABSENT+ 1: end);	%x is to-be-enhanced noisy speech
nx= length( x);	ncol= fix(( nx- noverlap)/ nslide);
colindex = 1 + (0: (ncol- 1))* nslide;
if nx< (nwind + colindex(ncol) - 1)
   x(nx+ 1: nwind+ colindex(ncol) - 1) = ...
      rand( nwind+ colindex( ncol)- 1- nx, 1)* (2^ (-15));   % zero-padding 
end



es_old= zeros( noverlap, 1);
%es_old is actually the second half of the previous enhanced speech frame,
%it is used for overlap-add



img=sqrt(-1);
for k= 1: ncol
   
   y= x( colindex( k): colindex( k)+ nwind- 1);
   y= y.* w;	%WINDOWING NOISY SPEECH DATA
   
   y_spec= fft(y,FFTlen);	y_specmag= abs( y_spec);	y_specang= angle( y_spec);
   %they are the frequency spectrum, spectrum magnitude and spectrum phase, respectively
   
   y_ps= y_specmag.^ 2;	%power spectrum of noisy speech
   
   %we must have a set of initial LPC coefficients to use for iterative wiener algorithm, 
   %we get it from original noisy speech file
   
   lpc_coeffs= (lpc(y, pred_order))';
   %now we get initial lpc coefficients of all current speech frame
   
   for m= 1: pred_order+ 1
      %exp_matrix(m, :)= exp(-i* (m- 1)* ((1: nwind)- 1)* 2* pi/ nwind);
      exp_matrix(m, :)= exp(-img* (m- 1)* ((1: FFTlen)- 1)* 2* pi/FFTlen);
   end
   
   
   x_old_spec=y_spec;
   for n=1:iter_num
     	
      xx= 1./ (abs( exp_matrix'* lpc_coeffs).^ 2);
      
      lpc_energy= mean( xx);
      
      
      tmp= y_ps- noise_ps;
      
      g= max( mean( tmp)./ lpc_energy, 1e-16);
      
      tmp1= g.* xx;
      h_spec= tmp1./ (tmp1+ noise_ps);  % Wiener filter
      
      es_tmpspec= x_old_spec.* h_spec;
      es_tmp= real( ifft( es_tmpspec,FFTlen));   
      
      x_old_spec = fft(es_tmp, FFTlen);
      % ----
      if n~= iter_num
         lpc_coeffs= lpc( es_tmp, pred_order)';   
      end   
   end
   
   es_data( colindex( k): colindex( k)+ nwind- 1)= [es_tmp( 1: noverlap)+ es_old;...
         es_tmp( noverlap+ 1: nwind)];
   %overlap-add
   es_old= es_tmp( nwind- noverlap+ 1: nwind);
end

wavwrite( es_data, Fs, bits, es_file);


