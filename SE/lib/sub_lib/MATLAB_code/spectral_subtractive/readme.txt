This folder contains spectral subtractive algorithms (Chapter 5):

 								Reference
	specsub.m	Basic spectral subtraction algorithm	[4]	   
	mband.m		Multi-band spectral subtraction.	[9]	   
	ss_rdc.m	Spectral subtraction with adaptive
			gain averaging and reduced delay 
			convolution				[10]	 
--------------------------------------------------------------------------
USAGE
>>	specsub(infile.wav,outfile.wav)

>>	mband(infile.wav,outfile.wav,Number_Of_Channels,Freq_Spacing)
	    where 
		'Number_of_Channels' is the number of bands 
		'Freq_spacing' is: 'linear', 'log', 'mel'

	    Example usage:
		 mband('sp04_babble_sn10.wav','outmb.wav',6,'linear');

>>	ss_rdc(infile.wav, outfile.wav)

--------------------------------------------------------------------------
References:

[4] 	Berouti, M., Schwartz, M., and Makhoul, J. (1979). Enhancement of speech 
	corrupted by acoustic noise. Proc. IEEE Int. Conf. Acoust., Speech, 
	Signal Processing, 208-211.
[9] 	Kamath, S. and Loizou, P. (2002). A multi-band spectral subtraction 
	method for enhancing speech corrupted by colored noise. Proc. IEEE Int. 
	Conf. Acoust.,Speech, Signal Processing
[10] 	Gustafsson, H., Nordholm, S., and Claesson, I. (2001). Spectral sub-
	traction using reduced delay convolution and adaptive averaging. IEEE 
	Trans. on Speech and Audio Processing, 9(8), 799-807.

Copyright (c) 2006 by Philipos C. Loizou
$Revision: 0.0 $  $Date: 07/30/2006 $
------------------------------------------------------------------------------