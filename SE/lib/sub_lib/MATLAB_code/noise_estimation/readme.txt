This folder contains noise-estimation algorithms (Chapter 9)

	specsub_ns.m		Basic spectral subtraction algorithm implemented 
				with different noise estimation algorithms:	
	martin_estimation.m	Martin’s minimum tracking			[7]
	mcra_estimation.m	MCRA algorithm					[22]
	mcra2_estimation.m	MCRA-2 algorithm				[8]
	imcra_estimation.m	IMCRA algorithm					[23]
	doblinger_estimation.m	Continuous minimal tracking 			[24]
	hirsch_estimation.m	Weighted spectral average			[25]
	connfreq_estimation.m	Connected time-frequency regions		[26]

------------------------------------------------------------------------------------
USAGE
>> specsub_ns(infile.wav,method,outfile.wav)
   where 'method' is:

   'martin'    = Martin's minimum tracking algorithm
   'mcra'      = Minimum controlled recursive average algorithm (Cohen) 
   'mcra2'     = variant of Minimum controlled recursive average algorithm 
   'imcra'     = improved Minimum controlled recursive average algorithm (Cohen) 
   'doblinger' = continuous spectral minimum tracking (Doblinger)
   'hirsch'    = weighted spectral average (Hirsch & Ehrilcher)  
   'conn_freq' = connected frequency regions (Sorensen & Andersen)

------------------------------------------------------------------------------------
REFERENCES:

[7] 	Martin, R. (2001). Noise power spectral density estimation based on optimal
	smoothing and minimum statistics. IEEE Transactions on Speech and Audio 
	Processing, 9(5), 504-512.
[8] 	Rangachari, S. and Loizou, P. (2006). A noise estimation algorithm  for 
	highly nonstationary environments. Speech Communication, 28, 220-231.
[22] 	Cohen, I. (2002). Noise estimation by minima controlled recursive averaging 
	for robust speech enhancement. IEEE Signal Processing Letters, 9(1), 12-15.
[23] 	Cohen, I. (2003). Noise spectrum estimation in adverse environments: 
	Improved minima controlled recursive averaging. IEEE Transactions on Speech 
	and Audio Processing, 11(5), 466-475.
[24] 	Doblinger, G. (1995). Computationally efficient speech enhancement by 
	spectral minima tracking in subbands. Proc. Eurospeech, 2, 1513-1516.
[25] 	Hirsch, H. and Ehrlicher, C. (1995). Noise estimation techniques for robust 
	speech recognition. Proc. IEEE Int. Conf. Acoust. , Speech, Signal 
	Processing, 153-156.
[26] 	Sorensen, K. and Andersen, S. (2005). Speech enhancement with natural 
	sounding residual noise based on connected time-frequency speech presence 
	regions. EURASIP J. Appl. Signal Process., 18, 2954-2964.

Copyright (c) 2006 by Philipos C. Loizou
$Revision: 0.0 $  $Date: 07/30/2006 $
------------------------------------------------------------------------------------