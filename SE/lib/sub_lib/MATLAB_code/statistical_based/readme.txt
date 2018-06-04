This folder contains statistical model based algorithms (Chapters 6 and 7):
										Ref
	wiener_iter.m	Iterative Wiener algorithm based on all-pole speech 
			production model.					[11] 	
	wiener_as.m	Wiener algorithm based on a priori SNR estimation	[12]
 	wiener_wt.m	Wiener algorithm based on wavelet thresholding 
			multi-taper spectra					[13]
 	mt_mask.m	Psychoacoustically-motivated algorithm			[14] 
	audnoise.m	Audible noise suppression algorithm			[15] 	
	mmse.m		MMSE algorithm with and without speech-presence 
			uncertainty						[16] 	
	logmmse.m	Log MMSE algorithm					[17] 
	logmmse_SPU.m	Log MMSE algorithm incorporating speech-presence 
			uncertainty						[18] 
	stsa_weuclid.m	Bayesian estimator based on weighted Euclidean 
				distortion measure.				[19] 
	stsa_wcosh.m	Bayesian estimator based on weighted cosh 
			distortion measure.					[19] 
	stsa_wlr.m	Bayesian estimator based on weighted likelihood 
			ratio distortion measure.				[19] 
	stsa_mis.m	Bayesian estimator based on modified Itakura-Saito 
			distortion measure.					[19] 
------------------------------------------------------------------------------------
USAGE

>> wiener_iter(infile.wav,outfile.wav,NumberOfIterations)
   where 'NumberOfIterations' is the number of iterations involved in iterative
   Wiener filtering.

>> wiener_as(infile.wav,outfile.wav)

>> wiener_wt(infile.wav,outfile.wav)

>> mt_mask(infile.wav,outfile.wav)

>> audnoise(infile.wav,outfile.wav)  
   Runs 2 iterations (iter_num=2) of the algorithm. 

>> mmse(infile.wav,outfile.wav,SPU)
	where SPU=1 - includes speech presence uncertainty
    	      SPU=0 - does not includes speech presence uncertainty

>> logmmse(infile.wav,outfile.wav)

>> logmmse_SPU(infile.wav,outfile.wav,option)
   where option=
      1  - hard decision ( Soon et al)
      2  - soft decision (Soon et al.)
      3  - Malah et al.(1999) 
      4  - Cohen (2002) 

>> stsa_weuclid(infile.wav,outfile.wav,p)
	where p>-2

>> stsa_wcosh(infile.wav,outfile.wav,p)
	where p>-1

>> stsa_wlr(infile.wav,outfile.wav);

>> stsa_mis(infile.wav,outfile.wav);

-----------------------------------------------------------------------------------
REFERENCES

[11] 	Lim, J. and Oppenheim, A. V. (1978). All-pole modeling of degraded speech. 
	IEEE Trans. Acoust. , Speech, Signal Proc., ASSP-26(3), 197-210.
[12] 	Scalart, P. and Filho, J. (1996). Speech enhancement based on a priori 
	signal to noise estimation. Proc. IEEE Int. Conf. Acoust. , Speech, Signal 
	Processing, 629-632.
[13] 	Hu, Y. and Loizou, P. (2004). Speech enhancement based on wavelet 
	thresholding the multitaper spectrum. IEEE Trans. on Speech and Audio 
	Processing, 12(1), 59-67.
[14] 	Hu, Y. and Loizou, P. (2004). Incorporating a psychoacoustical model in 
	frequency domain speech enhancement. IEEE Signal Processing Letters, 11(2), 
	270-273.
[15] 	Tsoukalas, D. E., Mourjopoulos, J. N., and Kokkinakis, G. (1997). Speech 
	enhancement based on audible noise suppression. IEEE Trans. on Speech and 
	Audio Processing, 5(6), 497-514.
[16] 	Ephraim, Y. and Malah, D. (1984). Speech enhancement using a minimum 
	mean-square error short-time spectral amplitude estimator. IEEE Trans. 
	Acoust.,Speech, Signal Process., ASSP-32(6), 1109-1121.
[17] 	Ephraim, Y. and Malah, D. (1985). Speech enhancement using a minimum 
	mean-square error log-spectral amplitude estimator. IEEE Trans. Acoust., 
	Speech, Signal Process., ASSP-23(2), 443-445.
[18] 	Cohen, I. (2002). Optimal speech enhancement under signal presence 
	uncertainty using log-spectra amplitude estimator. IEEE Signal Processing 
	Letters, 9(4), 113-116.
[19] 	Loizou, P. (2005). Speech enhancement based on perceptually motivated 
	Bayesian estimators of the speech magnitude spectrum. IEEE Trans. on Speech 
	and Audio Processing, 13(5), 857-869.

Copyright (c) 2006 by Philipos C. Loizou
$Revision: 0.0 $  $Date: 07/30/2006 $
------------------------------------------------------------------------------------