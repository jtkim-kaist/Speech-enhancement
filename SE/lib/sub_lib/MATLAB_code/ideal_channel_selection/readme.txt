Present folder contains implementations of ideal channel selection (ICS) algorithms (also known as binary mask algorithms) described in Chapter 13.

Updated: April 2012

Direct any inquiries about the code to: Philip Loizou (loizou@utdallas.edu)


Usage: ics(noisefile, clfile, outfile, nsnr, thrd)

% noisefile - name of masker file
% clfile - name of clean stimulus file
% outfile - name of output file
% nsnr is the overall input SNR (in dB) for noisy file
% thrd is the SNR threshold  (in dB)


Examples:

In MATLAB, type:

>> ics('babble1.wav','S_01_01.wav','out.wav',-10,-5)

For the above example, input SNR=-10 dB and SNR threshold=-5 dB.
The noisy (mixture) file is contained in 'S_01_01-noisy.wav' file.
Segregated file is in 'out.wav'

The wav files can be played via a Media Player, Cool Edit, Audition, etc.
It can also be viewed and played through our toolbox 'Colea':
http://www.utdallas.edu/~loizou/speech/software.htm

Reference:
Li, N. and Loizou, P. (2008). "Factors influencing intelligibility of ideal binary-masked speech: Implications for noise reduction," Journal of Acoustical Society of America, 123(3), 1673-1682


% ----------- -competing-talker demo --------------------------

Demo of separating two talkers speaking simultaneously:

Usage: ics_competing_talker(filename, clfile, t_outfile, m_outfile,thrd)

% filename - mixture filename
% clfile - clean target filename
% t_outfile - output file: Target talker
% m_outfile - output file: Competing talker
% thrd - SNR threshold in dB

Example:

>> ics_competing_talker('talker_mixture.wav','S_01_10.wav','target.wav','masker.wav',-5)

In 'talker_mixture.wav' mixture file, the competing talker was added at SNR=-5 dB
 (target and competing talkers were same for this example).
Files 'target.wav' and 'masker.wav' contain processed sentences of the segregated target and competing-talker talkers respectively. The SNR threshold was set to -5 dB.




%------------- SNResi rule (constraint rule) ----------------

Usage: ics_constr_rule(filename, clfile, outfile, GAIN)

% filename - noisy speech filename (mixture)
% clfile - clean speech filename
% outilfe - name of output file
% GAIN='Wiener'; 'MMSE', 'logMMSE', 'MMSE-SPU'; 'pMMSE'; 'SpecSub'

Example:

>> ics_constr_rule('S_01_02-babble_m10dB.wav', 'S_01_02.wav','out_constr.wav','Wiener')

Target was corrupted with babble at -10 dB SNR. The Wiener gain function was used.
Other possible gain functions: 'MMSE', 'logMMSE', 'MMSE-SPU'; 'pMMSE'; 'SpecSub'

Example with competing-talker:
>> ics_constr_rule('talker_mixture.wav','S_01_10.wav','out.wav','Wiener')

Another example in babble at input SNR=-5 dB
>> ics_constr_rule('S_02_02-babble_m5dB.wav', 'S_02_02.wav','out_constr.wav','Wiener')


References
Kim, G. and Loizou, P. (2011). “Gain-induced speech distortions and the absence of intelligibility benefit with existing noise-reduction algorithms,” J. Acoust. Soc. Am. 130(3), 1581-1596.

Loizou, P.  and Kim, G. (2011). "Reasons why Current Speech-Enhancement Algorithms do not Improve Speech Intelligibility and Suggested Solutions," IEEE Trans. Audio, Speech, Language Processing, 19(1), 47-56.


% --------------------- reverberation rule ----------------

Usage:  ics_reverb(reverbfile, clfile, outfile, thrd)

% reverbfile - name of  file containing reverberated stimulus
% clfile - name of clean sentence file
% outfile - name of output (processed) file
% thrd is the threshold (in dB) for signal-to-reverberant ratio criterion

Example:

>> ics_reverb('rev800_2.wav','clean_2.wav','outrev.wav',-8)

File was corrupted with RT60=0.8 sec reverberation.
The signal-to-reverberant ratio (SRR) threshold was set to -8 dB.

Reference
Kokkinakis, K., Hazrati, O. and Loizou, P. (2011). "A channel-selection criterion for suppressing reverberation in cochlear implants," Journal of the Acoustical Society of America, 129(5), 3221-3232.


%------------------- masker-based rule ------------------

Usage: ics_masker_rule(filename, clfile, outfile)

% filename - noisy speech filename (mixture)
% clfile - clean speech filename
% outilfe - name of output file

Example file corrupted at -10 dB SNR with babble:

>> ics_masker_rule('S_01_02-babble_m10dB.wav','S_01_02.wav','out_masker_rule.wav')
 
Reference:
Kim, G. and Loizou, P. (2010). "A new binary mask based on noise constraints for improved speech intelligibility," Proc. INTERSPEECH, Makuhari, Japan, pp. 1632-1635.

REFERENCES


Publications (from our lab) assuming ideal conditions (see Chapter 13):
======================================================
Hu, Y. and Loizou, P. (2008). "A new sound coding strategy for suppressing noise in cochlear implants," Journal of Acoustical Society of America, 124(1), 498-509.

Kim, G. and Loizou, P. (2011). “Gain-induced speech distortions and the absence of intelligibility benefit with existing noise-reduction algorithms,” J. Acoust. Soc. Am. 130(3), 1581-1596.

Kim, G. and Loizou, P. (2010). "A new binary mask based on noise constraints for improved speech intelligibility," Proc. INTERSPEECH, Makuhari, Japan, pp. 1632-1635.

Kokkinakis, K., Hazrati, O. and Loizou, P. (2011). "A channel-selection criterion for suppressing reverberation in cochlear implants," Journal of the Acoustical Society of America, 129(5), 3221-3232.

Li, N. and Loizou, P. (2008). "Factors influencing intelligibility of ideal binary-masked speech: Implications for noise reduction," Journal of Acoustical Society of America, 123(3), 1673-1682

Li, N. and Loizou, P. (2008). "Effect of spectral resolution on the intelligibility of ideal binary masked speech," Journal of Acoustical Society of America, 123(4), EL59- EL64

Loizou, P.  and Kim, G. (2011). "Reasons why Current Speech-Enhancement Algorithms do not Improve Speech Intelligibility and Suggested Solutions," IEEE Trans. Audio, Speech, Language Processing, 19(1), 47-56.




Publications (from our lab) assuming realistic conditions (see Chapter 13):
===========================================================

Hu, Y. and Loizou, P. (2008). "Techniques for estimating the ideal binary mask,&" Proc. of 11th International Workshop on Acoustic Echo and Noise Control, September 14th-17th, Seattle, Washington.

Hu, Y. and Loizou, P. (2010). "Environment-specific noise suppression for improved speech intelligibility by cochlear implant users," Journal of the Acoustical Society of America, 127(6), 3689-3695.

Kim, G., Lu, Y., Hu, Y. and Loizou, P. (2009). "An algorithm that improves speech intelligibility in noise for normal-hearing listeners," Journal of the Acoustical Society of America, 126(3), 1486-1494

Kim, G. and Loizou, P. (2010). "Improving Speech Intelligibility in Noise Using Environment-Optimized Algorithms," IEEE Trans. Audio, Speech, Language Processing, 18(8), 2080-2090.

Kim, G. and Loizou, P. (2010. "Improving Speech Intelligibility in Noise Using a Binary Mask that is Based on Magnitude Spectrum Constraints, " IEEE Signal Processing Letters, 17(2), 1010-1013

Kim, G. and Loizou, P. (2009). "A data-driven approach for estimating the time-frequency binary mask," Proc. Interspeech, Brighton, UK, Sept 6-9, 2009


