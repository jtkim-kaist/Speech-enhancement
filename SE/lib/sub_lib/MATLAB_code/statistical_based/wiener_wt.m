function wiener_wt(noisy_file, outfile)

%
%  Implements the Wiener filtering algorithm based wavelet-thresholding
%  multi-taper spectra [1].
% 
%  Usage:  wiener_wt(noisyFile, outputFile)
%           
%         infile - noisy speech file in .wav format
%         outputFile - enhanced output file in .wav format
%
%         
%  Example call:  wiener_wt('sp04_babble_sn10.wav','out_wt.wav');
%
%  References:
%   [1] Hu, Y. and Loizou, P. (2004). Speech enhancement based on wavelet 
%       thresholding the multitaper spectrum. IEEE Trans. on Speech and Audio 
%       Processing, 12(1), 59-67.
%   
% Authors: Yi Hu and Philipos C. Loizou
%
% Copyright (c) 2006 by Philipos C. Loizou
% $Revision: 0.0 $  $Date: 10/09/2006 $
%-------------------------------------------------------------------------
if nargin<2
   fprintf('Usage: wiener_wt(noisyfile.wav,outFile.wav) \n\n');
   return;
end

% Initialize wavelet-threshold parameters
%
wavname='db4'; % name of the wavelet used to generate wavelet filters
thre_type='ds'; %  threshold type, scale-dependent ('d'), or
                %   'scale-independent' ('i')
thre_func_type='s'; % threshold function type: soft ('s') or hard ('h');
q_0=5;  %  decomposition level
taper_num=16; % number of tapers to use
mu_max=5;

%just like the 'alpha' factor in Berouti's method. It can be shown that mu
%plays the same role in our approach as 'alpha' in Berouti's method.
%We adopt the same methodology as Berouti, let mu vary with SNRlog (use
%linear relation) with a value not equal to 0, we fix mu equal to MU_opt.
mu_toplus= 1; %mu value with SNRlog>= 20dB
mu_tominus= mu_max;   %mu value with SNRlog< -5dB
mu_slope= (mu_tominus- mu_toplus )/ 25;
mu0= mu_toplus+ 20* mu_slope;

%------------------get the noisy speech data
[noisy_speech, Srate, NBITS]= wavread( noisy_file);


frame_dur= 20;  %unit is milli-second
len= floor( Srate* frame_dur/ 1000);
if rem( len, 2)~= 0
    len= len+ 1;
end
NFFT= len;
tapers= sine_taper( taper_num, NFFT);
diga= digamma( taper_num)- log( taper_num);

L120= floor( 120* Srate/ 1000);
noise= noisy_speech( 1: L120);
noise_ps= psd_mt_sine( noise, tapers);
%estimate the noise psd using multitaper method

%=================using wavelet thresholding by Walden's paper
%----------construct Eq. (8) in Walden's paper
q= ceil( log2( len));
M= 2^ q;

N_autoc= trigamma( taper_num)* ( 1- ( 0: taper_num+ 1)/ ( taper_num+ 1));
N_autoc( M/ 2+ 1)= 0;
Sigma_N_firstrow= [N_autoc( 1: M/ 2+ 1), fliplr( N_autoc( 2: M/ 2))];
noise_stat= real( fft( Sigma_N_firstrow));
%----------

[wfilter( 1, :), wfilter( 2, :), wfilter( 3, :), wfilter( 4, :)]= ...
    wfilters( wavname);
%------get the wavelet/scaling filter for decomposition/reconstruction

log_noise_ps= log( noise_ps)- diga;
den_log_noise_ps= thre_wavelet( log_noise_ps, noise_stat, thre_type, ...
    thre_func_type, wfilter, q_0);
den_log_noise_ps= [den_log_noise_ps( 1: len/ 2+ 1); ...
    flipud( den_log_noise_ps( 2: len/ 2))];
noise_ps= exp( den_log_noise_ps);
%=================

mu_vad= 0.98; % smoothing factor in noise spectrum update
a_dd= 0.98; % smoothing factor in priori update
eta= 0.15; % VAD threshold

win= hamming( len);
PERC= 50; % window overlap in percent of frame size
len1=floor(len* PERC/ 100);
len2= len- len1;
FLOOR= 0.002;

k= 1;

Nframes= floor( length( noisy_speech)/ len2)- 1;
x_old= zeros( len1, 1);
xfinal= zeros( Nframes* len2, 1);

%===============================  Start Processing

for n= 1: Nframes

    insign= noisy_speech( k: k+ len- 1);
    insign_spec= fft( insign.* win);

    %estimate the noisy speech power spectrum
    ns_ps= psd_mt_sine( insign, tapers);
    log_ns_ps= log( ns_ps)- diga;
    den_log_ns_ps= thre_wavelet( log_ns_ps, noise_stat, thre_type, ...
        thre_func_type, wfilter, q_0);
    den_log_ns_ps= [den_log_ns_ps( 1: len/ 2+ 1); ...
        flipud( den_log_ns_ps( 2: len/ 2))];
    noisy_ps= exp( den_log_ns_ps);
    

    % ============ voice activity detection
    posteri= abs( insign_spec).^ 2/ (norm( win)^2)./ noise_ps;
    posteri_prime= posteri- 1;
    posteri_prime( find( posteri_prime< 0))= 0;
    
    if (n== 1) % initialize posteri
        priori= a_dd+ (1-a_dd)* posteri_prime;
    else
        priori= a_dd* (G_prev.^ 2).* posteri_prev+ ...
            (1-a_dd)* posteri_prime;
    end    

    log_sigma_k= posteri.* priori./ (1+ priori)- log(1+ priori);    
    vad_decision(n)= sum( log_sigma_k)/ len;    
    if (vad_decision(n)< eta) 
        % noise only frame found
        noise_ps= mu_vad* noise_ps+ (1- mu_vad)* noisy_ps;
        vad( k: k+ len- 1)= 0;
    else
        vad( k: k+ len- 1)= 1;
    end
    % ===end of vad===
    
    cl_ps= noisy_ps- noise_ps;
    %estimate the clean speech power spectrum
    cl_ps= max( cl_ps, FLOOR* noise_ps);

    SNR= sum( cl_ps)/ sum( noise_ps);
    SNRlog( n)= 10* log10( SNR+ eps);

    if SNRlog( n)>= 20
        mu( n)= mu_toplus;
        %actually this corresponds to wiener filtering
    elseif ( SNRlog( n)< 20) & ( SNRlog( n)>= -5)
        mu( n)= mu0- SNRlog( n)* mu_slope;
    else
        mu( n)= mu_tominus;
    end

    G= cl_ps./ ( cl_ps+ mu( n)* noise_ps);
    
    xi_w_freq= G.* insign_spec;    
    Xk_prev= abs(xi_w_freq).^2; 

    xi_w= ifft( xi_w_freq);
    xi_w= real( xi_w);  % to be safe

    xfinal( k: k+ len2- 1)= x_old+ xi_w( 1: len1);
    x_old= xi_w( len1+ 1: len);
    k= k+ len2;
    
    G_prev= G; 
    posteri_prev= posteri;

end
%========================================================================================

wavwrite( xfinal, Srate, 16, outfile);



% ================ E N D   ===================================
function after_thre= thre_wavelet( before_thre, noise_stat, ...
	thre_type, thre_func_type, wfilter, q_0)

%this function implements the wavelet thresholding technique
%   refer to the paper by Walden/1998, Donoho/1995, Johnstone/1997

%note on the parameters
%   before_thre: data before thresholding
%   noise_stat: the power spectrum of the noise (i.e., noise statistics), 
%       DFT of the first row of Sigma_N, refer to Eq. (8) in Walden's paper
%   thre_type: threshold type, scale-dependent Universal ('d'), 
%		scale-independent Universal ('i'), scale-dependent SURE ('ds'), 
%		scale-independent SURE ('is'), or scale-dependent Generalized 
%		Corss-Validation ('dg')
%   thre_func_type: threshold function type: soft ('s') or hard ('h');
%   wfilter: wavelet low pass and high pass decomposition/reconstruction filters [lo_d, hi_d, lo_r, hi_r]
%       the 1st row is lo_d, the 2nd row is hi_d, the 3rd row is lo_r, and the 4th row is hi_r
%   q_0 is the decomposition level

%   after_thre: data after thresholding

s= size( before_thre);  
before_thre= before_thre( :)';   %make it a row vector
noise_stat= noise_stat( :)';

N= length( before_thre);    %length of before-thresholded data
q= ceil( log2( N)); 
M= 2^ q;

%==get the low pass and high pass decomposition/reconstruction filters from wfilter
lo_d= wfilter( 1, :);   %low pass decomposition filter/ scaling filter
hi_d= wfilter( 2, :);   %high pass decomposition filter/ wavelet filter
lo_r= wfilter( 3, :);   %low pass reconstruction filter/ scaling filter
hi_r= wfilter( 4, :);   %high pass reconstruction filter/ wavelet filter

%==refer to pp. 3155 in Walden's paper
H= zeros( q_0, M);
H( 1, :)= fft( hi_d, M);   %frequency response of wavelet filter
G( 1, :)= fft( lo_d, M);  %frequency response of scaling filter
for i= 2: q_0- 1
    G( i, :)= G( 1, rem( (2^ (i- 1) )* (0: M- 1), M)+ 1);
end

for j= 2: q_0
    H( j, :)= prod( [G( 1: j- 1, :); H( 1, rem( (2^ (j- 1) )* (0: M- 1), M)+ 1)], 1);
end

[y_coeff, len_info]= wavedec( before_thre, q_0, lo_d, hi_d);

% --decompose before_thre into q_0 levels using wavelet filter hi_d and scaling filter lo_d
% --where y_coeff contains the coefficients and len_info contains the length information
% --different segments of y_coeff correspond approximation and detail coefficients;
% -- length of len_info should be q_0+ 2

%===============processing according to 'thre_type'
%-------with 'd'--scale-dependent thresholding, threshold has to be computed for each level
%-------with 'i'--scale-independent thresholding, threshold is set to a fixed level

if thre_type== 'i' %scale-independent universal thresholding
    sigma_square= mean( noise_stat);
    thre= sqrt( sigma_square* 2* log( M)) ;    %mean( noise_stat) is sigma_eta_square in Eq. (6)
    y_coeff( len_info( 1)+ 1: end)= ...
        wthresh( y_coeff( len_info( 1)+ 1: end), thre_func_type, thre);
    
elseif thre_type== 'd' %scale-dependent universal thresholding
    %------first we need to compute the energy level of each scale from j= 1: q_0
    for i= 1: q_0   %refer to Eq. (9) in Walden's paper
        sigma_j_square( i)= mean( noise_stat.* (abs( H( i, :)).^ 2), 2);   %average along the row          
    end

    for i= 2: q_0+ 1    %thresholding for each scale
        
        sp= sum( len_info( 1: i- 1), 2)+ 1; %starting point
        ep= sp+ len_info( i)- 1;
        thre= sqrt( sigma_j_square( q_0- i+ 2)* 2* log( len_info( i)));
        y_coeff( sp: ep)= wthresh( y_coeff( sp: ep), thre_func_type, thre);
        
    end
    
elseif thre_type== 'ds' %scale-dependent SURE thresholding
    
%=======use Eq. (9) in Walden's paper to get sigma_j, MDA estimate seems to be better
%     for i= 1: q_0   
%         sigma_j_square( i)= mean( noise_stat.* (abs( H( i, :)).^ 2), 2);   %average along the row 
%         sigma_j( i)= sqrt( sigma_j_square( i));
%     end

%======MDA estimate of sigma_j
    sigma_j= wnoisest( y_coeff, len_info, 1: q_0);  
    
    for i= 2: q_0+ 1    %thresholding for each scale
        
        sp= sum( len_info( 1: i- 1), 2)+ 1; %starting point
        ep= sp+ len_info( i)- 1;    %ending point
        if sigma_j( q_0- i+ 2)< sqrt( eps)* max(  y_coeff( sp: ep));
            thre= 0;
        else
            thre= sigma_j( q_0- i+ 2)* thselect( y_coeff( sp: ep)/ ...
                sigma_j( q_0- i+ 2), 'heursure');
        end
        
        %fprintf( 1, 'sigma_j is %6.2f, thre is %6.2f\n', sigma_j, thre);
        y_coeff( sp: ep)= wthresh( y_coeff( sp: ep), thre_func_type, thre);
        
    end
    
elseif thre_type== 'dn' %new risk function defined in Xiao-ping Zhang's paper
    
    sigma_j= wnoisest( y_coeff, len_info, 1: q_0);  
    sigma_j_square= sigma_j.^ 2;
    
    for i= 2: q_0+ 1    %thresholding for each scale
        
        sp= sum( len_info( 1: i- 1), 2)+ 1; %starting point
        ep= sp+ len_info( i)- 1;    %ending point       
        if sigma_j( q_0- i+ 2)< sqrt( eps)* max(  y_coeff( sp: ep));
            thre= 0;
        else    
            
            %based on some evidece, the following theme let thre vary with SNR
            %   with ultra low SNR indicating low probability of signal presence, 
            %       hence using universal threshold
            %   and very high SNR indicates high probability of signal presence,
            %       hence using SURE threshold
            
            thre_max= sigma_j( q_0- i+ 2)* sqrt( 2* log( len_info( i))); %thre with SNRlog< -5dB
            thre_min= sigma_j( q_0- i+ 2)* fminbnd( @riskfunc, 0, sqrt(2* log( ep- sp+ 1)), ...
                optimset( 'MaxFunEvals',1000,'MaxIter',1000), ...
                y_coeff( sp: ep)/ sigma_j( q_0- i+ 2), 3);   %thre with SNRlog> 20dB
            slope= (thre_max- thre_min)/ 25;
            thre_0= thre_min+ 20* slope;        

            SNRlog= 10* log10( mean( max( y_coeff( sp: ep).^ 2/ sigma_j_square( q_0- i+ 2)- 1, 0)));            
            if SNRlog>= 20
                thre= thre_min;  %actually this corresponds to SURE threshold
            elseif ( SNRlog< 20) & ( SNRlog>= -5)
                thre= thre_0- SNRlog* slope;
            else
                thre= thre_max;   %this corresponds to oversmooth threshold
            end
            
            %the theme below is similar to the option 'heursure' in the function 'thselect'
%             univ_thr = sqrt(2* log( len_info( i)));  %universal thresholding
%             eta = (norm( y_coeff( sp: ep)/ sigma_j( q_0- i+ 2)).^2)/ ( len_info( i))- 1;
%             crit = (log2( len_info( i)))^(1.5)/ sqrt( len_info( i));
%             if 1%eta > crit   %high probility that speech exists
%                 thre= sigma_j( q_0- i+ 2)* fminbnd( @riskfunc, 0, sqrt(2* log( ep- sp+ 1)), ...
%                     optimset( 'MaxFunEvals',1000,'MaxIter',1000), ...
%                     y_coeff( sp: ep)/ sigma_j( q_0- i+ 2), 3); 
%             else
%                 thre = sigma_j( q_0- i+ 2)* univ_thr;                    
%             end

        end
        
        y_coeff( sp: ep)= wthresh( y_coeff( sp: ep), thre_func_type, thre);
        
    end

elseif thre_type== 'dg' %scale-dependent Generalized Cross Validation thresholding
    
    for i= 2: q_0+ 1    %thresholding for each scale
        
        sp= sum( len_info( 1: i- 1), 2)+ 1; %starting point
        ep= sp+ len_info( i)- 1;    %ending point       
        [y_coeff( sp: ep), thre]= mingcv( y_coeff( sp: ep), thre_func_type);
        
    end   
   
else 
    error( 'wrong thresholding type');
end

%--reconstruct the thresholded coefficients
after_thre= waverec( y_coeff, len_info, lo_r, hi_r);

if s(1)>1 
    after_thre= after_thre'; 
end
%fprintf( 1, 'thre is %f\n', thre);



function mt_psd= psd_mt_sine( data, sine_tapers)

% this function uses sine tapers to get multitaper power spectrum estimation
% 'x' is the incoming data, 'sine_tapers' is a matrix with each column being
% sine taper, sine_tapers can be obtained using the function sine_taper

[frame_len, taper_num]= size( sine_tapers);

eigen_spectra= zeros( frame_len, taper_num);

data= data( :);
data_len= length( data);
data_hankel= hankel( data( 1: frame_len), data( frame_len: data_len));

x_mt_psd= zeros( frame_len, data_len- frame_len+ 1);

for pp= 1: data_len- frame_len+ 1
    for index= 1: taper_num
        x_taperd= sine_tapers( :, index).* data_hankel( :, pp);
        x_taperd_spec= fft( x_taperd);
        eigen_spectra( :, index)= abs( x_taperd_spec).^ 2;
    end
    x_mt_psd(:, pp)= mean( eigen_spectra, 2);
end

mt_psd= mean( x_mt_psd, 2);



function tapers= sine_taper( L, N)

% this function is used to generate the sine tapers proposed by Riedel et
% al in IEEE Transactions on Signal Processing, pp. 188- 195, Jan. 1995

% there are two parameters, 'L' is the number of the sine tapers generated,
% and 'N' is the length of each sine taper; the returned value 'tapers' is
% a N-by-L matrix with each column being sine taper

tapers= zeros( N, L);

for index= 1: L
    tapers( :, index)= sqrt( 2/ (N+ 1))* sin (pi* index* (1: N)'/ (N+ 1));
end





function y = trigamma(z,method,debug)

%  y = trigamma(z)   ... Trigamma-Function for real positive z
%
%  trigamma(z) = (d/dz)^2 log(gamma(z)) = d/dz digamma(z)
%
%  if 'z' is a matrix, then the digamma-function is evaluated for
%  each element. Results are inaccurate for real arguments < 10 which are
%  neither integers nor half-integers.
%
%  y = trigamma(z,method)
%
%  possible values for optional argument 'method':
%    method = 1           : quick asymptotic series expansion (approximate)
%    method = 2           : finite recursion for integer values (exact)
%    method = 3           : finite recursion for half-integer values (exact)
%    method = 4 (default) : automatic selection of 1,2 or 3 for individual
%                           elements in z whichever is appropriate.
%
%  see also: digamma, gamma, gammaln, gammainc, specfun


%  reference: Abramowitz & Stegun, "Handbook of Mathematical Functions"
%             Chapter "Gamma Function and Related Functions" :
%  implemented by: Christoph Mecklenbraeuker
%             (email: cfm@sth.ruhr-uni-bochum.de), July 4, 1995.


dim = size(z); 				% save original matrix dimension
z = reshape(z,dim(1)*dim(2),1); 	% make a column vector
I1 = ones(length(z),1); 		% auxiliary vector of ones

if(nargin==1)
    method=4; debug=0;
elseif(nargin==2)
    debug=0;
end;


if(debug == 1) 				% if debug==1: track recursion
    [m,n] =size(z);
    fprintf(1,'trigamma: method =	%d, size(z)=[%d %d],\t min(z)=%f, max(z)=%f\n',...
        method,m,n,min(min(z)),max(max(z)));
end;

if(method==1) 				% use 9th order asymptotic expansion
    if(any(z<1))
        fprintf(1,'Warning: some elements in argument of "trigamma(z,1)" are < 1\n');
        fprintf(1,'minimal argument = %g: trigamma-result is inaccurate!\n',min(min(z)));
    end

    % calculate powers of 1/z :
    w1 = 1./z; w2 = w1.*w1; w3 = w1.*w2; w5 = w2.*w3; w7 = w2.*w5; w9 = w2.*w7;
    % generate coefficients of expansion: matrix with constant columns
    a = [ I1   I1/2   I1/6  -I1/30  I1/42 -I1/30];
    % make vector of powers of 1/z:
    w = [ w1   w2     w3     w5      w7    w9];
    % calculate expansion by summing the ROWS of (a .* w) :
    y = sum((a.*w).').';
elseif(method==2)
    zmax = max(max(floor(z)));
    ytab = zeros(zmax,1);
    ytab(1) = pi^2/6; 			% = psi'(1)
    for n=1:zmax-1;
        ytab(n+1) = ytab(n) - 1/n^2; 	% generate lookup table
    end;
    y = ytab(z);
elseif(method==3)
    zmax = max(max(floor(z)));
    ytab = zeros(zmax+1,1);
    ytab(1) = pi^2/2; 			% = psi'(1/2)
    for n=1:zmax;
        ytab(n+1) = ytab(n) - 4/(2*n-1)^2; % generate lookup table
    end;
    y = ytab(z+0.5);
elseif(method==4) 			% decide here which method to use
    Less0 = find(z<0); 			% negative arguments evaluated by reflexion formula
    Less1 = find(z>0 & z<1); 		% values between 0 and 1.
    fraction = rem(z,1); 			% fractional part of arguments
    f2 = rem(2*fraction,1);
    Integers = find(fraction==0 & z>0); 	% Index set of positive integer arguments
    NegInts  = find(fraction==0 & z<=0); 	% Index set of positive integer arguments
    HalfInts = find(abs(fraction-0.5)<1e-7 & z>0); % Index set of positive half-integers
    Reals    = find(f2>1e-7 & z>1); 	% Index set of all other arguments > 1
    if(~isempty(Reals)) y(Reals)    = trigamma(z(Reals),1,debug);    end;
    if(~isempty(Less1)) y(Less1)    = trigamma(z(Less1)+2,1,debug) + ...
            1./z(Less1).^2+1./(z(Less1)+1).^2;end;
    % reflexion formula:
    if(~isempty(Less0)) y(Less0)= -trigamma(1-z(Less0),1,debug)+(pi./sin(pi*z(Less0))).^2; end;
    % integers:
    if(~isempty(Integers)) y(Integers) = trigamma(z(Integers),2,debug); end;
    % half-integers:
    if(~isempty(HalfInts)) y(HalfInts) = trigamma(z(HalfInts),3,debug); end;
    % negative integers:
    if(~isempty(NegInts))  y(NegInts)  = Inf * NegInts; end;
end

y = reshape(y,dim(1),dim(2));
return;




function psi = digamma(z,method,debug)
%
%  psi = digamma(z)   ... Digamma-Function for real argument z.
%
%  digamma(z) = d/dz log(gamma(z)) = gamma'(z)/gamma(z)
%
%  if 'z' is a matrix, then the digamma-function is evaluated for
%  each element. Results may be inaccurate for real arguments < 10
%  which are neither integers nor half-integers.
%
%  psi = digamma(z,method)
%
%  possible values for optional argument 'method':
%    method = 1           : quick asymptotic series expansion (approximate)
%    method = 2           : finite recursion for integer values (exact)
%    method = 3           : finite recursion for half-integer values (exact)
%    method = 4 (default) : automatic selection of 1,2 or 3 for individual
%                           elements in z whichever is appropriate.
%
%  see also:  trigamma, gamma, gammaln, gammainc, specfun

%  reference: Abramowitz & Stegun, "Handbook of Mathematical Functions"
%             Chapter "Gamma Function and Related Functions" :
%  implemented by: Christoph Mecklenbraeuker
%             (email: cfm@sth.ruhr-uni-bochum.de), July 1, 1995.


dim = size(z); 				% save original matrix dimension
z = reshape(z,dim(1)*dim(2),1); 	% make a column vector
I1 = ones(length(z),1); 		% auxiliary vector of ones

if(nargin==1)
    method=4; debug=0;
elseif(nargin==2)
    debug=0;
end;

if(debug == 1) 				% if debug==1: track recursion
    [m,n] = size(z);
    fprintf(1,'digamma: method = %d, size(z)=[%d %d],\t min(z)=%f, max(z)=%f\n',...
        method,m,n,min(min(z)),max(max(z)));
end;


if(method==1) 				% use 8th order asymptotic expansion
    if(any(z<1))
        fprintf(1,'Warning: some elements in argument of "digamma(z,1)" are < 1\n');
        fprintf(1,'minimal argument = %g: digamma-result is inaccurate!\n',min(min(z)));
    end
    % calculate powers of 1/z :
    w1 = 1./z; w2 = w1.*w1; w4 = w2.*w2; w6 = w2.*w4; w8 = w4.*w4;
    % generate coefficients of expansion: matrix with constant columns
    a = [ -I1/2   -I1/12   I1/120  -I1/252  I1/240 ];
    % make vector of powers of 1/z:
    w = [  w1      w2       w4       w6       w8   ];
    % calculate expansion by summing the ROWS of (a .* w) :
    psi = log(z) + sum((a.*w).').';
elseif(method==2)
    zmax = max(max(floor(z)));
    psitab = zeros(zmax,1);
    psitab(1) = -0.5772156649015328606;
    for n=1:zmax-1;
        psitab(n+1) = psitab(n) + 1/n; 	% generate lookup table
    end;
    psi = psitab(z);
elseif(method==3)
    zmax = max(max(floor(z)));
    psitab = zeros(zmax+1,1);
    psitab(1) = -0.5772156649015328606 - 2*log(2);  % = psi(1/2)
    for n=1:zmax;
        psitab(n+1) = psitab(n) + 2/(2*n-1); % generate lookup table
    end;
    psi = psitab(z+0.5);
elseif(method==4) 			% decide here which method to use
    Less0 = find(z<0); 			% negative arguments evaluated by reflexion formula
    Less1 = find(z>0 & z<1); 		% values between 0 and 1.
    fraction = rem(z,1); 			% fractional part of arguments
    f2 = rem(2*fraction,1);
    Integers = find(fraction==0 & z>0); 	% Index set of positive integer arguments
    NegInts  = find(fraction==0 & z<=0); 	% Index set of positive integer arguments
    HalfInts = find(abs(fraction-0.5)<1e-7 & z>0); % Index set of positive half-integers
    Reals    = find(f2>1e-7 & z>1); 	% Index set of all other arguments > 1
    if(~isempty(Reals)) psi(Reals)    = digamma(z(Reals),1,debug);    end;
    if(~isempty(Less1)) psi(Less1)    = digamma(z(Less1)+2,1,debug) - ...
            1./z(Less1)-1./(z(Less1)+1);end;
    % reflexion formula:
    if(~isempty(Less0)) psi(Less0) = digamma(1-z(Less0),1,debug) - pi./tan(pi*z(Less0)); end;
    if(~isempty(Integers)) psi(Integers) = digamma(z(Integers),2,debug); end;
    if(~isempty(HalfInts)) psi(HalfInts) = digamma(z(HalfInts),3,debug); end;
    if(~isempty(NegInts))  psi(NegInts)  = Inf * NegInts; end;
end

psi = reshape(psi,dim(1),dim(2));

return;

