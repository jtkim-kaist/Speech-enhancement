function stsa_wlr(filename,outfile)

%
%  Implements the Bayesian estimator based on the weighted likelihood ratio
%  distortion measure [1, Eq. 37].
% 
%  Usage:  stsa_wlr(noisyFile, outputFile)
%           
%         infile - noisy speech file in .wav format
%         outputFile - enhanced output file in .wav format
%  
%
%  Example call:  stsa_wlr('sp04_babble_sn10.wav','out_wlr.wav');
%
%  References:
%   [1] Loizou, P. (2005). Speech enhancement based on perceptually motivated 
%       Bayesian estimators of the speech magnitude spectrum. IEEE Trans. on Speech 
%       and Audio Processing, 13(5), 857-869.
%   
% Author: Philipos C. Loizou
%
% Copyright (c) 2006 by Philipos C. Loizou
% $Revision: 0.0 $  $Date: 10/09/2006 $
%-------------------------------------------------------------------------

if nargin<2
   fprintf('Usage: stsa_wlr inFile outFile.wav \n\n');
   return;
end

[x, Srate, bits]= wavread( filename);	


% =============== Initialize variables ===============
%
len=floor(20*Srate/1000); % Frame size in samples
if rem(len,2)==1, len=len+1; end;
PERC=50; % window overlap in percent of frame size
len1=floor(len*PERC/100);
len2=len-len1; 


win=hanning(len);  % define window
win = win*len2/sum(win);  % normalize window for equal level output 


% Noise magnitude calculations - assuming that the first 6 frames is noise/silence
%
nFFT=len;
nFFT2=len/2;
noise_mean=zeros(nFFT,1);
j=1;
for k=1:5
   noise_mean=noise_mean+abs(fft(win.*x(j:j+len-1),nFFT));
   j=j+len;
end
noise_mu=noise_mean/5;
noise_mu2=noise_mu.^2;

%--- allocate memory and initialize various variables
   
img=sqrt(-1);
x_old=zeros(len1,1);
Nframes=floor(length(x)/len2)-1;
xfinal=zeros(Nframes*len2,1);
xinterv=0.001:0.01:10;
k=1;
aa=0.98;

%===============================  Start Processing =======================================================
%
fprintf('This might take some time ...\n')
for n=1:Nframes 
   
  
   insign=win.*x(k:k+len-1);
    
   %--- Take fourier transform of  frame
   
   spec=fft(insign,nFFT);   
   sig=abs(spec); % compute the magnitude
   sig2=sig.^2;
   
       gammak=min(sig2./noise_mu2,40);  % post SNR. Limit it to avoid overflows
              
       if n==1
           ksi=aa+(1-aa)*max(gammak-1,0);
       else
           ksi=aa*Xk_prev./noise_mu2 + (1-aa)*max(gammak-1,0);     % a priori SNR   
       end
     
       vk=ksi.*gammak./(1+ksi);
      
       
       xx=solve_wlr(vk,gammak,sig,xinterv); % solves Eq. 37 in [1]
       
       sig_hat=xx;
       Xk_prev=sig_hat.^2;
       
       xi_w= ifft( sig_hat.* exp(img*angle(spec))); 
	   xi_w= real( xi_w);
	  
      
	% --- Overlap and add ---------------
    %
    xfinal(k:k+ len2-1)= x_old+ xi_w(1:len1);
	x_old= xi_w(len1+ 1: len);

    if rem(n,20)==0, fprintf('Frame: %d Percent completed:%4.2f \n',n,n*100/Nframes); end;
    
 k=k+len2;
end
%========================================================================================




wavwrite(xfinal,Srate,16,outfile);


%==========================================================================
function x=solve_wlr(vk,gammak,Yk,xx);

% solves non-linear Eq. 37 in [1]
%

Len=length(vk);
L2=Len/2+1;

lk05=sqrt(vk).*Yk./gammak;
Ex=gamma(1.5)*lk05.*confhyperg(-0.5,1,-vk,100);
Elogx=1-0.5*(2*log(lk05)+log(vk)+expint(vk));

x=zeros(Len,1);
  
for n=1:L2

    a=Elogx(n);
    b=Ex(n);
    ff=sprintf('log(x)+%f - %f/x',a,b);
    y=log(xx)+a-b./xx;
    bet=xx(1); tox=200;
    if y(1)<0
            ind=find(y>0);
            bet=xx(1)/2;
            tox=xx(ind(1));
             
            [x(n),fval,flag]=fzero(inline(ff),[bet tox]);
            if flag<0
                x(n)=x(n-1);
           end
    else
        ind=find(y<0);
        if ~isempty(ind)
            bet=xx(1);
            tox=xx(ind(1));
            [x(n),fval]=fzero(inline(ff),[bet tox]);
            
        else
   
            x(n)=0.001;  % spectral floor
   
        end
    end    
       

end

x(L2+1:Len)=flipud(x(2:L2-1));

