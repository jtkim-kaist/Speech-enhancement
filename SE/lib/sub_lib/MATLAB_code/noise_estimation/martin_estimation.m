function [parameters]=martin_estimation(ns_ps,parameters);

YFRAME = ns_ps;
alpha_corr = parameters.alpha_corr;
alpha = parameters.alpha;
P = parameters.P;
Pbar = parameters.Pbar;
Psqbar = parameters.Psqbar;
actmin = parameters.actmin;
actmin_sub = parameters.actmin_sub;
minact = parameters.minact;
Pmin_u = parameters.Pmin_u;
subwc = parameters.subwc;
u = parameters.u;
lmin_flag = parameters.lmin_flag;
n = parameters.n;
L = parameters.L;
R = parameters.R;
Um = parameters.Um;
V = parameters.V;
D = parameters.D;
Av = parameters.Av;
alpha_max = parameters.alpha_max;
alpha_min = parameters.alpha_min;
beta_max = parameters.beta_max;
M_D = parameters.M_D;
M_V = parameters.M_V;
H_D = parameters.H_D;
H_V = parameters.H_V;
noise_est = parameters.noise_ps;

%calculating the optimal smoothing correction factor
alpha_corr_t=1/(1+ ((sum(P)/sum(YFRAME)-1)^2));
alpha_corr=0.7*alpha_corr+0.3*max(alpha_corr_t,0.7);

%calculating the optimal smoothing factor

alpha=(alpha_max*alpha_corr)./((P./(noise_est+eps) -1).^2 +1);  
alpha = max(0.3, alpha);



%calculation of smoothed periodogram
P=alpha.*P+((1-alpha).*YFRAME);
%calculation of variance of P(i,k) and Qeq(i,k)
bet=alpha.^2;
bet=min(bet,beta_max);
Pbar=bet.*Pbar+(1-bet).*P;
Psqbar=bet.*Psqbar+(1-bet).*(P.^2);
varcap_P=abs(Psqbar-(Pbar.^2));
Qeqinv=varcap_P./(2*(noise_est.^2));
Qeqinv=min(Qeqinv,0.5);
Qeq=1./(Qeqinv+eps);
%calculation of Bmin(i,k) and Bmin_sub(i,k)

Qeq_tild=(Qeq-2*M_D)./(1-M_D);
Qeq_tild_sub=(Qeq-2*M_V)./(1-M_V);
%Bmin=1+(((D-1)*2./Qeq_tild).*gamma((1+(2./Qeq)).^H_D));
%Bmin_sub=1+(((V-1)*2./Qeq_tild).*gamma((1+(2./Qeq)).^H_V));
Bmin=1+(D-1)*2./Qeq_tild;  % using the approximation in Eq. 17
Bmin_sub=1+(V-1)*2./Qeq_tild_sub;

%calculation of Bc(i)
Qinv_bar=(1/L)*sum((1/Qeq));
Bc=1+Av*sqrt(Qinv_bar);

%calculation of actmin(i,k) and actmin_sub(i,k)
k_mod=zeros([L 1]);
k_mod(find(P.*Bmin.*Bc<actmin))=1;
actmin_sub(find(k_mod))=P(find(k_mod)).*Bmin_sub(find(k_mod)).*Bc;
actmin(find(k_mod))=P(find(k_mod)).*Bmin(find(k_mod)).*Bc;

if subwc==V
    
    %check whether the minimum is local minimum or not
    lmin_flag(find(k_mod))=0;
    
    %storing the value of actmin(i,k)
    minact(:,u)=actmin;

    %calculation of Pmin_u the minimum of the last U stored values of actmin
    Pmin_u=min(minact,[],2);
        
    %calculation of noise slope max
    if Qinv_bar<0.03
        noise_slope_max=8;
    elseif Qinv_bar<0.05
        noise_slope_max=4;
    elseif Qinv_bar<0.06
        noise_slope_max=2;
    else
        noise_slope_max=1.2;
    end
    
    %update Pmin_u if the minimum falls within the search range
    test=find((lmin_flag & (actmin_sub<(noise_slope_max*Pmin_u)) & (actmin_sub>Pmin_u)));
    Pmin_u(test)=actmin_sub(test);
    %noise_est=min(actmin_sub,Pmin_u);
    for x=1:Um
        minact(test,x)=actmin_sub(test);
    end
    actmin(test) = actmin_sub(test);
    lmin_flag(:)=0;
    subwc=1;
    actmin=P;
    actmin_sub=P;   
    if u == Um
        u=1;
    else
        u=u+1;
    end
else
    if subwc>1
        lmin_flag(find(k_mod))=1;
        noise_est=min(actmin_sub,Pmin_u);
        Pmin_u=noise_est;                        
    end
    %noise_est=min(actmin_sub,Pmin_u);    
    subwc=subwc+1;
end

parameters.alpha_corr = alpha_corr;
parameters.alpha = alpha;
parameters.P = P;
parameters.Pbar = Pbar;
parameters.Psqbar = Psqbar;
parameters.actmin = actmin;
parameters.actmin_sub = actmin_sub;
parameters.minact = minact;
parameters.Pmin_u = Pmin_u;
parameters.subwc = subwc;
parameters.u = u;
parameters.lmin_flag = lmin_flag;
parameters.n = n+1;
parameters.L = L;
parameters.R = R;
parameters.Um = Um;
parameters.V = V;
parameters.D = D;
parameters.Av = Av;
parameters.alpha_max = alpha_max;
parameters.alpha_min = alpha_min;
parameters.beta_max = beta_max;
parameters.M_D = M_D;
parameters.M_V = M_V;
parameters.H_D = H_D;
parameters.H_V = H_V;
parameters.noise_ps = noise_est;