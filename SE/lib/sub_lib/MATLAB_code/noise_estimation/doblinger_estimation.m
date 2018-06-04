function [parameters] = doblinger_estimation(ns_ps,parameters)

%         parameters = struct('n',2,'len',len_val,'alpha',0.7,'beta',0.96,'gamma',0.998,'noise_ps',ns_ps,'pxk_old',ns_ps,...
%             'pxk',ns_ps,'pnk_old',ns_ps,'pnk',ns_ps);

n = parameters.n;
len = parameters.len;
alpha = parameters.alpha;
beta = parameters.beta;
gamma = parameters.gamma;
noise_ps = parameters.noise_ps;

pxk_old = parameters.pxk_old;
pxk = parameters.pxk;
pnk_old = parameters.pnk_old;
pnk = parameters.pnk;

pxk=alpha*pxk_old+(1-alpha)*ns_ps;
for t=1:len
    if pnk_old(t)<=pxk(t)
        pnk(t)=(gamma.*pnk_old(t))+(((1-gamma)/(1-beta)).*(pxk(t)-beta.*pxk_old(t)));
    else
        pnk(t)=pxk(t);
    end
end
pxk_old=pxk;
pnk_old=pnk;
noise_ps = pnk;

parameters.n = n+1;
parameters.noise_ps = noise_ps;
parameters.pnk = pnk;
parameters.pnk_old = pnk_old;
parameters.pxk = pxk;
parameters.pxk_old = pxk_old;

