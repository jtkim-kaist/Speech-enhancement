function parameters = newmethod_estimation(ns_ps,parameters)

% parameters = struct('n',2,'len',len_val,'ad',0.95,'as',0.8,'ap',0.2,'beta',0.8,'beta1',0.98,'gamma',0.998,'alpha',0.7,...
%             'pk',zeros(len_val,1),'noise_ps',ns_ps,'pxk_old',ns_ps,'pxk',ns_ps,'pnk_old',ns_ps,'pnk',ns_ps);

n = parameters.n;
len = parameters.len;
ad = parameters.ad;
as = parameters.as;
ap = parameters.ap;
beta = parameters.beta;
gamma = parameters.gamma;
alpha = parameters.alpha;
pk = parameters.pk;
delta = parameters.delta;

noise_ps = parameters.noise_ps;
pxk = parameters.pxk;
pnk = parameters.pnk;
pxk_old = parameters.pxk_old;
pnk_old = parameters.pnk_old;

    pxk=alpha*pxk_old+(1-alpha)*(ns_ps);  
    pnk=pxk;
    index=find(pnk_old<pxk);
    pnk(index)=(gamma*pnk_old(index))+(((1-gamma)/(1-beta)).*...
         (pxk(index)-beta*pxk_old(index)));
    pxk_old=pxk;
    pnk_old=pnk;
    
    Srk=zeros(len,1);
    Srk=pxk./pnk;
    Srk_data(:,n)=Srk;
    Ikl=zeros(len,1);    
    ikl_indx=find(Srk > delta);
    Ikl(ikl_indx)=1;
    pk = ap*pk+(1-ap)*Ikl;  % Eq. 14 
    adk = ad+(1-ad)*pk;  % Eq. 5
    noise_ps=adk.*noise_ps + (1-adk).*pxk;  % Eq. 4

parameters.n = n+1;
parameters.pk = pk;
parameters.noise_ps = noise_ps;
parameters.pnk = pnk;
parameters.pnk_old = pnk_old;
parameters.pxk = pxk;
parameters.pxk_old = pxk_old;
