function parameters = cohenMCRA_estimation(ns_ps,parameters);

as = parameters.as;
ad = parameters.ad;
ap = parameters.ap;
pk = parameters.pk;
delta = parameters.delta;
L = parameters.L;
n = parameters.n;
len = parameters.len;
noise_ps = parameters.noise_ps;
P = parameters.P;
Pmin = parameters.Pmin;
Ptmp = parameters.Ptmp;

P=as*P+(1-as)*ns_ps;  % Eq. 7 

if rem(n,L)==0
    Pmin=min(Ptmp,P);  % Eq. 10
    Ptmp=P;            % Eq. 11
else
    Pmin=min(Pmin,P); % Eq. 8
    Ptmp=min(Ptmp,P); % Eq. 9
end

Srk=P./Pmin; 

Ikl=zeros(len,1);
ikl_indx=find(Srk > delta);
Ikl(ikl_indx)=1;
pk = ap*pk+(1-ap)*Ikl;  % Eq. 14 
adk = ad+(1-ad)*pk;  % Eq. 5
noise_ps=adk.*noise_ps + (1-adk).*ns_ps;  % Eq. 4

parameters.pk = pk;
parameters.n = n+1;
parameters.noise_ps = noise_ps;
parameters.P = P;
parameters.Pmin = Pmin;
parameters.Ptmp = Ptmp;