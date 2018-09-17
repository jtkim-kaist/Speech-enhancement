function parameters = hirsch_estimation(ns_ps,parameters)

%         parameters = struct('n',2,'len',len_val,'as',0.85,'as1',0.7,'beta',1.5,'omin',1.5,'noise_ps',ns_ps,'P',ns_ps);

n = parameters.n;
len = parameters.len;
as = parameters.as;
beta = parameters.beta;
omin = parameters.omin;

noise_ps = parameters.noise_ps;
P = parameters.P;

P=as*P+(1-as)*ns_ps;
index=find(P<beta*noise_ps);
noise_ps(index)=as*noise_ps(index)+(1-as)*P(index);

parameters.P = P;
parameters.noise_ps = noise_ps;