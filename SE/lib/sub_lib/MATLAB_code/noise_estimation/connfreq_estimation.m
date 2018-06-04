function parameters = connfreq_estimation(ns_ps,parameters)

% parameters = struct('n',2,'len',len_val,'ad',0.95,'as',0.8,'ap',0.2,'beta',0.8,'beta1',0.98,'gamma',0.998,'alpha',0.7,...
%             'pk',zeros(len_val,1),'noise_ps',ns_ps,'pxk_old',ns_ps,'pxk',ns_ps,'pnk_old',ns_ps,'pnk',ns_ps);

D = parameters.D;
len = parameters.len;
b = parameters.b;
V = parameters.V;
U = parameters.U;
gamma1 = parameters.gamma1;
gamma2 = parameters.gamma2;
alpha_max = parameters.alpha_max;
beta_min = parameters.beta_min;

alpha_c = parameters.alpha_c;
noise_ps = parameters.noise_ps;
Rmin_old = parameters.Rmin_old;
Pmin = parameters.Pmin;
Pmin_sw = parameters.Pmin_sw;
P = parameters.SmthdP;
u1 = parameters.u1;
j = parameters.j;
stored_min = parameters.stored_min;

P_noise_est = zeros(len,1);

%Spectral smoothing
%by equation(4)
% P_vector = zeros(2*D+1,1);
% P_y = ns_ps; 
% for k = 1:len
%     if k>D&k<=(len - D)
%         for i = 1:2*D+1
%             P_vector(i) = ns_ps(k-D+i-1);%.^2;
%         end
%         P_y(k) = b*P_vector;
%     end
% end

P_y=smoothing(ns_ps,b,D);   % spectral smoothing according to Eq. 4

R = sum(P)/sum(ns_ps);
alpha_c_tild = 1/(1+(R-1)^2);
alpha_c = alpha_c*0.7 + 0.3*max(alpha_c_tild, 0.7);
alpha = (alpha_max*alpha_c)./((P./(noise_ps+eps) -1).^2 +1);

%
%Temporal smoothing
power_min = sum(Pmin);
power_noise_ps = sum(noise_ps); %Eq.(8)

%Pold=P; Pminold=Pmin; Pmin_swold=Pmin_sw;
%Decision=zeros(len,1);
% for k = 1:len
%     P(k) = alpha(k)*P(k) + (1-alpha(k))*P_y(k); %Eq.(5)
%     %Speech presence decision
%     if P(k)<Pmin(k)
%         Pmin(k) = P(k);
%     end
%     if P(k)<Pmin_sw(k)
%         Pmin_sw(k) = P(k);
%     end
%     if P(k)>gamma1*Pmin(k)
%         D_1(k) = 1;
%     else
%         D_1(k) = 0;
%     end
% 
%     if P(k)>(Pmin(k) + gamma2*power_min/len)
%         D_2(k) = 1;
%     else
%         D_2(k) = 0;
%     end
%     Decision(k) = D_1(k)*D_2(k);
% end

Decision=zeros(len,1);
P=alpha.*P+(1-alpha).*P_y;
%Pmin2=Pminold;
Pmin=min(Pmin,P);
%Pmin_sw2=Pmin_swold;
Pmin_sw=min(Pmin_sw,P);
D_1a=zeros(len,1);
indx=find(P>gamma1*Pmin);
if ~isempty(indx),D_1a(indx)=1; end;
D_2a=zeros(len,1);
indx2=find(P>(Pmin+gamma2*power_min/len));
if ~isempty(indx2), D_2a(indx2)=1; end;
Decision=D_1a.*D_2a;


%Noise periodogram estimation
Rmin_tild = power_noise_ps/(power_min+eps); % Bias factor

if sum(Decision)>0
    Rmin = Rmin_old;
else
    Rmin = beta_min*Rmin_old + (1-beta_min)*Rmin_tild;  %Eq.(18)
end

% for k = 1:len
%     if Decision(k)==1
%         noise_ps(k) = Rmin*Pmin(k);
%     else
%         noise_ps(k) = ns_ps(k);%
%     end
% end

noise_ps=ns_ps;
indd=find(Decision==1);
if ~isempty(indd), noise_ps(indd)=Rmin*Pmin(indd);, end;


%Temporal minimum tracking
%use window to find the minimum
j = j+1;
if j==V
    stored_min(:,u1) = Pmin_sw;
    u1 = u1+1;
    if u1==U+1; 
        u1=1;
    end
    Pmin = min(stored_min,[],2);
    Pmin_sw = P;
    j = 0;
end
%

parameters.alpha_c = alpha_c;
parameters.noise_ps = noise_ps;
parameters.Rmin_old = Rmin;
parameters.Pmin = Pmin;
parameters.Pmin_sw = Pmin_sw;
parameters.SmthdP = P;
parameters.u1 = u1;
parameters.j = j;
parameters.alpha = alpha;
parameters.stored_min = stored_min;
parameters.Decision = Decision;


% ----------------------------------------------
function y=smoothing (x,win,N);


len=length(x);
win1=win(1:N+1);
win2=win(N+2:2*N+1);
y1=filter(fliplr(win1),[1],x);

x2=zeros(len,1);
x2(1:len-N)=x(N+1:len);

y2=filter(fliplr(win2),[1],x2);

y=(y1+y2); 
