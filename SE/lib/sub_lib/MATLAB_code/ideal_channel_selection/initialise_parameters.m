function parameters = initialise_parameters(ns_ps,Srate,method)
len_val = length(ns_ps);
switch lower(method)
    case 'martin'
        L_val=len_val; 
        R_val=len_val/2; 
        D_val=150; V_val=15; Um_val=10; Av_val=2.12; 
        alpha_max_val=0.96; 
        alpha_min_val=0.3; 
        beta_max_val=0.8;
        x_val=[1 2 5 8 10 15 20 30 40 60 80 120 140 160];
        Y_M_val=[0 .26 .48 .58 .61 .668 .705 .762 .8 .841 .865 .89 .9 .91];
        Y_H_val=[0 .15 .48 .78 .98 1.55 2.0 2.3 2.52 2.9 3.25 4.0 4.1 4.1];
        xi_val=D_val;
        M_D_val=interp1(x_val,Y_M_val,xi_val);
        H_D_val=interp1(x_val,Y_H_val,xi_val);
        xi_val=V_val;
        M_V_val=interp1(x_val,Y_M_val,xi_val);
        H_V_val=interp1(x_val,Y_H_val,xi_val);
        minact_val(1:L_val,1:Um_val)=max(ns_ps);
        parameters = struct('n',2,'len',len_val,'alpha_corr',0.96,'alpha',0.96*ones(len_val,1),'P',ns_ps,'noise_ps',ns_ps,'Pbar',ns_ps,...
            'Psqbar',ns_ps,'actmin',ns_ps,'actmin_sub',ns_ps,'Pmin_u',ns_ps,'subwc',2,'u',1,'minact',minact_val,'lmin_flag',zeros(len_val,1),...
            'L',L_val,'R',R_val,'D',D_val,'V',V_val,'Um',Um_val,'Av',Av_val,'alpha_max',alpha_max_val,'alpha_min',alpha_min_val,...
            'beta_max',beta_max_val,'Y_M',Y_M_val,'Y_H',Y_H_val,'M_D',M_D_val,'H_D',H_D_val,'M_V',M_V_val,'H_V',H_V_val);
    case 'mcra'
        parameters = struct('n',2,'len',len_val,'P',ns_ps,'Pmin',ns_ps,'Ptmp',ns_ps,'pk',zeros(len_val,1),'noise_ps',ns_ps,...
            'ad',0.95,'as',0.8,'L',round(1000*2/20),'delta',5,'ap',0.2);
    case 'imcra'
        alpha_d_val=0.85;
        alpha_s_val=0.9;
        U_val=8;V_val=15;
        Bmin_val=1.66;gamma0_val=4.6;gamma1_val=3;
        psi0_val=1.67;alpha_val=0.92;beta_val=1.47;
        j_val=0;
        b_val=hanning(3);
        B_val=sum(b_val);
        b_val=b_val/B_val;
        Sf_val=zeros(len_val,1);Sf_tild_val=zeros(len_val,1);
        Sf_val(1) = ns_ps(1);
        for f=2:len_val-1
            Sf_val(f)=sum(b_val.*[ns_ps(f-1);ns_ps(f);ns_ps(f+1)]);
        end
        Sf_val(len_val)=ns_ps(len_val);
        Sf_tild_val = zeros(len_val,1);
        parameters = struct('n',2,'len',len_val,'noise_ps',ns_ps,'noise_tild',ns_ps,'gamma',ones(len_val,1),'Sf',Sf_val,...
            'Smin',Sf_val,'S',Sf_val,'S_tild',Sf_val,'GH1',ones(len_val,1),'Smin_tild',Sf_val,'Smin_sw',Sf_val,'Smin_sw_tild',Sf_val,...
            'stored_min',max(ns_ps)*ones(len_val,U_val),'stored_min_tild',max(ns_ps)*ones(len_val,U_val),'u1',1,'u2',1,'j',2,...
            'alpha_d',0.85,'alpha_s',0.9,'U',8,'V',15,'Bmin',1.66,'gamma0',4.6,'gamma1',3,'psi0',1.67,'alpha',0.92,'beta',1.47,...
            'b',b_val,'Sf_tild',Sf_tild_val);
    case 'doblinger'
        parameters = struct('n',2,'len',len_val,'alpha',0.7,'beta',0.96,'gamma',0.998,'noise_ps',ns_ps,'pxk_old',ns_ps,...
            'pxk',ns_ps,'pnk_old',ns_ps,'pnk',ns_ps);
    case 'hirsch'
        parameters = struct('n',2,'len',len_val,'as',0.85,'beta',1.5,'omin',1.5,'noise_ps',ns_ps,'P',ns_ps);
    case 'mcra2'
        freq_res=Srate/len_val;
        k_1khz=floor(1000/freq_res);
        k_3khz=floor(3000/freq_res);
        %delta_val=[2*ones(k_1khz,1);2*ones(k_3khz-k_1khz,1);5*ones(len_val/2-k_3khz,1);...
        %    5*ones(len_val/2-k_3khz,1);2*ones(k_3khz-k_1khz,1);2*ones(k_1khz,1)];
         delta_val=[2*ones(k_1khz,1);2*ones(k_3khz-k_1khz,1);5*ones(len_val/2-k_3khz,1)];
			delta_val=[delta_val;5;flipud(delta_val(2:end))];

        parameters = struct('n',2,'len',len_val,'ad',0.95,'as',0.8,'ap',0.2,'beta',0.8,'beta1',0.98,'gamma',0.998,'alpha',0.7,...
            'delta',delta_val,'pk',zeros(len_val,1),'noise_ps',ns_ps,'pxk_old',ns_ps,'pxk',ns_ps,'pnk_old',ns_ps,'pnk',ns_ps);
        
      case 'conn_freq'
        D = 7; 
        b = triang(2*D+1)/sum(triang(2*D+1));
        b = b';
        beta_min = 0.7; % for R's recursion
        U = 5;
        V = 8;
        gamma1 = 6; 
        gamma2 = 0.5; 
        K_tild = 2*sum(b.^2)^2/sum(b.^4);
        alpha_max_val=0.96; 
        alpha_min_val=0.3;
        stored_min = max(ns_ps)*ones(len_val,U);
        
        alpha_c = 0.7;
        noise_ps = ns_ps;
        Rmin_old = 1;
        Pmin_sw = ns_ps;
        Pmin = ns_ps;
        P = ns_ps;
        Decision = zeros(size(P));
        u1 = 1;
        j = 0;
        parameters = struct('len',len_val,'D',D,'b',b,'U',U,'V',V,'gamma1',gamma1,'gamma2',gamma2,'K_tild',K_tild,'alpha_c',alpha_c,...
            'noise_ps',noise_ps,'Rmin_old',Rmin_old,'Pmin_sw',Pmin_sw,'Pmin',Pmin,'SmthdP',P,'u1',u1,'j',j,'alpha',0,'alpha_max',alpha_max_val,...
            'stored_min',stored_min,'beta_min',beta_min,'Decision',Decision);

    otherwise
            error('Method not implemented. Check spelling.');
end