function parameters = noise_estimation(noisy_ps,method,parameters)
switch lower(method)
    case 'martin'
        parameters = martin_estimation(noisy_ps,parameters);
    case 'mcra'
        parameters = mcra_estimation(noisy_ps,parameters);
    case 'imcra'
        parameters = imcra_estimation(noisy_ps,parameters);
    case 'doblinger'
        parameters = doblinger_estimation(noisy_ps,parameters);
    case 'hirsch'
        parameters = hirsch_estimation(noisy_ps,parameters);
    case 'mcra2'
        parameters = mcra2_estimation(noisy_ps,parameters);
    case 'conn_freq'
        parameters = connfreq_estimation(noisy_ps,parameters);
end
return;