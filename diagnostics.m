clear all
close all
clc
addpath(genpath('./src/'),genpath('./params/'))
DEFAULTS

PARAMS_landau_damping
%PARAMS_ion_acoustic_waves

% NuFi
params.method="NuFi";
params = Sim(params);

figure(21);
params.data_dir= "data/" + params.mycase+"_"+params.method+"/";
plot_diagnostics(params,'-');

% predictor corrector
params.method="predcorr";
%params = Sim(params);
params.data_dir= "data/" + params.mycase+"_"+params.method+"/";
plot_diagnostics(params,':')

%%
if params.k == 0.2
    x = linspace(0,2*pi/params.k,512)
    eps = 0.001
    w_r = 1.0640
    w_i = 5.510*1e-5
    r=1.129664
    phi = 0.001273771
elseif params.k ==0.5
    eps = params.alpha;
    w_r = 1.4156;
    w_i = -0.1533;
    r=0.3677;
    phi = 0.536245;
end
time = linspace(0,params.Tend,1000);
x =  [0:params.Nx-1] * params.Lx / params.Nx;
dx = x(2)-x(1);
[T,X] = meshgrid(time,x);
Electric_energy_analytic = 4 * eps * r * exp(w_i*T).*sin(params.k*X).*cos(w_r*T-phi);
Epot =0.5*sum(Electric_energy_analytic.^2)*dx;
subplot(3,2,3); hold on
semilogy(time,Epot,'--')















function plot_diagnostics(params, linestyle)

hold on;
species_colors = lines(params.Ns); % Assign different colors for each species

for s = 1:params.Ns
    species_name = params.species_name(s); % Species name
    filename = fullfile(params.data_dir, species_name+ '.csv');

    if exist(filename, 'file')
        filename
        % Read the diagnostic data
        data = readtable(filename);
        time = data.time; % Assuming each row is a new iteration
        data_spec{s} = data;


        % Plot various diagnorstics over iterations
        subplot(3,2,1); hold on;
        plot(time, data.Mass/data.Mass(1), 'Color', species_colors(s,:), 'DisplayName', species_name, 'LineStyle',linestyle);
        xlabel('time'); ylabel('rel Mass'); grid on;

        subplot(3,2,2); hold on;
        plot(time, data.L2norm/data.L2norm(1), 'Color', species_colors(s,:), 'DisplayName', species_name, 'LineStyle',linestyle);
        xlabel('time'); ylabel('rel $||f||^2_2$'); grid on;

        subplot(3,2,6); hold on;
        plot(time, data.Momentum, 'Color', species_colors(s,:), 'DisplayName', species_name, 'LineStyle',linestyle);
        title('Momentum'); xlabel('time'); ylabel('Momentum'); grid on;

        subplot(3,2,4); hold on;
        plot(time, data.Ekin, 'Color', species_colors(s,:), 'DisplayName', species_name, 'LineStyle',linestyle);
        title('Kinetic Energy'); xlabel('time'); ylabel('Ekin'); grid on;

        subplot(3,2,5); hold on;
        plot(time, data.Etot, 'Color', species_colors(s,:), 'DisplayName', species_name, 'LineStyle',linestyle);
        title('Total Energy'); xlabel('Iteration'); ylabel('Etot'); grid on;

    else
        warning('File %s does not exist. Skipping.', filename);
    end
    
end

subplot(3,2,3); hold on
    semilogy(data_spec{1}.time, data_spec{1}.Epot, 'k', 'LineStyle',linestyle);
    xlabel('time'); ylabel('Epot'); grid on;
    ylim([1e-13,data_spec{1}.Epot(1)])



% Add legends to all subplots
for i = 1:6
    subplot(3,2,i);
    legend('show');
end

%hold off;
end