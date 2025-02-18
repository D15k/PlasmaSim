%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main script for CMM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all
clear all
clc
addpath(genpath('./src/'),genpath('./params/'))
DEFAULTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% select case:
%PARAMS_two_stream;
%PARAMS_landau_damping;
PARAMS_ion_acoustic_waves;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% simulate
tic()
params.method = "NuFi";
params = Sim(params);
t(1) = toc()

tic()
params.method = "predcorr";
params = Sim(params);
t(2) = toc()

fprintf("tcpu Nufi: %2.2f\n", t(1))
fprintf("tcpu predcorr: %2.2f \n", t(2))
%% some plotting
figure(44)
params.method = "predcorr"
h = plot_qty_vs_time(params,"Epot");
hold on
params.method="NuFi";
h = plot_qty_vs_time(params,"Epot");
for s = 1:params.Ns
h(s).LineStyle = "--";
end
legend()

figure(45)
params.method = "predcorr"
h = plot_qty_vs_time(params,"Etot");
hold on
params.method="NuFi";
h = plot_qty_vs_time(params,"Etot");
for s = 1:params.Ns
h(s).LineStyle = ":";
h(s).LineWidth=1;
h(s).Color = "k";
end
legend()

