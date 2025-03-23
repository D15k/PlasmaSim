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
PARAMS_two_stream;
%PARAMS_landau_damping;
%PARAMS_ion_acoustic_waves;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% simulate
tic()
params.method = "NuFi";
[params, data] = Sim(params);
t(1) = toc()

tic()
params.method = "predcorr";
[params, data] = Sim(params);
t(2) = toc()

fprintf("tcpu Nufi: %2.2f\n", t(1))
fprintf("tcpu predcorr: %2.2f \n", t(2))



%% some plotting
figure(44)
h = plot_qty_vs_time(params,"Epot");
figure(45)
h = plot_qty_vs_time(params,"Etot");
