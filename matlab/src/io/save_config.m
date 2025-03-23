function data = save_config(params,data,fs,Ns,save_to_file)

if nargin < 5  % Check if save_to_file is provided
    save_to_file = 0;
end

for s = 1:params.Ns
    data.fs(:,:,Ns,s) = fs(:,:,s);
    data.Efield(:,Ns) = params.Efield;
end

% Save data to file if requested
if save_to_file == 1
    if isfield(params, 'data_dir')
        filename = fullfile(params.data_dir, 'config_data.mat');
        save(filename, 'data',"params");
        fprintf('Data saved to %s\n', filename);
    else
        error('params.data_dir is not defined or is not a valid string.');
    end
end
end