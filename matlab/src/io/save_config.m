function data = save_config(params,data,fs,Ns)


for s = 1:params.Ns
    data.fs(:,:,Ns,s) = fs(:,:,s);
    data.Efield(:,Ns) = params.Efield;
end

end