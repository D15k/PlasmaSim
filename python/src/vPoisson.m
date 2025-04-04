function [Efield] = vPoisson(fs, grids, charge)
    
% compute charge density
rho = 0;
Ns = length(grids); %CB number of species, but why not use params.Ns?
for s = 1:Ns
    rho = rho + charge(s)*sum(fs(:,:,s))*grids(s).dv;
end

Nx = grids(1).Nx;
Lx = grids(1).Lx;
kx = (2*pi/Lx) * [-Nx/2:Nx/2-1];
kx = fftshift(kx);

% laplacian is division -|k|^2
K2 = kx.^2;
K2(1) = 1; % avoid devision by 0
% to avoid a division by zero, we set the zeroth wavenumber to one.
% this leaves it's respective Fourier coefficient unaltered, so the
% zero mode of Sk is conserved.dphi_dx_h = 1i*phi_fft.*kx(1,:); This way, Sk's zero mode implicitly
% defined the zero mode of the result
% Note that the zero mode is NOT uniquely defined: in a periodic
% setting, the solution of Laplace's (or Poisson's) equation is only
% defined up to a constant! You can freely overwrite the zero mode,
% therefore.
b = fft(rho); 
phi_fft =  -b ./(K2); % solves second equation of vlassov poisson
phi_fft(1) = 0; % set mean to zero
dphi_dx_h = 1i*phi_fft.*kx;
    Efield = -reshape(ifft(dphi_dx_h, "symmetric"), 1, []);
    
end