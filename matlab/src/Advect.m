function [f] = Advect(f,Efield,grid,dt)

f = Adv_x(f,grid,dt/2);
f = Adv_vx(f,grid,Efield,dt);
f = Adv_x(f,grid,dt/2);

end

function f = Adv_x(f,grid,dt)
X = grid.X;
V = grid.V;

X_new = mod(X - V*dt,grid.Lx-grid.dx);
for j = 1:size(X_new,1)
    f(j,:) = interp1(X(j,:),f(j,:),X_new(j,:),grid.method);
end

end

function f = Adv_vx(f,grid,Efield,dt)
V = grid.V;
Lv = grid.Lv;
V_new = mod(V + Efield*dt+Lv,2*grid.Lv-grid.dv)-Lv;
for j = 1:size(V,2)
    f(:,j) = interp1(V(:,j),f(:,j),V_new(:,j),grid.method);
end

end