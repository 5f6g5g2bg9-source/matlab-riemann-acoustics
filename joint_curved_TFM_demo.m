%% joint_curved_TFM_demo.m
% Joint simulation:
%   (1) Nonuniform temperature -> sound-speed field -> Riemann metric
%       and geodesic/curved-ray travel time.
%   (2) FMC/TFM simulation and comparison of straight-ray and curved-ray
%       delay laws.
%
% This is a self-contained 2-D x-z demonstrator. It is deliberately written
% in 2-D because a full 3-D all-pair ray solve is computationally expensive.
% The same functions can be extended to x-y-z by adding one coordinate.
%
% Units: m, s, Hz, degC.
% MATLAB: R2019b or newer (local functions in scripts supported).
%
% Important model choice:
% The synthetic FMC data are generated using the curved-ray travel-time model.
% Standard TFM images the data with straight/constant-speed delays; corrected
% TFM images them with the same curved-ray travel-time model.

clear; close all; clc;
rng(4);

%% 1. Domain, temperature and acoustic model
Lx = 0.040;                         % 40 mm lateral aperture
Lz = 0.060;                         % 60 mm depth
nx = 161; nz = 241;
x = linspace(0,Lx,nx);
z = linspace(0,Lz,nz);
[X,Z] = meshgrid(x,z);

Tmin = 20; Tmax = 2500;
T = temperatureField(X,Z,Tmin,Tmax);

% Approximate isotropic gas-like law. For a solid, replace this with
% c(T)=sqrt((lambda(T)+2*mu(T))/rho(T)) using material data.
c_ref = 5900; T_ref = 20;
c = c_ref .* sqrt((T+273.15)/(T_ref+273.15));
c = min(max(c,1000),12000);

% Riemann metric for acoustic rays: ds_R^2 = n(x)^2 (dx^2+dz^2), n=1/c.
% The geodesic equation is integrated below in the equivalent ray ODE form.
n = 1./c;
[dndx,dndz] = gradient(n,x,z);

fprintf('T range: %.2f ... %.2f degC\n',min(T(:)),max(T(:)));
fprintf('c range: %.1f ... %.1f m/s\n',min(c(:)),max(c(:)));

figure('Color','w','Name','Medium');
subplot(1,2,1); imagesc(x*1e3,z*1e3,T); axis image ij; colorbar;
xlabel('x [mm]'); ylabel('z [mm]'); title('Temperature T(x,z) [degC]');
subplot(1,2,2); imagesc(x*1e3,z*1e3,c); axis image ij; colorbar;
xlabel('x [mm]'); ylabel('z [mm]'); title('Sound speed c(x,z) [m/s]');

%% 2. Array and point reflector
nel = 24; pitch = 0.8e-3;
xe = (0:nel-1)*pitch + (Lx-(nel-1)*pitch)/2;
ze = zeros(1,nel) + 1e-3;       % array at shallow boundary
f0 = 2.0e6;                     % center frequency
fs = 40e6;                      % RF sample rate
tmax = 2*Lz/min(c(:)) + 5/f0;
t = (0:ceil(tmax*fs))/fs;

% One point target (use several targets for PSF/contrast studies).
target = [Lx*0.58, 0.038];

%% 3. Travel-time tables: straight and curved
% Tables are one-way element -> every image pixel.
% Curved travel time is obtained by ray shooting in the slowness field.
% To keep runtime reasonable, use a coarser ray table and interpolate it.
rayStride = 4;
xc = x(1:rayStride:end); zc = z(1:rayStride:end);
[Xc,Zc] = meshgrid(xc,zc);
ntp = numel(Xc);

tauStraight = zeros(nel,ntp);
tauCurved = zeros(nel,ntp);

fprintf('Computing travel-time tables for %d elements x %d pixels...\n',nel,ntp);
for ie=1:nel
    for ip=1:ntp
        p0 = [xe(ie),ze(ie)]; p1 = [Xc(ip),Zc(ip)];
        tauStraight(ie,ip) = norm(p1-p0)/mean(c(:));
        tauCurved(ie,ip) = curvedTravelTime(p0,p1,x,z,c,dndx,dndz);
    end
    fprintf('  element %d/%d\n',ie,nel);
end

% Interpolate tables to the full image grid.
tauS = zeros(nel,nz,nx); tauC = zeros(nel,nz,nx);
for ie=1:nel
    tauS(ie,:,:) = reshape(interp2(Xc,Zc,tauStraight(ie,:),X,Z,'linear','extrap'),nz,nx);
    tauC(ie,:,:) = reshape(interp2(Xc,Zc,tauCurved(ie,:),X,Z,'linear','extrap'),nz,nx);
end

%% 4. Generate FMC data with the curved model
% A narrow-band pulse is delayed by tau_tx(target)+tau_rx(target).
% This is a synthetic FMC model, not a full elastodynamic solver.
nt = numel(t);
rf = zeros(nel,nel,nt);
amp = 1 ./ max(1e-6,(tauC(:,findNearest(x,target(1)),findNearest(z,target(2)))+1e-5));
% Compute target one-way times directly, avoiding indexing ambiguity.
targetTau = zeros(nel,1);
for ie=1:nel
    targetTau(ie) = curvedTravelTime(xe(ie),ze(ie),x,z,c,dndx,dndz);
end
pulse = @(tt) sin(2*pi*f0*tt).*exp(-(pi*f0*tt/2.2).^2);
for itx=1:nel
    for irx=1:nel
        tau2 = targetTau(itx)+targetTau(irx);
        rf(itx,irx,:) = pulse(t-tau2) ./ sqrt(max(tau2,1e-8));
    end
end
% Add low-level noise.
rf = rf + 0.01*max(abs(rf(:)))*randn(size(rf));

%% 5. TFM with two delay laws
% Delay-and-sum samples the FMC data at tau_tx+tau_rx.
% Linear interpolation is used; no toolbox is required.
fprintf('Beamforming standard and corrected TFM...\n');
imgS = tfmDAS(rf,t, tauS, tauS);
imgC = tfmDAS(rf,t, tauC, tauC);
imgS = abs(hilbert2D(imgS)); imgC = abs(hilbert2D(imgC));
imgS = imgS/max(imgS(:)); imgC = imgC/max(imgC(:));

%% 6. Metrics and plots
[~,iS] = max(imgS(:)); [izS,ixS] = ind2sub(size(imgS),iS);
[~,iC] = max(imgC(:)); [izC,ixC] = ind2sub(size(imgC),iC);
fprintf('Standard peak:  x=%.2f mm, z=%.2f mm\n',x(ixS)*1e3,z(izS)*1e3);
fprintf('Corrected peak: x=%.2f mm, z=%.2f mm\n',x(ixC)*1e3,z(izC)*1e3);
fprintf('True target:    x=%.2f mm, z=%.2f mm\n',target(1)*1e3,target(2)*1e3);

figure('Color','w','Name','TFM comparison','Position',[100 100 1300 780]);
subplot(2,3,1); showDB(imgS,x,z,'Standard TFM: straight delays');
subplot(2,3,2); showDB(imgC,x,z,'Corrected TFM: curved delays');
subplot(2,3,3); imagesc(x*1e3,z*1e3,20*log10(imgC+1e-4)-20*log10(imgS+1e-4));
axis image ij; colorbar; xlabel('x [mm]'); ylabel('z [mm]'); title('Corrected - standard [dB]');
subplot(2,3,4); plot(x*1e3,imgS(izS,:),'b','LineWidth',1.5); hold on;
plot(x*1e3,imgC(izC,:),'r','LineWidth',1.5); grid on; xlabel('x [mm]'); ylabel('normalized');
legend('standard','corrected'); title('Lateral peak profiles');
subplot(2,3,5); plot(z*1e3,imgS(:,ixS),'b','LineWidth',1.5); hold on;
plot(z*1e3,imgC(:,ixC),'r','LineWidth',1.5); grid on; xlabel('z [mm]'); ylabel('normalized');
legend('standard','corrected'); title('Depth peak profiles');
subplot(2,3,6); plotRayExample(xe(ceil(nel/2)),ze(ceil(nel/2)),target,x,z,c);

% Show a time/phase mismatch statistic at the true target.
straightTarget = zeros(nel,1);
for ie=1:nel, straightTarget(ie)=norm(target-[xe(ie),ze(ie)])/mean(c(:)); end
fprintf('RMS one-way delay error of straight model at target: %.3f ns\n', ...
    rms(straightTarget-targetTau)*1e9);

%% Local functions
function T = temperatureField(X,Z,Tmin,Tmax)
    % Smooth field with gradient, hot spot, and sinusoidal perturbation.
    u = 0.15 + 0.55*X/max(X(:)) + 0.30*Z/max(Z(:));
    hot = 0.65*exp(-((X-0.026).^2/(0.006^2)+(Z-0.034).^2/(0.012^2)));
    ripple = 0.10*sin(2*pi*X/max(X(:))).*sin(3*pi*Z/max(Z(:)));
    q = min(max(u+hot+ripple,0),1);
    T = Tmin + (Tmax-Tmin)*q;
end

function tau = curvedTravelTime(p0,p1,x,z,c,dcdx,dcdz)
    % Curved ray by shooting a first-order ray equation in Cartesian space.
    % For isotropic slowness n=1/c, with path parameter s approximately
    % geometric arc length: dr/ds = q, dq/ds = (grad(log n)-q*(q.grad(log n)))/n.
    % The initial direction is iteratively adjusted so the endpoint is p1.
    if norm(p1-p0)<1e-12, tau=0; return; end
    d = (p1-p0)/norm(p1-p0); L=norm(p1-p0);
    for k=1:5
        [qend,rend,tt] = integrateRay(p0,d,L*1.8,x,z,c,dcdx,dcdz,p1);
        err = p1-rend;
        if norm(err)<2e-5, break; end
        d = d + 0.35*err/max(L,eps); d=d/norm(d);
    end
    [~,~,tau] = integrateRay(p0,d,L*1.8,x,z,c,dcdx,dcdz,p1);
end

function [q,r,tau] = integrateRay(p0,d,L,x,z,c,dcdx,dcdz,p1)
    % Fixed-step RK4 ray integration; stop when close to endpoint or boundary.
    h = max(min(L/100,0.0005),1e-5); N=ceil(L/h); h=L/N;
    r=p0(:); v=d(:); tau=0;
    for k=1:N
        f=@(rr,vv) rayRHS(rr,vv,x,z,c,dcdx,dcdz);
        k1r=v; k1v=f(r,v); k1t=interp2(x,z,1./c,rrSafe(r(1)),rrSafe(r(2)),'linear',1/mean(c(:)));
        k2r=v+0.5*h*k1v; k2v=f(r+0.5*h*k1r,v+0.5*h*k1v); k2t=interp2(x,z,1./c,rrSafe(r(1)+0.5*h*k1r(1)),rrSafe(r(2)+0.5*h*k1r(2)),'linear',1/mean(c(:)));
        k3r=v+0.5*h*k2v; k3v=f(r+0.5*h*k2r,v+0.5*h*k2v); k3t=interp2(x,z,1./c,rrSafe(r(1)+0.5*h*k2r(1)),rrSafe(r(2)+0.5*h*k2r(2)),'linear',1/mean(c(:)));
        k4r=v+h*k3v; k4v=f(r+h*k3r,v+h*k3v); k4t=interp2(x,z,1./c,rrSafe(r(1)+h*k3r(1)),rrSafe(r(2)+h*k3r(2)),'linear',1/mean(c(:)));
        r=r+h*(k1r+2*k2r+2*k3r+k4r)/6; v=v+h*(k1v+2*k2v+2*k3v+k4v)/6; v=v/norm(v);
        tau=tau+h*(k1t+2*k2t+2*k3t+k4t)/6;
        if norm(r(:)-p1(:))<max(2*h,2e-5), break; end
        if r(1)<min(x)||r(1)>max(x)||r(2)<min(z)||r(2)>max(z), break; end
    end
    q=v; r=r(:).';
end

function a=rayRHS(r,v,x,z,c,dcdx,dcdz)
    cc=interp2(x,z,c,rrSafe(r(1)),rrSafe(r(2)),'linear',mean(c(:)));
    gx=interp2(x,z,dcdx,rrSafe(r(1)),rrSafe(r(2)),'linear',0)/cc;
    gz=interp2(x,z,dcdz,rrSafe(r(1)),rrSafe(r(2)),'linear',0)/cc;
    gradlogc=[gx;gz]; a=gradlogc-v*(v.'*gradlogc);
end

function v=rrSafe(v), v=v; end
function ind=findNearest(a,val), [~,ind]=min(abs(a-val)); end

function image=tfmDAS(rf,t,tauTx,tauRx)
    [nel,~,~]=size(rf); [~,nz,nx]=size(tauTx); image=zeros(nz,nx);
    for iz=1:nz
        for ix=1:nx
            s=0;
            for itx=1:nel
                for irx=1:nel
                    td=tauTx(itx,iz,ix)+tauRx(irx,iz,ix);
                    s=s+interp1(t,squeeze(rf(itx,irx,:)),td,'linear',0);
                end
            end
            image(iz,ix)=s/(nel^2);
        end
    end
end

function out=hilbert2D(a)
    % Analytic signal along the first dimension without Signal Toolbox.
    n=size(a,1); A=fft(a,[],1); h=zeros(n,1);
    if mod(n,2)==0, h([1 n/2+1])=1; h(2:n/2)=2;
    else, h(1)=1; h(2:(n+1)/2)=2; end
    out=ifft(A.*h,[],1);
end

function showDB(im,x,z,str)
    q=20*log10(im/max(im(:))+1e-4); imagesc(x*1e3,z*1e3,max(q,-40));
    axis image ij; caxis([-40 0]); colorbar; xlabel('x [mm]'); ylabel('z [mm]'); title(str);
end

function plotRayExample(p0x,p0z,p1,x,z,c)
    % Visual diagnostic only: plot a ray by rerunning a coarse path.
    imagesc(x*1e3,z*1e3,c); axis image ij; colorbar; hold on;
    plot([p0x p1(1)]*1e3,[p0z p1(2)]*1e3,'w--','LineWidth',1.2);
    plot(p0x*1e3,p0z*1e3,'kv','MarkerFaceColor','y'); plot(p1(1)*1e3,p1(2)*1e3,'ko','MarkerFaceColor','r');
    xlabel('x [mm]'); ylabel('z [mm]'); title('Ray diagnostic');
end
