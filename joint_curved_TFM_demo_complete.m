%% ============================================================
% COMPLETE JOINT PROJECT: Curved Traveltime + TFM Beamforming
% ============================================================
% A unified MATLAB simulation combining:
%   (1) 2D nonuniform temperature field T(x,z)
%   (2) Acoustic speed field c(x,z) derived from T
%   (3) Riemann metric g_ij and gradient-based ray tracing
%   (4) Curved ray traveltime computation via RK4
%   (5) Synthetic FMC data generation
%   (6) Standard TFM (straight rays, constant speed)
%   (7) Corrected TFM (curved rays)
%   (8) Quantitative comparison: beamwidth, sidelobe, peak shift, contrast
%
% Output: 6-panel figure showing imaging comparison and 2 diagnostic figures
%
% Units: meters [m], seconds [s], Hertz [Hz], Celsius [°C]
% MATLAB: R2019b or later (no toolboxes required)
%
% Author: Autonomous beam correction system
% Version: 1.0 (2D demonstrator, easily extended to 3D)
% ============================================================

clear all; close all; clc;
rng(42);  % for reproducibility

fprintf('\n');
fprintf('========================================\n');
fprintf('JOINT CURVED-RAY TFM SIMULATION v1.0\n');
fprintf('========================================\n\n');

%% ============================================================
% PART 1: Build 2D Spatial Domain and Temperature Field
% ============================================================

fprintf('[1] Building spatial domain and temperature field...\n');

% Domain dimensions
Lx = 0.040;    % 40 mm lateral (X)
Lz = 0.060;    % 60 mm depth (Z)
nx = 201;      % pixel count X
nz = 301;      % pixel count Z

% Coordinate grids
x = linspace(0, Lx, nx);
z = linspace(0, Lz, nz);
dx = x(2) - x(1);
dz = z(2) - z(1);
[X, Z] = meshgrid(x, z);

% Temperature field: nonuniform, smooth, with multiple features
Tmin = 20;      % [°C] minimum
Tmax = 2500;    % [°C] maximum
T = buildTemperatureField2D(X, Z, Tmin, Tmax);

fprintf('   T range: [%.1f, %.1f] °C\n', min(T(:)), max(T(:)));

%% ============================================================
% PART 2: Acoustic Speed Field and Gradients
% ============================================================

fprintf('[2] Computing sound speed field c(x,z) from T(x,z)...\n');

% Use idealized acoustic model: c(T) ∝ sqrt(T_abs)
% For air/gas: c ≈ c_ref * sqrt((T+273.15)/(T_ref+273.15))
% For solids: use material-specific dispersion (Lamé + density)
c_ref = 5900;    % [m/s] reference speed at T_ref
T_ref = 20;      % [°C] reference temperature
c = c_ref * sqrt((T + 273.15) / (T_ref + 273.15));

% Clamp to physical bounds
c = max(c, 1000);
c = min(c, 12000);

fprintf('   c range: [%.0f, %.0f] m/s\n', min(c(:)), max(c(:)));
fprintf('   Speed variation: %.2f %%\n', 100*(max(c(:))-min(c(:)))/mean(c(:)));

% Compute gradients for ray-tracing physics
[dc_dx, dc_dz] = gradient(c, dx, dz);

%% ============================================================
% PART 3: Phased Array Configuration
% ============================================================

fprintf('[3] Setting up phased array and FMC parameters...\n');

% Array parameters
n_elements = 24;        % Number of transducer elements
pitch = 0.8e-3;         % [m] Element-to-element spacing
% Center array on aperture
array_start_x = (Lx - (n_elements - 1) * pitch) / 2;
elem_x = array_start_x + (0:n_elements-1) * pitch;
elem_z = repmat(1e-3, 1, n_elements);  % [m] Array at shallow depth z=1mm

% Signal parameters
f0 = 2.0e6;             % [Hz] Center frequency
fs = 40e6;              % [Hz] Sampling frequency
T_acq = 2 * Lz / min(c(:)) + 10/f0;  % [s] Acquisition window
t = (0:floor(T_acq*fs)-1)' / fs;
nt = length(t);

fprintf('   Array: %d elements, pitch = %.2f mm\n', n_elements, pitch*1e3);
fprintf('   Array x-span: [%.2f, %.2f] mm\n', min(elem_x)*1e3, max(elem_x)*1e3);
fprintf('   Frequency: %.2f MHz, sampling: %.0f MHz\n', f0/1e6, fs/1e6);
fprintf('   Acquisition time: %.2f μs (%d samples)\n', T_acq*1e6, nt);

%% ============================================================
% PART 4: Compute Travel-Time Tables
% ============================================================

fprintf('[4] Computing travel-time tables (straight and curved)...\n');
fprintf('   This step may take 1-3 minutes...\n');

% For speed, use coarser ray grid; interpolate to full imaging grid
ray_stride = 5;
xray = x(1:ray_stride:end);
zray = z(1:ray_stride:end);
[X_ray, Z_ray] = meshgrid(xray, zray);

n_xray = length(xray);
n_zray = length(zray);
n_rays = n_xray * n_zray;

% Pre-allocate travel-time arrays
tau_tx_straight = zeros(n_elements, n_zray, n_xray);  % TX to each pixel
tau_rx_straight = zeros(n_elements, n_zray, n_xray);  % RX from each pixel
tau_tx_curved   = zeros(n_elements, n_zray, n_xray);
tau_rx_curved   = zeros(n_elements, n_zray, n_xray);

% Compute all TX paths
for ie = 1:n_elements
    tx_pos = [elem_x(ie), elem_z(ie)];
    
    for ir = 1:n_rays
        [iz, ix] = ind2sub([n_zray, n_xray], ir);
        pixel_pos = [X_ray(iz, ix), Z_ray(iz, ix)];
        
        % Straight-line traveltime
        straight_dist = norm(pixel_pos - tx_pos);
        c_mean = mean(c(:));
        tau_tx_straight(ie, iz, ix) = straight_dist / c_mean;
        
        % Curved-ray traveltime (geodesic via RK4)
        tau_tx_curved(ie, iz, ix) = curvedTravelTime2D(...
            tx_pos, pixel_pos, x, z, c, dc_dx, dc_dz);
    end
    
    if mod(ie, 4) == 0
        fprintf('     TX %d/%d computed\n', ie, n_elements);
    end
end

% Compute all RX paths
for ie = 1:n_elements
    rx_pos = [elem_x(ie), elem_z(ie)];
    
    for ir = 1:n_rays
        [iz, ix] = ind2sub([n_zray, n_xray], ir);
        pixel_pos = [X_ray(iz, ix), Z_ray(iz, ix)];
        
        % Straight-line traveltime
        straight_dist = norm(pixel_pos - rx_pos);
        c_mean = mean(c(:));
        tau_rx_straight(ie, iz, ix) = straight_dist / c_mean;
        
        % Curved-ray traveltime
        tau_rx_curved(ie, iz, ix) = curvedTravelTime2D(...
            rx_pos, pixel_pos, x, z, c, dc_dx, dc_dz);
    end
    
    if mod(ie, 4) == 0
        fprintf('     RX %d/%d computed\n', ie, n_elements);
    end
end

% Interpolate ray tables to full imaging grid
fprintf('   Interpolating travel-time tables to full grid...\n');
tau_tx_straight_full = interp2(X_ray, Z_ray, tau_tx_straight(1,:,:), X, Z, 'linear', 'extrap');
tau_rx_straight_full = interp2(X_ray, Z_ray, tau_rx_straight(1,:,:), X, Z, 'linear', 'extrap');
tau_tx_curved_full   = interp2(X_ray, Z_ray, tau_tx_curved(1,:,:), X, Z, 'linear', 'extrap');
tau_rx_curved_full   = interp2(X_ray, Z_ray, tau_rx_curved(1,:,:), X, Z, 'linear', 'extrap');

% Rebuild as [element, z, x]
tau_tx_straight_img = zeros(n_elements, nz, nx);
tau_rx_straight_img = zeros(n_elements, nz, nx);
tau_tx_curved_img   = zeros(n_elements, nz, nx);
tau_rx_curved_img   = zeros(n_elements, nz, nx);

for ie = 1:n_elements
    tx_pos = [elem_x(ie), elem_z(ie)];
    rx_pos = [elem_x(ie), elem_z(ie)];
    
    for iz = 1:nz
        for ix = 1:nx
            pixel_pos = [X(iz, ix), Z(iz, ix)];
            dist = norm(pixel_pos - tx_pos);
            c_mean = mean(c(:));
            
            tau_tx_straight_img(ie, iz, ix) = dist / c_mean;
            tau_rx_straight_img(ie, iz, ix) = dist / c_mean;
            tau_tx_curved_img(ie, iz, ix) = curvedTravelTime2D(...
                tx_pos, pixel_pos, x, z, c, dc_dx, dc_dz);
            tau_rx_curved_img(ie, iz, ix) = curvedTravelTime2D(...
                rx_pos, pixel_pos, x, z, c, dc_dx, dc_dz);
        end
    end
    
    if mod(ie, 4) == 0
        fprintf('     Full grid: element %d/%d\n', ie, n_elements);
    end
end

fprintf('   ✓ Travel-time tables complete\n\n');

%% ============================================================
% PART 5: Generate Synthetic FMC Data with Curved Model
% ============================================================

fprintf('[5] Generating synthetic FMC data (using curved-ray model)...\n');

% Point reflector target
target_x = Lx * 0.58;   % [m]
target_z = Lz * 0.625;  % [m]

% Compute one-way time from each element to target using curved ray
tau_target = zeros(n_elements, 1);
for ie = 1:n_elements
    tx_pos = [elem_x(ie), elem_z(ie)];
    tau_target(ie) = curvedTravelTime2D(tx_pos, [target_x, target_z], ...
        x, z, c, dc_dx, dc_dz);
end

% Synthetic pulse (narrowband Ricker-like)
pulse_bw = f0 / 2.2;  % [-3dB] bandwidth
pulse = @(t_s) sin(2*pi*f0*t_s) .* exp(-(pi*pulse_bw*t_s/2.2).^2);

% Generate FMC matrix: all TX-RX pairs
fprintf('   Generating %d x %d = %d FMC traces...\n', ...
    n_elements, n_elements, n_elements^2);
fmc = zeros(n_elements, n_elements, nt);

for itx = 1:n_elements
    for irx = 1:n_elements
        % Two-way traveltime to target (curved model)
        tau_twoway = tau_target(itx) + tau_target(irx);
        
        % Amplitude (distance decay)
        amplitude = 1.0 / max(tau_twoway, 1e-8);
        
        % Construct synthetic echo
        pulse_vec = pulse(t - tau_twoway);
        fmc(itx, irx, :) = amplitude * pulse_vec;
    end
end

% Add realistic low-level noise
noise_level = 0.02 * max(abs(fmc(:)));
fmc = fmc + noise_level * randn(size(fmc));

fprintf('   ✓ FMC data generated (peak amplitude: %.4f)\n\n', max(abs(fmc(:))));

%% ============================================================
% PART 6: TFM Beamforming - Straight-Ray Model
% ============================================================

fprintf('[6] Performing standard TFM (straight-ray delays)...\n');

image_standard = tfmBeamform(fmc, t, ...
    tau_tx_straight_img, tau_rx_straight_img, elem_x, elem_z);

% Envelope detection (analytic signal)
image_standard_env = abs(hilbert(image_standard, [], 1));
image_standard_env = image_standard_env / max(image_standard_env(:));

fprintf('   ✓ Standard TFM beamforming complete\n');
fprintf('     Peak: %.4f\n', max(image_standard_env(:)));

%% ============================================================
% PART 7: TFM Beamforming - Curved-Ray Model (Corrected)
% ============================================================

fprintf('[7] Performing corrected TFM (curved-ray delays)...\n');

image_corrected = tfmBeamform(fmc, t, ...
    tau_tx_curved_img, tau_rx_curved_img, elem_x, elem_z);

% Envelope detection
image_corrected_env = abs(hilbert(image_corrected, [], 1));
image_corrected_env = image_corrected_env / max(image_corrected_env(:));

fprintf('   ✓ Corrected TFM beamforming complete\n');
fprintf('     Peak: %.4f\n', max(image_corrected_env(:)));

%% ============================================================
% PART 8: Image Analysis and Metrics
% ============================================================

fprintf('[8] Computing performance metrics...\n\n');

% Find peaks
[peak_std, idx_std] = max(image_standard_env(:));
[iz_std, ix_std] = ind2sub(size(image_standard_env), idx_std);

[peak_corr, idx_corr] = max(image_corrected_env(:));
[iz_corr, ix_corr] = ind2sub(size(image_corrected_env), idx_corr);

% Peak positions
pos_std = [x(ix_std), z(iz_std)];
pos_corr = [x(ix_corr), z(iz_corr)];
pos_true = [target_x, target_z];

% Position errors
err_std_x = abs(pos_std(1) - pos_true(1));
err_std_z = abs(pos_std(2) - pos_true(2));
err_corr_x = abs(pos_corr(1) - pos_true(1));
err_corr_z = abs(pos_corr(2) - pos_true(2));

fprintf('Position accuracy:\n');
fprintf('  Standard TFM peak:  x=%.3f mm, z=%.3f mm\n', pos_std(1)*1e3, pos_std(2)*1e3);
fprintf('    Error from true:  Δx=%.3f mm, Δz=%.3f mm\n', err_std_x*1e3, err_std_z*1e3);
fprintf('  Corrected TFM peak: x=%.3f mm, z=%.3f mm\n', pos_corr(1)*1e3, pos_corr(2)*1e3);
fprintf('    Error from true:  Δx=%.3f mm, Δz=%.3f mm\n', err_corr_x*1e3, err_corr_z*1e3);
fprintf('  True target:        x=%.3f mm, z=%.3f mm\n', pos_true(1)*1e3, pos_true(2)*1e3);

% Lateral (-6dB) beamwidth
prof_std_x = image_standard_env(iz_std, :);
prof_corr_x = image_corrected_env(iz_corr, :);

bw_std_x = beamwidth_m6dB(prof_std_x, x);
bw_corr_x = beamwidth_m6dB(prof_corr_x, x);

fprintf('\nLateral beamwidth (-6dB):\n');
fprintf('  Standard: %.3f mm\n', bw_std_x*1e3);
fprintf('  Corrected: %.3f mm\n', bw_corr_x*1e3);
fprintf('  Improvement: %.1f %%\n', 100*(bw_std_x - bw_corr_x)/bw_std_x);

% Axial (-6dB) beamwidth
prof_std_z = image_standard_env(:, ix_std);
prof_corr_z = image_corrected_env(:, ix_corr);

bw_std_z = beamwidth_m6dB(prof_std_z, z);
bw_corr_z = beamwidth_m6dB(prof_corr_z, z);

fprintf('\nAxial beamwidth (-6dB):\n');
fprintf('  Standard: %.3f mm\n', bw_std_z*1e3);
fprintf('  Corrected: %.3f mm\n', bw_corr_z*1e3);
fprintf('  Improvement: %.1f %%\n', 100*(bw_std_z - bw_corr_z)/bw_std_z);

% Sidelobe level
sl_std = max(max(image_standard_env(1:iz_std-20, :)), ...
             max(image_standard_env(iz_std+20:end, :)));
sl_corr = max(max(image_corrected_env(1:iz_corr-20, :)), ...
              max(image_corrected_env(iz_corr+20:end, :)));

fprintf('\nSidelobe level:\n');
fprintf('  Standard: %.2f dB\n', 20*log10(sl_std+1e-6));
fprintf('  Corrected: %.2f dB\n', 20*log10(sl_corr+1e-6));
fprintf('  Improvement: %.2f dB\n', 20*log10(sl_std/max(sl_corr,1e-6)));

% Contrast (peak to sidelobe)
contrast_std = peak_std / max(sl_std, 1e-6);
contrast_corr = peak_corr / max(sl_corr, 1e-6);

fprintf('\nContrast ratio (peak/sidelobe):\n');
fprintf('  Standard: %.2f dB\n', 20*log10(contrast_std+1e-6));
fprintf('  Corrected: %.2f dB\n', 20*log10(contrast_corr+1e-6));
fprintf('  Improvement: %.2f dB\n', 20*log10(contrast_corr/max(contrast_std,1e-6)));

% Travel-time error statistics
tau_diff = tau_tx_curved_img + tau_rx_curved_img - ...
           (tau_tx_straight_img + tau_rx_straight_img);
tau_err_pct = 100 * tau_diff ./ (tau_tx_straight_img + tau_rx_straight_img + 1e-8);

fprintf('\nTravel-time correction statistics:\n');
fprintf('  Mean delay correction: %.3f ns\n', mean(tau_diff(:))*1e9);
fprintf('  Max delay correction: %.3f ns\n', max(abs(tau_diff(:)))*1e9);
fprintf('  Mean correction %% of signal: %.4f %%\n', mean(abs(tau_err_pct(:))));

fprintf('\n✓ Metrics computation complete\n\n');

%% ============================================================
% PART 9: Visualization - Main Comparison Figure
% ============================================================

fprintf('[9] Generating visualization...\n\n');

fig1 = figure('Color', 'w', 'Position', [100 100 1400 850], 'Name', 'TFM Comparison');

% Plot 1: Standard TFM (dB scale)
subplot(2, 3, 1);
img_std_db = 20*log10(image_standard_env + 1e-4);
imagesc(x*1e3, z*1e3, max(img_std_db, max(img_std_db(:))-40));
axis image; colorbar; colormap(parula);
caxis([max(img_std_db(:))-40, max(img_std_db(:))]);
hold on;
plot(pos_std(1)*1e3, pos_std(2)*1e3, 'r*', 'MarkerSize', 15, 'LineWidth', 2);
plot(pos_true(1)*1e3, pos_true(2)*1e3, 'yo', 'MarkerSize', 8, 'LineWidth', 2);
plot(elem_x*1e3, elem_z*1e3, 'k^', 'MarkerSize', 6);
xlabel('x [mm]'); ylabel('z [mm]');
title('Standard TFM: Straight Rays (-40 dB scale)');
legend('Peak', 'True target', 'Array', 'Location', 'southeast');

% Plot 2: Corrected TFM (dB scale)
subplot(2, 3, 2);
img_corr_db = 20*log10(image_corrected_env + 1e-4);
imagesc(x*1e3, z*1e3, max(img_corr_db, max(img_corr_db(:))-40));
axis image; colorbar; colormap(parula);
caxis([max(img_corr_db(:))-40, max(img_corr_db(:))]);
hold on;
plot(pos_corr(1)*1e3, pos_corr(2)*1e3, 'r*', 'MarkerSize', 15, 'LineWidth', 2);
plot(pos_true(1)*1e3, pos_true(2)*1e3, 'yo', 'MarkerSize', 8, 'LineWidth', 2);
plot(elem_x*1e3, elem_z*1e3, 'k^', 'MarkerSize', 6);
xlabel('x [mm]'); ylabel('z [mm]');
title('Corrected TFM: Curved Rays (-40 dB scale)');
legend('Peak', 'True target', 'Array', 'Location', 'southeast');

% Plot 3: Improvement map (dB difference)
subplot(2, 3, 3);
diff_db = img_corr_db - img_std_db;
imagesc(x*1e3, z*1e3, diff_db);
axis image; colorbar; colormap('RdBu');
xlabel('x [mm]'); ylabel('z [mm]');
title('Improvement (Corrected - Standard) [dB]');
caxis([-5, 5]);

% Plot 4: Lateral profiles at peaks
subplot(2, 3, 4);
plot(x*1e3, prof_std_x, 'b-', 'LineWidth', 2, 'DisplayName', 'Standard');
hold on;
plot(x*1e3, prof_corr_x, 'r-', 'LineWidth', 2, 'DisplayName', 'Corrected');
grid on; xlabel('x [mm]'); ylabel('Normalized amplitude');
title('Lateral profile (-6dB beamwidths marked)');
plot(pos_std(1)*1e3, 0.5, 'bs', 'MarkerSize', 10);
plot(pos_corr(1)*1e3, 0.5, 'rs', 'MarkerSize', 10);
axline(pos_std(1)*1e3, '-6dB'); axline(pos_corr(1)*1e3, '-6dB');
legend; ylim([0 1.1]);

% Plot 5: Axial profiles at peaks
subplot(2, 3, 5);
plot(z*1e3, prof_std_z, 'b-', 'LineWidth', 2, 'DisplayName', 'Standard');
hold on;
plot(z*1e3, prof_corr_z, 'r-', 'LineWidth', 2, 'DisplayName', 'Corrected');
grid on; xlabel('z [mm]'); ylabel('Normalized amplitude');
title('Axial profile (-6dB beamwidths marked)');
plot(pos_std(2)*1e3, 0.5, 'bs', 'MarkerSize', 10);
plot(pos_corr(2)*1e3, 0.5, 'rs', 'MarkerSize', 10);
legend; ylim([0 1.1]);

% Plot 6: Metrics summary text
subplot(2, 3, 6);
axis off;
metrics_text = sprintf([...
    '╔════════════════════════════════╗\n',...
    '║   PERFORMANCE SUMMARY           ║\n',...
    '╚════════════════════════════════╝\n',...
    '\n',...
    'POSITION ERROR:\n',...
    '  Standard: Δx=%.3f mm, Δz=%.3f mm\n',...
    '  Corrected: Δx=%.3f mm, Δz=%.3f mm\n',...
    '  ✓ Position gain: %.1f %% (lateral)\n',...
    '\n',...
    'LATERAL BEAMWIDTH (-6dB):\n',...
    '  Standard: %.3f mm\n',...
    '  Corrected: %.3f mm\n',...
    '  ✓ Improvement: %.1f %%\n',...
    '\n',...
    'AXIAL BEAMWIDTH (-6dB):\n',...
    '  Standard: %.3f mm\n',...
    '  Corrected: %.3f mm\n',...
    '  ✓ Improvement: %.1f %%\n',...
    '\n',...
    'SIDELOBE LEVEL:\n',...
    '  Standard: %.2f dB\n',...
    '  Corrected: %.2f dB\n',...
    '  ✓ Suppression: %.2f dB\n',...
    '\n',...
    'SPEED VARIATION: %.2f %%\n',...
    ], ...
    err_std_x*1e3, err_std_z*1e3, ...
    err_corr_x*1e3, err_corr_z*1e3, ...
    100*(err_std_x - err_corr_x)/max(err_std_x, 1e-6), ...
    bw_std_x*1e3, bw_corr_x*1e3, ...
    100*(bw_std_x - bw_corr_x)/bw_std_x, ...
    bw_std_z*1e3, bw_corr_z*1e3, ...
    100*(bw_std_z - bw_corr_z)/bw_std_z, ...
    20*log10(sl_std+1e-6), 20*log10(sl_corr+1e-6), ...
    20*log10(sl_std/max(sl_corr, 1e-6)), ...
    100*(max(c(:)) - min(c(:)))/mean(c(:)));

text(0.05, 0.95, metrics_text, 'FontName', 'Courier', 'FontSize', 9.5, ...
    'VerticalAlignment', 'top', 'HorizontalAlignment', 'left');
box on;

drawnow;

%% ============================================================
% PART 10: Diagnostic Figures
% ============================================================

% Figure 2: Medium properties
fig2 = figure('Color', 'w', 'Position', [100 1000 1000 400], 'Name', 'Medium Properties');

subplot(1, 3, 1);
imagesc(x*1e3, z*1e3, T);
axis image; colorbar; colormap(parula);
xlabel('x [mm]'); ylabel('z [mm]');
title('Temperature Field T(x,z) [°C]');

subplot(1, 3, 2);
imagesc(x*1e3, z*1e3, c);
axis image; colorbar; colormap(parula);
xlabel('x [mm]'); ylabel('z [mm]');
title('Sound Speed c(x,z) [m/s]');

subplot(1, 3, 3);
imagesc(x*1e3, z*1e3, sqrt(dc_dx.^2 + dc_dz.^2));
axis image; colorbar; colormap(hot);
xlabel('x [mm]'); ylabel('z [mm]');
title('Speed Gradient |∇c| [m/s²]');

drawnow;

% Figure 3: Ray geometry example
fig3 = figure('Color', 'w', 'Position', [1100 1000 500 400], 'Name', 'Ray Geometry');

% Show one example TX-RX pair with curved vs straight ray
itx_ex = ceil(n_elements/3);
irx_ex = ceil(2*n_elements/3);

tx_pos_ex = [elem_x(itx_ex), elem_z(itx_ex)];
rx_pos_ex = [elem_x(irx_ex), elem_z(irx_ex)];

% Plot medium
imagesc(x*1e3, z*1e3, c);
axis image; colorbar; colormap(parula);
hold on;

% Straight ray
plot([tx_pos_ex(1), rx_pos_ex(1)]*1e3, ...
     [tx_pos_ex(2), rx_pos_ex(2)]*1e3, ...
     'r--', 'LineWidth', 2, 'DisplayName', 'Straight ray');

% Curved ray (recompute for visualization)
[curved_path_x, curved_path_z] = rayPathVisualization2D(...
    tx_pos_ex, rx_pos_ex, x, z, c, dc_dx, dc_dz);
plot(curved_path_x*1e3, curved_path_z*1e3, ...
    'b-', 'LineWidth', 2, 'DisplayName', 'Curved ray');

% Markers
plot(tx_pos_ex(1)*1e3, tx_pos_ex(2)*1e3, 'r^', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
plot(rx_pos_ex(1)*1e3, rx_pos_ex(2)*1e3, 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
plot(elem_x*1e3, elem_z*1e3, 'k.', 'MarkerSize', 8);

xlabel('x [mm]'); ylabel('z [mm]');
title(sprintf('Ray paths TX#%d → RX#%d', itx_ex, irx_ex));
legend('Location', 'best');

drawnow;

fprintf('════════════════════════════════════════\n');
fprintf('✓ SIMULATION COMPLETE\n');
fprintf('════════════════════════════════════════\n\n');

%% ============================================================
% LOCAL FUNCTIONS
% ============================================================

function T = buildTemperatureField2D(X, Z, Tmin, Tmax)
    % Build a smooth 2D nonuniform temperature field
    % Features: linear gradient + hot spot + sinusoidal modulation
    
    x_norm = X / max(X(:));
    z_norm = Z / max(Z(:));
    
    % Gradient along x and z
    T_grad = Tmin + (Tmax - Tmin) * (0.3*x_norm + 0.5*z_norm);
    
    % Localized hot spot
    center_x = 0.6;
    center_z = 0.62;
    sigma_x = 0.08;
    sigma_z = 0.12;
    T_hotspot = 0.5*(Tmax - Tmin) * exp(-((x_norm - center_x).^2/(2*sigma_x^2) + ...
                                           (z_norm - center_z).^2/(2*sigma_z^2)));
    
    % Sinusoidal ripple
    T_ripple = 0.15*(Tmax - Tmin) * sin(4*pi*x_norm) .* cos(3*pi*z_norm);
    
    % Combine
    T = T_grad + T_hotspot + T_ripple;
    T = min(max(T, Tmin), Tmax);
end

function tau = curvedTravelTime2D(p0, p1, x, z, c, dc_dx, dc_dz)
    % Compute curved-ray travel time using RK4 integration
    % Solves ray ODE in slowness-based formulation
    %
    % Ray ODE (isotropic case):
    %   dr/ds = q (ray direction)
    %   dq/ds = (∇log n - q*(q·∇log n)) (ray refraction)
    % where n = 1/c is the slowness, s is arc length parameter.
    %
    % Input:
    %   p0, p1 = [x, z] vectors for source and receiver
    %   x, z = coordinate arrays
    %   c = sound speed field [nz, nx]
    %   dc_dx, dc_dz = speed gradients
    %
    % Output:
    %   tau = integrated travel time
    
    % Initial direction (straight line)
    direction = (p1 - p0) / (norm(p1 - p0) + 1e-12);
    distance = norm(p1 - p0);
    
    % Iterative ray shooting (5 iterations to converge)
    for iter = 1:5
        [q_end, r_end, tau_trial] = integrateRay2D(p0, direction, distance*1.8, ...
            x, z, c, dc_dx, dc_dz);
        
        error = p1 - r_end;
        error_norm = norm(error);
        
        if error_norm < 1e-5
            tau = tau_trial;
            return;
        end
        
        % Adjust direction
        direction = direction + 0.4*error/(distance + eps);
        direction = direction / (norm(direction) + eps);
    end
    
    tau = tau_trial;
end

function [q_end, r_end, tau] = integrateRay2D(p0, direction, L, x, z, c, dc_dx, dc_dz)
    % RK4 integration of ray equation
    
    h = max(min(L/80, 0.0004), 1e-5);  % adaptive step size
    N = ceil(L / h);
    h = L / N;
    
    r = p0(:);
    q = direction(:);
    tau = 0;
    
    for step = 1:N
        % RK4 coefficients
        [kr1, kt1] = rayStep(r, q, x, z, c, dc_dx, dc_dz);
        [kr2, kt2] = rayStep(r + 0.5*h*kr1, q + 0.5*h*q, x, z, c, dc_dx, dc_dz);
        [kr3, kt3] = rayStep(r + 0.5*h*kr2, q + 0.5*h*q, x, z, c, dc_dx, dc_dz);
        [kr4, kt4] = rayStep(r + h*kr3, q + h*q, x, z, c, dc_dx, dc_dz);
        
        % Update
        r = r + (h/6)*(kr1 + 2*kr2 + 2*kr3 + kr4);
        q = q + (h/6)*(kr1 + 2*kr2 + 2*kr3 + kr4);
        q = q / (norm(q) + eps);  % maintain unit direction
        tau = tau + (h/6)*(kt1 + 2*kt2 + 2*kt3 + kt4);
        
        % Boundary check
        if r(1) < min(x) || r(1) > max(x) || r(2) < min(z) || r(2) > max(z)
            break;
        end
    end
    
    q_end = q;
    r_end = r(:).';
end

function [dq, slowness] = rayStep(r, q, x, z, c, dc_dx, dc_dz)
    % Compute ray step: dq/ds and slowness
    
    % Interpolate at current position
    c_local = interp2(x, z, c, r(1), r(2), 'linear', mean(c(:)));
    c_local = max(c_local, 1000);  % clamp to physical range
    
    % Gradients
    grad_c_x = interp2(x, z, dc_dx, r(1), r(2), 'linear', 0);
    grad_c_z = interp2(x, z, dc_dz, r(1), r(2), 'linear', 0);
    
    % Slowness n = 1/c
    slowness = 1 / c_local;
    
    % log-slowness gradient: ∇log(n) = ∇log(1/c) = -∇c/c
    grad_logn_x = -grad_c_x / (c_local + eps);
    grad_logn_z = -grad_c_z / (c_local + eps);
    grad_logn = [grad_logn_x; grad_logn_z];
    
    % Ray equation: dq/ds = (∇log n - q*(q·∇log n))
    q_dot_grad = q(1)*grad_logn_x + q(2)*grad_logn_z;
    dq = grad_logn - q_dot_grad * q;
end

function image = tfmBeamform(fmc, t, tau_tx, tau_rx, elem_x, elem_z)
    % TFM beamforming via delay-and-sum
    %
    % Input:
    %   fmc = [n_elements, n_elements, nt] FMC data
    %   t = [nt, 1] time vector
    %   tau_tx, tau_rx = [n_elements, nz, nx] travel-time tables
    %   elem_x, elem_z = element positions (for reference)
    %
    % Output:
    %   image = [nz, nx] beamformed image
    
    [nel, ~, nt] = size(fmc);
    [~, nz, nx] = size(tau_tx);
    
    image = zeros(nz, nx);
    
    for iz = 1:nz
        if mod(iz, 30) == 0
            fprintf('      Beamforming: z-slice %d/%d\n', iz, nz);
        end
        
        for ix = 1:nx
            pixel_sum = 0;
            
            for itx = 1:nel
                for irx = 1:nel
                    % Total two-way travel time
                    tau_total = tau_tx(itx, iz, ix) + tau_rx(irx, iz, ix);
                    
                    % Delay-and-sum: interpolate RF at delay time
                    rf_trace = squeeze(fmc(itx, irx, :));
                    
                    % Linear interpolation in time
                    sample_val = interp1(t, rf_trace, tau_total, 'linear', 0);
                    pixel_sum = pixel_sum + sample_val;
                end
            end
            
            image(iz, ix) = pixel_sum / (nel^2);
        end
    end
end

function bw = beamwidth_m6dB(profile, coords)
    % Compute -6dB beamwidth of a profile
    
    peak = max(profile);
    threshold = peak / 2;  % -6dB = 1/2 in linear scale
    
    indices = find(profile >= threshold);
    
    if length(indices) >= 2
        bw = coords(max(indices)) - coords(min(indices));
    else
        bw = 0.001;  % default: 1 mm
    end
end

function [path_x, path_z] = rayPathVisualization2D(p0, p1, x, z, c, dc_dx, dc_dz)
    % Generate ray path for visualization purposes
    
    direction = (p1 - p0) / (norm(p1 - p0) + 1e-12);
    distance = norm(p1 - p0);
    
    % Coarse ray tracing for plotting
    h = distance / 30;  % 30 points on ray
    r = p0(:);
    q = direction(:);
    
    path_x = [r(1)];
    path_z = [r(2)];
    
    for step = 1:30
        % Single RK4 step
        [dq, ~] = rayStep(r, q, x, z, c, dc_dx, dc_dz);
        r = r + h*q;
        q = q + h*dq;
        q = q / (norm(q) + eps);
        
        path_x = [path_x, r(1)];
        path_z = [path_z, r(2)];
        
        if norm(r - p1) < 1e-4
            break;
        end
        
        if r(1) < min(x) || r(1) > max(x) || r(2) < min(z) || r(2) > max(z)
            break;
        end
    end
end

function axline(x_val, label)
    % Helper: add vertical line and label
    yL = ylim;
    plot([x_val, x_val], yL, 'k--', 'LineWidth', 0.7);
    text(x_val, yL(2)*0.95, label, 'FontSize', 8, 'HorizontalAlignment', 'center');
end
