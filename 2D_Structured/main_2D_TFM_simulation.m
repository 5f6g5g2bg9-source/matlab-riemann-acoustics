%% ============================================================
% MAIN: 2D CURVED-RAY TFM SIMULATION
% Material: Nickel-Based Single-Crystal Superalloy
% ============================================================
% 目标：
%   实现完整的二维弯曲走时 TFM 波束合成仿真程序
%   用于验证温度梯度引起的声速变化对 TFM 成像的影响
%
% 结构：
%   通过模块化设计，将整个流程分解为：
%   初始化 → 物理场构造 → 走时计算 → 数据生成 → 成像 → 评估 → 可视化
%
% 运行时间：~3-5 分钟
% 依赖：无需外部工具箱
% ============================================================

clear; close all; clc;
rng(42);

fprintf('\n');
fprintf('═════════════════════════════════════════════════\n');
fprintf('2D CURVED-RAY TFM SIMULATION (NICKEL SUPERALLOY)\n');
fprintf('═════════════���═══════════════════════════════════\n\n');

%% ============================================================
% STEP 1: 初始化仿真参数
% ============================================================
% 这一步加载所有与仿真有关的参数：
%   - 计算域大小和网格分辨率
%   - 材料声学参数
%   - 超声阵列配置
%   - 信号处理参数
%
% 优点：所有参数集中在一个地方，便于修改和追溯

fprintf('[1] Initializing simulation parameters...\n');

domain = init_domain_2D();
material = init_material_params_nickel_alloy();
array = init_array_config();
signal = init_signal_params();

fprintf('    ✓ Domain: [%.1f mm x %.1f mm], grid: [%d x %d]\n', ...
    domain.Lx*1e3, domain.Lz*1e3, domain.nx, domain.nz);
fprintf('    ✓ Material: %s\n', material.name);
fprintf('    ✓ Array: %d elements, pitch = %.2f mm\n', array.n_elements, array.pitch*1e3);
fprintf('    ✓ Center frequency: %.2f MHz\n\n', signal.f0/1e6);

%% ============================================================
% STEP 2: 构造物理场（温度 → 声速）
% ============================================================
% 这一步是整个仿真的物理基础：
%   1) 根据镍基单晶合金在高温环境中的温度分布，构造非均匀温度场 T(x,z)
%   2) 利用温度-声速关系式（基于材料热学性质），计算声速场 c(x,z)
%   3) 计算速度梯度 ∇c，用于射线折射计算
%
% 物理背景：
%   镍基单晶高温合金在 20°C ~ 2500°C 温度范围内，声速会发生显著变化
%   这种变化导致声波传播路径弯曲，从而造成 TFM 成像偏差

fprintf('[2] Building physical fields (temperature -> sound speed)...\n');

[X, Z] = meshgrid(domain.x, domain.z);
T = buildTemperatureField2D(X, Z, domain.Lx, domain.Lz, material);
c = computeSoundSpeed2D(T, material);
[dc_dx, dc_dz] = computeGradients2D(c, domain.dx, domain.dz);

fprintf('    ✓ Temperature field: T ∈ [%.1f, %.1f] °C\n', min(T(:)), max(T(:)));
fprintf('    ✓ Sound speed field: c ∈ [%.0f, %.0f] m/s\n', min(c(:)), max(c(:)));
fprintf('    ✓ Speed variation: %.2f %%\n\n', ...
    100*(max(c(:)) - min(c(:))) / mean(c(:)));

%% ============================================================
% STEP 3: 计算走时表（标准 vs 弯曲）
% ============================================================
% 这是整个联合仿真的核心步骤：
%   - 标准走时：直线传播，使用均匀声速（会导致成像散焦）
%   - 弯曲走时：考虑速度梯度，通过射线追踪求解（更准确）
%
% 计算策略：
%   由于每个阵元到每个像素点的走时计算量较大（nel × nz × nx），
%   先在粗网格上计算，再插值到完整网格以加快速度

fprintf('[3] Computing travel-time tables...\n');
fprintf('    (This step may take 1-2 minutes)\n');

[tau_tx_straight_img, tau_rx_straight_img] = ...
    compute_traveltime_straight_2D(domain, array, c);

[tau_tx_curved_img, tau_rx_curved_img] = ...
    compute_traveltime_curved_2D(domain, array, c, dc_dx, dc_dz);

fprintf('    ✓ Travel-time tables computed\n');
fprintf('    ✓ Straight model: τ ∈ [%.3f, %.3f] μs\n', ...
    min(tau_tx_straight_img(:))*1e6, max(tau_tx_straight_img(:))*1e6);
fprintf('    ✓ Curved model: τ ∈ [%.3f, %.3f] μs\n\n', ...
    min(tau_tx_curved_img(:))*1e6, max(tau_tx_curved_img(:))*1e6);

%% ============================================================
% STEP 4: 生成合成 FMC 数据
% ============================================================
% FMC (Full Matrix Capture) 是全矩阵采集数据
%   - 每个发射阵元 TX 依次发射，所有接收阵元 RX 同时接收
%   - 得到一个 nel × nel × nt 的三维矩阵
%
% 生成策略：
%   使用弯曲走时模型生成数据，这样数据本身就包含了"非均匀介质中的
%   真实传播"，然后用标准/修正两种延时模型去成像，才能真实验证
%   弯曲走时修正的效果

fprintf('[4] Generating synthetic FMC data...\n');

target = [domain.Lx*0.58, domain.Lz*0.625];  % 点目标位置
fmc = generate_fmc_data(domain, array, signal, target, tau_tx_curved_img, tau_rx_curved_img);

fprintf('    ✓ FMC data generated\n');
fprintf('    ✓ Size: [%d TX × %d RX × %d samples]\n', ...
    array.n_elements, array.n_elements, length(signal.t));
fprintf('    ✓ Peak amplitude: %.4f\n\n', max(abs(fmc(:))));

%% ============================================================
% STEP 5: TFM 波束合成（标准与修正）
% ============================================================
% TFM 波束合成原理：
%   对于成像点 (x, z)，所有 Tx-Rx 对的回波按延时 τ_tx + τ_rx 对齐，
%   然后做相位求和（在窄带模型中简化为幅度求和）
%
% 两种对比：
%   1) 标准 TFM：使用直线走时（错误假设）→ 散焦
%   2) 修正 TFM：使用弯曲走时（正确模型）→ 聚焦
%
% 这正是我们要验证的核心问题

fprintf('[5] Performing TFM beamforming...\n');

fprintf('    Standard TFM (straight-ray delays)...\n');
image_standard = tfmBeamform_2D(fmc, signal.t, tau_tx_straight_img, tau_rx_straight_img);
image_standard_env = abs(hilbert(image_standard, [], 1)) / max(abs(hilbert(image_standard, [], 1)), [], 'all');

fprintf('    Corrected TFM (curved-ray delays)...\n');
image_corrected = tfmBeamform_2D(fmc, signal.t, tau_tx_curved_img, tau_rx_curved_img);
image_corrected_env = abs(hilbert(image_corrected, [], 1)) / max(abs(hilbert(image_corrected, [], 1)), [], 'all');

fprintf('    ✓ Beamforming complete\n');
fprintf('    ✓ Standard peak: %.4f\n', max(image_standard_env(:)));
fprintf('    ✓ Corrected peak: %.4f\n\n', max(image_corrected_env(:)));

%% ============================================================
% STEP 6: 性能评估
% ============================================================
% 通过量化指标来评估弯曲走时修正对成像的改善：
%   1) 峰值位置精度：目标位置偏移
%   2) 波束宽度：-6 dB 横向/纵向宽度
%   3) 旁瓣电平：背景噪声压制
%   4) 对比度：峰值 / 旁瓣比
%   5) 走时误差：弯曲 vs 直线走时的差异量化

fprintf('[6] Computing performance metrics...\n');

metrics = compute_metrics_2D(image_standard_env, image_corrected_env, ...
    domain, target, tau_tx_straight_img, tau_rx_straight_img, ...
    tau_tx_curved_img, tau_rx_curved_img);

fprintf('    ✓ Metrics computed\n');
fprintf('\n    === POSITION ACCURACY ===\n');
fprintf('    Standard: Δx=%.3f mm, Δz=%.3f mm\n', ...
    metrics.pos_err_std_x*1e3, metrics.pos_err_std_z*1e3);
fprintf('    Corrected: Δx=%.3f mm, Δz=%.3f mm\n', ...
    metrics.pos_err_corr_x*1e3, metrics.pos_err_corr_z*1e3);
fprintf('    Position gain: %.1f %% (lateral)\n\n', metrics.pos_gain_pct);

fprintf('    === LATERAL BEAMWIDTH (-6dB) ===\n');
fprintf('    Standard: %.3f mm\n', metrics.bw_std_x*1e3);
fprintf('    Corrected: %.3f mm\n', metrics.bw_corr_x*1e3);
fprintf('    Improvement: %.1f %%\n\n', metrics.bw_x_improve_pct);

fprintf('    === AXIAL BEAMWIDTH (-6dB) ===\n');
fprintf('    Standard: %.3f mm\n', metrics.bw_std_z*1e3);
fprintf('    Corrected: %.3f mm\n', metrics.bw_corr_z*1e3);
fprintf('    Improvement: %.1f %%\n\n', metrics.bw_z_improve_pct);

fprintf('    === SIDELOBE LEVEL ===\n');
fprintf('    Standard: %.2f dB\n', 20*log10(metrics.sl_std + 1e-6));
fprintf('    Corrected: %.2f dB\n', 20*log10(metrics.sl_corr + 1e-6));
fprintf('    Suppression: %.2f dB\n\n', 20*log10(metrics.sl_std / max(metrics.sl_corr, 1e-6)));

fprintf('    === TRAVEL-TIME CORRECTION ===\n');
fprintf('    Mean correction: %.3f ns\n', metrics.tau_mean_diff*1e9);
fprintf('    Max correction: %.3f ns\n\n', metrics.tau_max_diff*1e9);

%% ============================================================
% STEP 7: 可视化结果
% ============================================================
% 生成四个诊断图表：
%   Fig 1: TFM 成像对比（6 子图）
%   Fig 2: 介质属性（温度/声速/梯度）
%   Fig 3: 射线几何示意
%   Fig 4: 处理流程概览

fprintf('[7] Generating visualization...\n');

plot_tfm_comparison(domain, array, image_standard_env, image_corrected_env, ...
    metrics, target);

plot_medium_properties(domain, T, c, dc_dx, dc_dz);

plot_ray_geometry(domain, array, c, dc_dx, dc_dz);

fprintf('\n    ✓ Visualization complete\n\n');

%% ============================================================
% COMPLETION
% ============================================================

fprintf('═════════════════════════════════════════════════\n');
fprintf('✓ SIMULATION COMPLETE\n');
fprintf('═════════════════════════════════════════════════\n\n');

fprintf('Summary:\n');
fprintf('  - Nickel-based single-crystal superalloy modeled\n');
fprintf('  - Temperature range: %.0f - %.0f °C\n', material.Tmin, material.Tmax);
fprintf('  - Speed variation: %.2f %%\n', 100*(max(c(:))-min(c(:)))/mean(c(:)));
fprintf('  - Curved-ray correction demonstrates %.1f %% beamwidth improvement\n', ...
    metrics.bw_x_improve_pct);
fprintf('  - Peak position accuracy improved by %.1f %%\n\n', metrics.pos_gain_pct);
