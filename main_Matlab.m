%% Wind Data Analysis Script
% This script processes wind (and associated sea) data for a DTU 10 MW wind turbine.
% It performs the following tasks:
%   1. Define the hub height.
%   2. Interpolate wind speed and direction data at the hub height using PCHIP (and linear for comparison).
%   3. Visualize the interpolation results.
%   4. Plot wind speed and direction histograms, fit a two-parameter Weibull distribution,
%      and generate a wind rose.
%   5. Fit Weibull distributions to wind speed data in four directional sectors.
%   6. Compute monthly and yearly wind speed statistics.
%   7. Plot monthly histograms and fit Weibull PDFs.
%   8. Investigate selected wind situations by estimating the shear exponent and wind veer.
%   9. Examine the air–sea temperature difference for the same cases.
%
% Author: [Your Name]
% Date: [Current Date]

clearvars; close all; clc;
addpath('./functions');  % Add folder with additional functions

%% Question 1: Define Hub Height
% The DTU 10 MW wind turbine has a hub height of 119 m.
zHub = 119;

%% Question 2: Interpolate Wind Speed and Direction at Hub Height
% Read the wind and sea data table. The file has 3 header lines.
T = readtable('NORA10_1957_2018_wind_and_sea.txt', "NumHeaderLines", 3);

% Create a datetime vector using year, month, day and hour columns.
time = datetime(T.YEAR, T.M, T.D, T.H, zeros(size(T.H)), zeros(size(T.H)));

% Define measured wind speeds at different heights (m/s) and their corresponding heights.
oldU = [T.W10, T.W50, T.W80, T.W100, T.W150];
oldZ_U = [10 50 80 100 150];

% Define measured wind directions (deg) and their corresponding heights.
oldDir = [T.D10, T.D100, T.D150];
oldZ_Dir = [10 100 150];

% New vertical grid including the hub height.
newZ = sort([10 50 80 100 zHub 150]);
[~, indZ] = min(abs(newZ - zHub));  % index corresponding to hub height

% Create a time axis in hours (dt = 3 h)
N = size(oldU, 1);
dt = 3;
t = (0:N-1) * dt;

% Preallocate arrays for interpolated wind speeds using PCHIP and linear interpolation.
newU = zeros(numel(newZ), N);
newU_linear = zeros(numel(newZ), N);

% Loop over each time step and interpolate wind speeds at new heights.
tic
for ii = 1:N
    newU(:, ii) = interp1(oldZ_U, oldU(ii, :), newZ, 'pchip');
    newU_linear(:, ii) = interp1(oldZ_U, oldU(ii, :), newZ, 'linear');
end
toc

% Interpolate the wind direction.
% First, convert directional data to vector components.
newDir = zeros(numel(newZ), N);
tic
for ii = 1:N
    oldVx = cosd(oldDir(ii, :));
    oldVy = sind(oldDir(ii, :));
    
    % Interpolate the vector components at new heights.
    newVx = interp1(oldZ_Dir, oldVx, newZ, 'pchip');
    newVy = interp1(oldZ_Dir, oldVy, newZ, 'pchip');
    
    % Reconstruct wind direction from the interpolated vector components.
    newDir(:, ii) = atan2d(newVy, newVx);
end
toc
% Adjust negative angles to the range [0, 360)
newDir(newDir < 0) = newDir(newDir < 0) + 360;

%% Visualize the Interpolation Results
figure('position', [165 42 560 879]);
tiledlayout(3,1, 'TileSpacing', 'tight');

% Compare wind speeds at hub height with the levels immediately above and below.
nexttile
plot(newU(indZ, :), newU(indZ+1, :), 'r.')
hold on
plot(newU(indZ, :), newU(indZ-1, :), 'b.')
axis equal; axis tight; grid on; grid minor;
xlabel('Mean wind speed (m/s)');
ylabel('Mean wind speed (m/s)');
legend('119 m vs 150 m', '119 m vs 100 m');

% Compare linear vs. PCHIP interpolation at the hub height.
nexttile
plot(newU(indZ, :), newU_linear(indZ, :), '.')
axis equal; axis tight; grid on; grid minor;
xlabel('Linear interpolation @ 119 (m/s)');
ylabel('PCHIP interpolation @ 119 (m/s)');

% Compare interpolated wind directions at hub height and adjacent levels.
nexttile
plot(newDir(indZ, :), newDir(indZ+1, :), 'r.')
hold on
plot(newDir(indZ, :), newDir(indZ-1, :), 'b.')
axis equal; axis tight; grid on; grid minor;
xlabel('Mean wind direction (deg)');
ylabel('Mean wind direction (deg)');
legend('119 m vs 150 m', '119 m vs 100 m');
set(gcf, 'color', 'w');

%% Question 3: Wind Speed and Direction Distribution & Wind Rose
% Replace zeros with NaN and fill gaps using inpaint_nans (using method 4).
newU(newU == 0) = nan;
newU = inpaint_nans(newU, 4);

% Plot wind speed histogram and fit a two-parameter Weibull distribution.
figure('position', [811 516 476 321]);
[h1, parmHat1, kernelData] = plotWindDistribution(newU(indZ, :)', 'r');
xlim([0 30]);
grid on;
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 14, 'FontName', 'Times');
% Optionally export the figure:
% exportgraphics(gcf, ['./figures/weibull_fit.pdf'], 'ContentType', 'vector');

% Create a wind rose using only wind speeds above 5 m/s.
indSpeed = find(newU(indZ, :) > 5);
f = wind_rose(90 - newDir(indZ, indSpeed), newU(indZ, indSpeed), 'nDirections', 36, ...
    'labels', {'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'}, 'cMap', 'viridis', ...
    'vWinds', 5:3:30);
set(f, 'position', [151 113 732 764]);

%% Question 4: Sector Analysis with Weibull and Kernel Estimates
% Define the four wind direction sectors.
sector = {'315-45 deg', '45-135 deg', '135-225 deg', '225-315 deg'};
ind{1} = find(newDir(indZ, :) >= 315 | newDir(indZ, :) < 45);
ind{2} = find(newDir(indZ, :) >= 45 & newDir(indZ, :) < 135);
ind{3} = find(newDir(indZ, :) >= 135 & newDir(indZ, :) < 225);
ind{4} = find(newDir(indZ, :) >= 225 & newDir(indZ, :) < 315);

% Plot kernel density estimates for each sector.
COLOR = distinguishable_colors(numel(ind));
figure('position', [407 240 568 637]);
tiledlayout(2,1, "TileSpacing", "tight");

nexttile
clear h1 leg
for ii = 1:numel(ind)
    U = newU(indZ, ind{ii});
    u = linspace(min(U), max(U), 100);
    pd1 = fitdist(U(:), 'kernel');
    Y1 = pdf(pd1, u);
    h1(ii) = plot(u, Y1, 'color', COLOR(ii, :), 'linewidth', 2);
    hold on;
    leg{ii} = ['Kernel estimate (', sector{ii}, ')'];
end
xlabel('$\overline{u}$ (m s$^{-1}$)', 'interpreter', 'latex');
ylabel('Probability density function');
legend(h1, leg{:});
set(gcf, 'color', 'w');
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 12, 'FontName', 'Times');

% Plot Weibull fits for each sector.
nexttile
clear h1
for ii = 1:numel(ind)
    U = newU(indZ, ind{ii});
    parmHat1 = wblfit(U(:));
    u_vals = 0:0.1:ceil(max(U));
    y = wblpdf(u_vals, parmHat1(1), parmHat1(2));
    h1(ii) = plot(u_vals, y, 'color', COLOR(ii, :), 'linewidth', 2);
    hold on;
    fitlabel = ['a = ', num2str(round(parmHat1(1)*10)/10), ', b = ', num2str(round(parmHat1(2)*10)/10)];
    leg{ii} = [fitlabel, '  (', sector{ii}, ')'];
end
xlabel('$\overline{u}$ (m s$^{-1}$)', 'interpreter', 'latex');
ylabel('Probability density function');
legend(h1, leg{:});
set(gcf, 'color', 'w');
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 12, 'FontName', 'Times');

%% Question 5: Monthly and Yearly Wind Speed Statistics
% Compute monthly mean, median, maximum, and standard deviation at nacelle level.
clear monthlyStat
for ii = 1:12
    indMonth = find(month(time) == ii);
    monthlyStat.meanU(ii)   = mean(newU(indZ, indMonth));
    monthlyStat.medianU(ii) = median(newU(indZ, indMonth));
    monthlyStat.maxU(ii)    = max(newU(indZ, indMonth));
    monthlyStat.stdU(ii)    = std(newU(indZ, indMonth));
    monthlyStat.x(ii)       = categorical(month(time(indMonth(1)), 'name'));
end

figure('position', [407 240 568 400]);
plot(monthlyStat.x, monthlyStat.meanU, 'ko--', 'markerfacecolor', 'r');
hold on;
plot(monthlyStat.x, monthlyStat.medianU, 'kd--', 'markerfacecolor', 'c', 'markersize', 5);
plot(monthlyStat.x, monthlyStat.maxU, 'ko--', 'markerfacecolor', 'g');
plot(monthlyStat.x, monthlyStat.stdU, 'ko--', 'markerfacecolor', [1 1 1]*0.5);
grid on;
ylim([0 35]);
set(gcf, 'color', 'w');
ylabel('mean wind statistics (m/s)');
legend('Mean', 'Median', 'Max', 'std', 'location', 'northoutside', 'orientation', 'horizontal');
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 12, 'FontName', 'Times');

% Yearly statistics (without sliding window)
clear yearlyStat
tyear = unique(year(time));
Nyear = numel(tyear);

for ii = 1:Nyear
    indYear = find(year(time) == tyear(ii));
    if ~isempty(indYear)
        yearlyStat.meanU(ii)   = mean(newU(indZ, indYear));
        yearlyStat.medianU(ii) = median(newU(indZ, indYear));
        yearlyStat.maxU(ii)    = max(newU(indZ, indYear));
        yearlyStat.stdU(ii)    = std(newU(indZ, indYear));
        yearlyStat.x(ii)       = tyear(ii);
    end
end

COLOR = distinguishable_colors(4);
figure('position', [407 240 568 400]);
plot(yearlyStat.x, yearlyStat.meanU, '.', 'color', COLOR(1, :), 'markersize', 10);
hold on;
plot(yearlyStat.x, yearlyStat.medianU, '.', 'color', COLOR(2, :), 'markersize', 10);
plot(yearlyStat.x, yearlyStat.maxU, '.', 'color', COLOR(3, :), 'markersize', 10);
plot(yearlyStat.x, yearlyStat.stdU, '.', 'color', COLOR(4, :), 'markersize', 10);
grid on;

% Fit and plot linear regression for each statistic.
fNames = fields(yearlyStat);
for ii = 1:4
    [p_fit, S, mu] = polyfit(tyear, yearlyStat.(fNames{ii}), 1);
    y_fit = polyval(p_fit, tyear, S, mu);
    plot(tyear, y_fit, 'color', COLOR(ii, :));
    label(['Linear regression (', fNames{ii}, ') ', num2str(round(p_fit(1)*10)/10), ...
        'x + (', num2str(round(10*p_fit(2))/10), ')'], 0.03, 0.3+0.06*ii);
end
ylim([0 35]);
set(gcf, 'color', 'w');
ylabel('mean wind statistics (m/s)');
legend('Mean', 'Median', 'Max', 'std', 'location', 'northoutside', 'orientation', 'horizontal');
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 12, 'FontName', 'Times');

%% Question 6: Monthly Histograms of Wind Speed at Nacelle Level
clf; close all;
COLOR = viridis(12);
figure('position', [402 46 723 868]);
tiledlayout(2,1, "TileSpacing", "tight");
nexttile
COLOR = distinguishable_colors(12);
clear leg
for ii = 1:12
    indMonth = find(month(time) == ii);
    U = newU(indZ, indMonth);
    u = linspace(min(U), max(U), 100);
    pd1 = fitdist(U(:), 'kernel');
    Y1 = pdf(pd1, u);
    plot(u, Y1, 'color', COLOR(ii, :), 'linewidth', 1.2);
    hold on;
    leg{ii} = char(month(time(indMonth(1)), 'name'));
end
legend(leg{:});
xlabel('$\overline{u}$ (m s$^{-1}$)', 'interpreter', 'latex');
ylabel('Probability density function');
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 12, 'FontName', 'Times');
set(gcf, 'color', 'w');

%% Question 7: Monthly Weibull Fits
nexttile
COLOR = distinguishable_colors(12);
clear leg
for ii = 1:12
    indMonth = find(month(time) == ii);
    U = newU(indZ, indMonth);
    u = linspace(min(U), max(U), 100);
    parmHat1 = wblfit(U(:));
    y = wblpdf(0:0.1:ceil(max(U)), parmHat1(1), parmHat1(2));
    plot(0:0.1:ceil(max(U)), y, 'color', COLOR(ii, :), 'linewidth', 1.2);
    hold on;
    leg{ii} = [char(month(time(indMonth(1)), 'name')), ' (a = ', ...
        num2str(round(parmHat1(1)*10)/10), ', b = ', num2str(round(parmHat1(2)*10)/10), ')'];
end
legend(leg{:});
xlabel('$\overline{u}$ (m s$^{-1}$)', 'interpreter', 'latex');
ylabel('Probability density function');
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 12, 'FontName', 'Times');
set(gcf, 'color', 'w');

%% Question 8: Shear Exponent and Wind Veer Analysis
% Choose 4 mid-day wind situations in January, April, July, and October.
targetDate = [datetime(2015, 01, 01, 12, 0, 0), datetime(2015, 04, 01, 12, 0, 0), ...
    datetime(2015, 07, 01, 12, 0, 0), datetime(2015, 10, 01, 12, 0, 0)];

COLOR = distinguishable_colors(4);
myFun = @(alpha, zr) zr.^alpha;  % Power law function: u_r = (z/zHub)^alpha
guess = 0.1;
z1 = linspace(min(newZ), max(newZ), 100);

% Set options for the least-squares curve fit.
options = optimoptions('lsqcurvefit', 'Display', 'off');
clf; close all;
clear p leg a
figure('position', [197 504 1000 426]);

for ii = 1:numel(targetDate)
    % Find index closest to target time.
    [~, ind] = min(abs(time - targetDate(ii)));
    
    % Normalize heights and wind speeds.
    zr = newZ ./ newZ(indZ);
    ur = newU(:, ind) ./ newU(indZ, ind);
    
    % Estimate the shear exponent using lsqcurvefit.
    a(ii) = lsqcurvefit(myFun, guess, zr(:), ur(:), -1, 1, options);
    
    % Plot measured wind profile.
    plot(newU(:, ind), newZ, '.', 'color', COLOR(ii, :), 'markersize', 25);
    hold on;
    % Plot fitted power law curve.
    p(ii) = plot(myFun(a(ii), z1 ./ newZ(indZ)) .* newU(indZ, ind), z1, ...
        'color', COLOR(ii, :), 'linewidth', 1.5);
    leg{ii} = [datestr(time(ind)), ' (a = ', num2str(round(a(ii)*1000)/1000), ')'];
end
ylim([0 160]);
legend(p, leg{:}, 'location', 'eastoutside');
xlabel('$\overline{u}$ (m s$^{-1}$)', 'interpreter', 'latex');
ylabel('Wind speed (m/s)');
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 14, 'FontName', 'Times');
grid on;
set(gcf, 'color', 'w');

%% Question 9: Air–Sea Temperature Difference Analysis
% Check the temperature difference between 100 m (air temperature) and sea surface.
myTable = readtable('NORA10_1957_2018_SST_RH_T.txt', "NumHeaderLines", 2);
dT = myTable.T100 - myTable.SST;  % Temperature difference (°C)
time2 = datetime(myTable.YEAR, myTable.M, myTable.D, myTable.H, 0, 0);

% Compare dT for each selected wind situation.
for ii = 1:numel(a)
    [~, ind] = min(abs(time2 - targetDate(ii)));
    if dT(ind) > 0
        fprintf([datestr(time2(ind)), ' --- a = %1.2f and dT = %1.2f (air warmer than sea, stable regime (?)\n'], ...
            [a(ii), dT(ind)]);
    else
        fprintf([datestr(time2(ind)), ' --- a = %1.2f and dT = %1.2f (air colder than sea, unstable regime (?)\n'], ...
            [a(ii), dT(ind)]);
    end
end
