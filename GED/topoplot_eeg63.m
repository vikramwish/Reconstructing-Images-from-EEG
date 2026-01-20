function topoplot_eeg63(values, ch_names)
% TOPOPLOT_EEG63 Create a topographic plot for 63 EEG channels
%   This function creates a topographic plot using your specific 63 channels
%   values: 63x1 vector of channel values
%   ch_names: 63x1 cell array or char array of channel names

% Convert char array to cell array if needed
if ischar(ch_names)
    ch_names_cell = cell(size(ch_names, 1), 1);
    for i = 1:size(ch_names, 1)
        ch_names_cell{i} = strtrim(ch_names(i,:));
    end
    ch_names = ch_names_cell;
else
    % If already cell array, just trim spaces
    for i = 1:length(ch_names)
        ch_names{i} = strtrim(ch_names{i});
    end
end

% Define channel coordinates (using standard 10-20 positions)
coords = containers.Map();
% Frontal
coords('Fp1') = [-0.25, 0.9]; coords('Fp2') = [0.25, 0.9];
coords('AF7') = [-0.4, 0.8]; coords('AF3') = [-0.2, 0.8]; 
coords('AFz') = [0, 0.8]; coords('AF4') = [0.2, 0.8]; coords('AF8') = [0.4, 0.8];
coords('F7') = [-0.6, 0.6]; coords('F5') = [-0.45, 0.65]; coords('F3') = [-0.3, 0.7]; 
coords('F1') = [-0.15, 0.7]; coords('Fz') = [0, 0.7]; coords('F2') = [0.15, 0.7]; 
coords('F4') = [0.3, 0.7]; coords('F6') = [0.45, 0.65]; coords('F8') = [0.6, 0.6];
coords('FT9') = [-0.8, 0.45]; coords('FT7') = [-0.7, 0.5]; coords('FC5') = [-0.5, 0.5]; 
coords('FC3') = [-0.3, 0.5]; coords('FC1') = [-0.15, 0.5]; coords('FCz') = [0, 0.5]; 
coords('FC2') = [0.15, 0.5]; coords('FC4') = [0.3, 0.5]; coords('FC6') = [0.5, 0.5]; 
coords('FT8') = [0.7, 0.5]; coords('FT10') = [0.8, 0.45];
% Central
coords('T7') = [-0.8, 0.2]; coords('C5') = [-0.6, 0.2]; coords('C3') = [-0.4, 0.2]; 
coords('C1') = [-0.2, 0.2]; coords('Cz') = [0, 0.2]; coords('C2') = [0.2, 0.2]; 
coords('C4') = [0.4, 0.2]; coords('C6') = [0.6, 0.2]; coords('T8') = [0.8, 0.2];
% Temporal-Parietal
coords('TP9') = [-0.8, -0.1]; coords('TP7') = [-0.7, -0.1]; coords('CP5') = [-0.5, -0.1]; 
coords('CP3') = [-0.3, -0.1]; coords('CP1') = [-0.15, -0.1]; coords('CPz') = [0, -0.1]; 
coords('CP2') = [0.15, -0.1]; coords('CP4') = [0.3, -0.1]; coords('CP6') = [0.5, -0.1]; 
coords('TP8') = [0.7, -0.1]; coords('TP10') = [0.8, -0.1];
% Parietal
coords('P7') = [-0.65, -0.35]; coords('P5') = [-0.5, -0.35]; coords('P3') = [-0.35, -0.4]; 
coords('P1') = [-0.15, -0.4]; coords('Pz') = [0, -0.4]; coords('P2') = [0.15, -0.4]; 
coords('P4') = [0.35, -0.4]; coords('P6') = [0.5, -0.35]; coords('P8') = [0.65, -0.35];
% Parietal-Occipital
coords('PO7') = [-0.4, -0.6]; coords('PO3') = [-0.2, -0.6]; coords('POz') = [0, -0.6]; 
coords('PO4') = [0.2, -0.6]; coords('PO8') = [0.4, -0.6];
% Occipital
coords('O1') = [-0.2, -0.8]; coords('Oz') = [0, -0.85]; coords('O2') = [0.2, -0.8];

% Create x, y coordinates arrays
x = zeros(length(ch_names), 1);
y = zeros(length(ch_names), 1);

% Match channels to coordinates
for i = 1:length(ch_names)
    if coords.isKey(ch_names{i})
        pos = coords(ch_names{i});
        x(i) = pos(1);
        y(i) = pos(2);
    else
        warning('Channel %s not found in predefined coordinates', ch_names{i});
        x(i) = 0;
        y(i) = 0;
    end
end

% Create interpolation grid
gridRes = 100;
[xi, yi] = meshgrid(linspace(-1, 1, gridRes), linspace(-1, 1, gridRes));

% Interpolate values onto the grid
zi = griddata(x, y, values, xi, yi, 'v4');

% Create circular mask
mask = (xi.^2 + yi.^2) <= 0.95^2;
zi(~mask) = NaN;

% Create the plot
%figure;
contourf(xi, yi, zi, 40, 'LineStyle', 'none');
hold on;
contour(xi, yi, zi, 6, 'LineColor', 'k', 'LineWidth', 0.5);

% Plot electrodes
%scatter(x, y, 30, 'k', 'filled');

% Add channel labels
%for i = 1:length(ch_names)
%    text(x(i), y(i), ch_names{i}, 'FontSize', 7, 'HorizontalAlignment', 'center');
%end

% Add head outline
theta = linspace(0, 2*pi, 100);
headRadius = 0.95;
earRadius = 0.065;
noseLength = 0.15;

% Head
plot(headRadius*cos(theta), headRadius*sin(theta), 'k', 'LineWidth', 2);

% Nose
noseX = [0, 0, 0];
noseY = [headRadius, headRadius+noseLength, headRadius];
plot(noseX, noseY, 'k', 'LineWidth', 2);

% Left ear
earX = headRadius*cos(pi/2 + pi/16);
earY = headRadius*sin(pi/2 + pi/16);
plot([earX, earX-earRadius], [earY, earY], 'k', 'LineWidth', 2);

% Right ear
earX = headRadius*cos(pi/2 - pi/16);
earY = headRadius*sin(pi/2 - pi/16);
plot([earX, earX+earRadius], [earY, earY], 'k', 'LineWidth', 2);

% Set plot properties
axis equal;
axis off;
%colorbar;
colormap(jet);
%title('EEG Topographic Map');
end