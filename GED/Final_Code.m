%% GEDBounds analysis for multiple sessions per subject
% This script extends the original gedBounds_empirical.m to handle multiple sessions
% per subject, with options for individual session analysis or averaging.
%
% Adapted from Mike X Cohen's original code

clear

%% Parameters to set
dataDir = 'G:/Resting state/'; % Set your data directory
outputDir = 'G:/Resting state/Avg'; % Set where to save outputs
saveIndividualSessions = true; % Set to true to save results for each session
saveAveragedSessions = true;   % Set to true to save averaged results
saveFigures = true;            % Set to true to save figures

% List of subjects to process
%subjectList = {'sub-01', 'sub-03'}; % Update with your subject IDs

subjectList = {'sub-06'};
sessionList = {'ses-01', 'ses-02', 'ses-03', 'ses-04'}; % Four sessions per subject

% Frequency parameters
numfrex = 100;
lowfreq = 2;  % Hz
highfreq = 80; % Hz
frex = logspace(log10(lowfreq), log10(highfreq), numfrex);
stds = linspace(2, 5, numfrex);

%% Loop through subjects
for subi = 1:length(subjectList)
    subjectID = subjectList{subi};
    fprintf('Processing subject %s\n', subjectID);
    
    % Initialize arrays for storing session-level results
    allSessionsMap = [];  % Will initialize properly after first load
    allSessionsEvecs = [];
    allSessionsEvals = [];
    allSessionsCorMat = cell(length(sessionList), 1);
    allSessionsFreqBands = cell(length(sessionList), 1);
    allSessionsAveCorCoef = cell(length(sessionList), 1);
    allSessionsFreqBandLimits = cell(length(sessionList), 1);
    
    % Process each session
    for sesi = 1:length(sessionList)
        sessionID = sessionList{sesi};
        fprintf('  Processing session %s\n', sessionID);
        
        % Load data file
        dataFilename = fullfile(dataDir, subjectID, subjectID, sessionID,'data.mat');
        load(dataFilename);
        
        % Get channel names if available
        if isfield(EEG, 'chanlocs') && isfield(EEG.chanlocs, 'labels')
            channelNames = {EEG.chanlocs.labels};
        else
            % Create generic channel names if not available
            channelNames = cell(EEG.nbchan, 1);
            for chi = 1:EEG.nbchan
                channelNames{chi} = ['Ch' num2str(chi)];
            end
        end
        
        % Adjust dimensions if this is the first session (once we know EEG dimensions)
        if sesi == 1
            allSessionsMap = zeros(length(sessionList), numfrex, EEG.nbchan);
            allSessionsEvecs = zeros(length(sessionList), numfrex, EEG.nbchan);
            allSessionsEvals = zeros(length(sessionList), numfrex, EEG.nbchan);
        end
        
        % Initialize session-specific variables
        [evals, evecs, maps] = deal(zeros(numfrex, EEG.nbchan));
        
        % Define onsets for epoching resting-state data
        onsets = EEG.srate*2:2*EEG.srate:EEG.pnts-EEG.srate*4;
        snipn = 2*EEG.srate;
        
        %% Create R covariance matrix
        R = zeros(length(onsets), EEG.nbchan, EEG.nbchan);
        for segIdx = 1:length(onsets)
            snipdat = EEG.data(:, onsets(segIdx):onsets(segIdx)+snipn);
            snipdat = bsxfun(@minus, snipdat, mean(snipdat, 2));
            R(segIdx, :, :) = snipdat*snipdat'/snipn;
        end
        
        % Clean R
        meanR = squeeze(mean(R));
        dists = zeros(1, size(R, 1));
        for segIdx = 1:size(R, 1)
            r = R(segIdx, :, :);
            dists(segIdx) = sqrt(sum((r(:)-meanR(:)).^2));
        end
        R = squeeze(mean(R(zscore(dists)<3, :, :), 1));
        
        % Regularized R
        gamma = .01;
        Rr = R*(1-gamma) + eye(EEG.nbchan)*gamma*mean(eig(R));
        
        %% Loop over frequencies
        for fi = 1:numfrex
            % Filter data
            fdat = filterFGx(EEG.data, EEG.srate, frex(fi), stds(fi));
            
            % Compute S
            S = zeros(length(onsets), EEG.nbchan, EEG.nbchan);
            for segIdx = 1:length(onsets)
                snipdat = fdat(:, onsets(segIdx):onsets(segIdx)+snipn);
                snipdat = bsxfun(@minus, snipdat, mean(snipdat, 2));
                S(segIdx, :, :) = snipdat*snipdat'/snipn;
            end
            
            % Clean S
            meanS = squeeze(mean(S));
            dists = zeros(1, size(S, 1));
            for segIdx = 1:size(S, 1)
                s = S(segIdx, :, :);
                dists(segIdx) = sqrt(sum((s(:)-meanS(:)).^2));
            end
            S = squeeze(mean(S(zscore(dists)<3, :, :), 1));
            
            % Global variance normalize
            S = S / (std(S(:))/std(R(:)));
            
            % GED
            [W, L] = eig(S, Rr);
            [evals(fi, :), sidx] = sort(diag(L), 'descend');
            W = W(:, sidx);
            
            % Store top component map and eigenvector
            maps(fi, :) = W(:, 1)'*S;
            evecs(fi, :) = W(:, 1);
        end
        
        %% Correlation matrices for clustering
        E = zscore(evecs, [], 2);
        evecCorMat = (E*E'/(EEG.nbchan-1)).^2;
        
        %% Determine the optimal epsilon value
        nepsis = 50;
        epsis = linspace(.001, .05, nepsis);
        qvec = nan(nepsis, 1);
        
        for epi = 1:length(epsis)
            % Scan
            freqbands = dbscan(evecCorMat, epsis(epi), 3, 'Distance', 'Correlation');
            if max(freqbands) < 4, continue; end
            
            % Compute q
            qtmp = zeros(max(freqbands), 1);
            MA = false(size(evecCorMat));
            for i = 1:max(freqbands)
                M = false(size(evecCorMat));
                M(freqbands==i, freqbands==i) = 1;
                qtmp(i) = mean(mean(evecCorMat(M))) / mean(mean(evecCorMat(~M)));
                MA = MA + M;
            end
            qvec(epi) = mean(qtmp) + log(mean(MA(:)));
        end
        
        % Run it again on the best epsilon value
        [~, epsiidx] = findpeaks(qvec, 'NPeaks', 1, 'SortStr', 'descend');
        if isempty(epsiidx), epsiidx = round(nepsis/2); end
        freqbands = dbscan(evecCorMat, epsis(epsiidx), 3, 'Distance', 'Correlation');
        
        % Dissolve tiny clusters, and renumber all clusters consecutively
        newc = cell(4, 1); n = 1;
        for i = 1:max(freqbands)
            cc = bwconncomp(freqbands==i);
            for ci = 1:cc.NumObjects
                if length(cc.PixelIdxList{ci}) > 2
                    newc{n} = cc.PixelIdxList{ci};
                    n = n + 1;
                end
            end
        end
        freqbands = -ones(size(frex));
        for ni = 1:n-1
            freqbands(newc{ni}) = ni;
        end
        
        %% Average correlation coefficient within each cluster
        avecorcoef = zeros(max(freqbands), 2);
        for i = 1:max(freqbands)
            submat = evecCorMat(freqbands==i, freqbands==i);
            avecorcoef(i, 1) = mean(nonzeros(tril(submat, -1)));
            avecorcoef(i, 2) = mean(frex(freqbands==i));
        end
        
        %% Store session results
        allSessionsMap(sesi, :, :) = maps;
        allSessionsEvecs(sesi, :, :) = evecs;
        allSessionsEvals(sesi, :, :) = evals;
        allSessionsCorMat{sesi} = evecCorMat;
        allSessionsFreqBands{sesi} = freqbands;
        allSessionsAveCorCoef{sesi} = avecorcoef;
        
        %% Save individual session outputs if requested
        if saveIndividualSessions
            % Create output filename
            outfilename = fullfile(outputDir, [subjectID '_' sessionID '_gedbounds.mat']);
            chanlocs = EEG.chanlocs;
            groupidx = freqbands; % This variable is used in original code's save
            
            % Create a diary file to save printed output
            diaryFilename = fullfile(outputDir, [subjectID '_' sessionID '_freqbands.txt']);
            diary(diaryFilename);
            
            fprintf('\n===== Frequency Bands for %s %s =====\n', subjectID, sessionID);
            tbnds_all = cell(max(freqbands), 1);
            freqbandLimits = zeros(max(freqbands), 2);
            
            for i = 1:max(freqbands)
                tbnds = frex(freqbands==i);
                tbnds = dsearchn(frex', tbnds([1 end])');
                tbnds_all{i} = tbnds;
                freqbandLimits(i, :) = [frex(tbnds(1)), frex(tbnds(2))];
                fprintf('Frequency Band %d: [%f Hz, %f Hz]\n', i, frex(tbnds(1)), frex(tbnds(2)));
            end
            
            diary off;
            
            % Save the frequency band limits
            allSessionsFreqBandLimits{sesi} = freqbandLimits;
            
            % Save results
            save(outfilename, 'maps', 'evecs', 'evals', 'frex', 'evecCorMat', 'groupidx', ...
                'chanlocs', 'avecorcoef', 'epsis', 'qvec', 'epsiidx', 'freqbandLimits');
            
               


            %% Plot and save figures for individual session if requested
            if saveFigures
                % Plot GED eigenvalues
                h0 = figure('visible', 'off');
                plot(evals(2, :)/max(evals(2, :)), 'ks-', 'markersize', 10, 'markerfacecolor', 'm');
                axis square;
                set(gca, 'xlim', [0 30.5]);
                title('GED eigenvalues');
                xlabel('Component number'), ylabel('Power ratio (\lambda)');
                
                % Save eigenvalues figure
                eigFilename = fullfile(outputDir, [subjectID '_' sessionID '_eigenvalues.png']);
                saveas(h0, eigFilename);
                close(h0);
                
                % Correlation matrix and band boundaries
                h1 = figure('visible', 'off');
                colormap bone;
                
                imagesc(1-evecCorMat), hold on;
                f2u = round(linspace(1, length(frex), 10));
                set(gca, 'clim', [.2 1], 'xtick', f2u, 'xticklabel', round(frex(f2u), 1), ...
                    'ytick', f2u, 'yticklabel', round(frex(f2u), 1));
                xlabel('Frequency (Hz)'), ylabel('Frequency (Hz)');
                axis square, axis xy;
                title(['Eigenvectors correlation matrix RS ' subjectID ' ' sessionID]);
                
                for i = 1:max(freqbands)
                    tbnds = frex(freqbands==i);
                    tbnds = dsearchn(frex', tbnds([1 end])');
                    
                    % Box
                    plot(tbnds, [1 1]*tbnds(1), 'm', 'linew', 2);
                    plot(tbnds, [1 1]*tbnds(2), 'm', 'linew', 2);
                    plot([1 1]*tbnds(1), tbnds, 'm', 'linew', 2);
                    plot([1 1]*tbnds(2), tbnds, 'm', 'linew', 2);
                end
                
                % Save correlation matrix figure
                figFilename = fullfile(outputDir, [subjectID '_' sessionID '_cormat.png']);
                saveas(h1, figFilename);
                close(h1);
                
                % Plot the topoplots of eigenvectors
                h2 = figure('visible', 'off');
                for i = 1:min(max(freqbands), 12) % Limit to 12 subplots
                    subplot(3, 4, i);
                    m = pca(evecs(freqbands==i, :));
                    % Get the first principal component to plot
                    topoval = m(:, 1);
                    % Pass the channelNames to topoplot_eeg63
                    topoplot_eeg63(topoval, channelNames);
                    title(['E-vecs: ' num2str(round(mean(frex(freqbands==i)), 2)) ' Hz']);
                end
                
                % Save topoplots figure
                figFilename = fullfile(outputDir, [subjectID '_' sessionID '_topos.png']);
                saveas(h2, figFilename);
                close(h2);
            end
        end
    end
    
    %% Process averaged data (if requested)
    if saveAveragedSessions
        fprintf('  Computing averaged results for subject %s\n', subjectID);
        
        % Average maps, evecs, and evals across sessions
        avgMaps = squeeze(mean(allSessionsMap, 1));
        avgEvecs = squeeze(mean(allSessionsEvecs, 1));
        avgEvals = squeeze(mean(allSessionsEvals, 1));
        
        % Compute correlation matrix from averaged eigenvectors
        E = zscore(avgEvecs, [], 2);
        avgEvecCorMat = (E*E'/(EEG.nbchan-1)).^2;
        
        % Determine the optimal epsilon value for averaged data
        nepsis = 50;
        epsis = linspace(.001, .05, nepsis);
        qvec = nan(nepsis, 1);
        
        for epi = 1:length(epsis)
            % Scan
            freqbands = dbscan(avgEvecCorMat, epsis(epi), 3, 'Distance', 'Correlation');
            if max(freqbands) < 4, continue; end
            
            % Compute q
            qtmp = zeros(max(freqbands), 1);
            MA = false(size(avgEvecCorMat));
            for i = 1:max(freqbands)
                M = false(size(avgEvecCorMat));
                M(freqbands==i, freqbands==i) = 1;
                qtmp(i) = mean(mean(avgEvecCorMat(M))) / mean(mean(avgEvecCorMat(~M)));
                MA = MA + M;
            end
            qvec(epi) = mean(qtmp) + log(mean(MA(:)));
        end
        
        % Run it again on the best epsilon value
        [~, epsiidx] = findpeaks(qvec, 'NPeaks', 1, 'SortStr', 'descend');
        if isempty(epsiidx), epsiidx = round(nepsis/2); end
        freqbands = dbscan(avgEvecCorMat, epsis(epsiidx), 3, 'Distance', 'Correlation');
        
        % Dissolve tiny clusters, and renumber all clusters consecutively
        newc = cell(4, 1); n = 1;
        for i = 1:max(freqbands)
            cc = bwconncomp(freqbands==i);
            for ci = 1:cc.NumObjects
                if length(cc.PixelIdxList{ci}) > 2
                    newc{n} = cc.PixelIdxList{ci};
                    n = n + 1;
                end
            end
        end
        freqbands = -ones(size(frex));
        for ni = 1:n-1
            freqbands(newc{ni}) = ni;
        end
        
        %% Average correlation coefficient within each cluster
        avecorcoef = zeros(max(freqbands), 2);
        for i = 1:max(freqbands)
            submat = avgEvecCorMat(freqbands==i, freqbands==i);
            avecorcoef(i, 1) = mean(nonzeros(tril(submat, -1)));
            avecorcoef(i, 2) = mean(frex(freqbands==i));
        end
        
        %% Save averaged outputs
        outfilename = fullfile(outputDir, [subjectID '_averaged_gedbounds.mat']);
        maps = avgMaps;
        evecs = avgEvecs;
        evals = avgEvals;
        evecCorMat = avgEvecCorMat;
        groupidx = freqbands;
        
        % Create a diary file to save printed output for averaged data
        diaryFilename = fullfile(outputDir, [subjectID '_averaged_freqbands.txt']);
        diary(diaryFilename);
        
        fprintf('\n===== Frequency Bands for %s (Averaged) =====\n', subjectID);
        tbnds_all = cell(max(freqbands), 1);
        freqbandLimits = zeros(max(freqbands), 2);
        
        for i = 1:max(freqbands)
            tbnds = frex(freqbands==i);
            tbnds = dsearchn(frex', tbnds([1 end])');
            tbnds_all{i} = tbnds;
            freqbandLimits(i, :) = [frex(tbnds(1)), frex(tbnds(2))];
            fprintf('Frequency Band %d: [%f Hz, %f Hz]\n', i, frex(tbnds(1)), frex(tbnds(2)));
        end
        
        diary off;
        
        save(outfilename, 'maps', 'evecs', 'evals', 'frex', 'evecCorMat', 'groupidx', ...
            'chanlocs', 'avecorcoef', 'epsis', 'qvec', 'epsiidx', 'freqbandLimits');
        
        
        
     
        %% Plot and save averaged figures if requested
        if saveFigures
            % Plot GED eigenvalues for averaged data
            h0 = figure('visible', 'off');
            plot(evals(2, :)/max(evals(2, :)), 'ks-', 'markersize', 10, 'markerfacecolor', 'm');
            axis square;
            set(gca, 'xlim', [0 30.5]);
            title('GED eigenvalues (Averaged)');
            xlabel('Component number'), ylabel('Power ratio (\lambda)');
            
            % Save eigenvalues figure
            eigFilename = fullfile(outputDir, [subjectID '_averaged_eigenvalues.png']);
            saveas(h0, eigFilename);
            close(h0);
            
            % Correlation matrix and band boundaries
            h1 = figure('visible', 'off');
            colormap bone;
            
            imagesc(1-avgEvecCorMat), hold on;
            f2u = round(linspace(1, length(frex), 10));
            set(gca, 'clim', [.2 1], 'xtick', f2u, 'xticklabel', round(frex(f2u), 1), ...
                'ytick', f2u, 'yticklabel', round(frex(f2u), 1));
            xlabel('Frequency (Hz)'), ylabel('Frequency (Hz)');
            axis square, axis xy;
            title(['Eigenvectors correlation matrix RS ' subjectID ' (Averaged)']);
            
            for i = 1:max(freqbands)
                tbnds = frex(freqbands==i);
                tbnds = dsearchn(frex', tbnds([1 end])');
                
                % Box
                plot(tbnds, [1 1]*tbnds(1), 'm', 'linew', 2);
                plot(tbnds, [1 1]*tbnds(2), 'm', 'linew', 2);
                plot([1 1]*tbnds(1), tbnds, 'm', 'linew', 2);
                plot([1 1]*tbnds(2), tbnds, 'm', 'linew', 2);
            end
            
            % Save correlation matrix figure
            figFilename = fullfile(outputDir, [subjectID '_averaged_cormat.png']);
            saveas(h1, figFilename);
            close(h1);
            
            % Plot the topoplots of eigenvectors
            h2 = figure('visible', 'off');
            for i = 1:min(max(freqbands), 12) % Limit to 12 subplots
                subplot(3, 4, i);
                m = pca(evecs(freqbands==i, :));
                % Get the first principal component to plot
                topoval = m(:, 1);
                % Pass the channelNames to topoplot_eeg63
                topoplot_eeg63(topoval, channelNames);
                title(['E-vecs: ' num2str(round(mean(frex(freqbands==i)), 2)) ' Hz']);
            end
            
            % Save topoplots figure
            figFilename = fullfile(outputDir, [subjectID '_averaged_topos.png']);
            saveas(h2, figFilename);
            close(h2);
        end
    end
end

%% Compare frequency bands across subjects
if saveAveragedSessions
    fprintf('Comparing frequency bands across subjects...\n');
    
    % Load all subject data
    allSubjectsBands = cell(length(subjectList), 1);
    allSubjectsBandLimits = cell(length(subjectList), 1);
    
    for subi = 1:length(subjectList)
        subjectID = subjectList{subi};
        dataFilename = fullfile(outputDir, [subjectID '_averaged_gedbounds.mat']);
        data = load(dataFilename);
        allSubjectsBands{subi} = data.groupidx;
        allSubjectsBandLimits{subi} = data.freqbandLimits;
    end
    
    % Create comparison figure
    h3 = figure;
    for subi = 1:length(subjectList)
        subplot(length(subjectList), 1, subi);
        % Create a "rainbow" plot showing frequency bands
        imagesc(frex, [0 1], allSubjectsBands{subi}');
        colormap(gca, jet);
        ylabel(subjectList{subi});
        if subi == length(subjectList)
            xlabel('Frequency (Hz)');
        end
        set(gca, 'YTick', []);
        % Set frequency ticks
        freqTicks = [2 5 10 20 40 80];
        freqTickIdxs = zeros(size(freqTicks));
        for i = 1:length(freqTicks)
            [~, freqTickIdxs(i)] = min(abs(frex - freqTicks(i)));
        end
        set(gca, 'XTick', frex(freqTickIdxs), 'XTickLabel', freqTicks);
        set(gca, 'XScale', 'log');
    end
    
    % Save comparison figure
    figFilename = fullfile(outputDir, 'all_subjects_frequency_bands.png');
    saveas(h3, figFilename);
    
    % Create table of frequency band limits across subjects
    h4 = figure('Position', [100, 100, 800, 400]);
    maxBands = 0;
    for subi = 1:length(subjectList)
        maxBands = max(maxBands, size(allSubjectsBandLimits{subi}, 1));
    end
    
    % Create a table to display all frequency bands
    bandData = cell(length(subjectList), maxBands*2);
    rowNames = cell(length(subjectList), 1);
    colNames = cell(1, maxBands*2);
    
    for subi = 1:length(subjectList)
        rowNames{subi} = subjectList{subi};
        numBands = size(allSubjectsBandLimits{subi}, 1);
        
        for bandi = 1:numBands
            colNames{bandi*2-1} = ['Band' num2str(bandi) ' Low'];
            colNames{bandi*2} = ['Band' num2str(bandi) ' High'];
            
            bandData{subi, bandi*2-1} = num2str(allSubjectsBandLimits{subi}(bandi, 1), '%.2f');
            bandData{subi, bandi*2} = num2str(allSubjectsBandLimits{subi}(bandi, 2), '%.2f');
        end
    end
    
    % Create table for display
    uitable(h4, 'Data', bandData, 'RowName', rowNames, 'ColumnName', colNames, ...
        'Position', [20 20 760 350]);
    
    % Save the table figure
    figFilename = fullfile(outputDir, 'all_subjects_band_limits_table.png');
    saveas(h4, figFilename);
    
    % Also save the table data as a CSV file
    fid = fopen(fullfile(outputDir, 'all_subjects_band_limits.csv'), 'w');
    fprintf(fid, 'Subject');
    for i = 1:length(colNames)
        if ~isempty(colNames{i})
            fprintf(fid, ',%s', colNames{i});
        end
    end
    fprintf(fid, '\n');
    
    for subi = 1:length(subjectList)
        fprintf(fid, '%s', rowNames{subi});
        for bandi = 1:maxBands*2
            if bandi <= size(bandData, 2) && ~isempty(bandData{subi, bandi})
                fprintf(fid, ',%s', bandData{subi, bandi});
            else
                fprintf(fid, ',');
            end
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
    
    close(h3);
    close(h4);
end

fprintf('Processing complete!\n');


%%
clear

%%

dataDir = 'G:/Resting state/sub-01/sub-01/ses-01'; % Set your data directory
outputDir = 'G:/Resting state/sub-01/sub-01/ses1fig'; % Set where to save outputs
saveIndividualSessions = true; % Set to true to save results for each session
saveAveragedSessions = true;   % Set to true to save averaged results
saveFigures = true;   

%%
subjectList = {'sub-06'};
sessionList = {'ses-01', 'ses-02', 'ses-03', 'ses-04'}; % Four sessions per subject
%%
channelNames = [
    'Fp1 '; 'F3  '; 'F7  '; 'FT9 '; 'FC5 '; 'FC1 '; 'C3  '; 'T7  '; 'TP9 ';
    'CP5 '; 'CP1 '; 'Pz  '; 'P3  '; 'P7  '; 'O1  '; 'Oz  '; 'O2  '; 'P4  ';
    'P8  '; 'TP10'; 'CP6 '; 'CP2 '; 'Cz  '; 'C4  '; 'T8  '; 'FT10'; 'FC6 ';
    'FC2 '; 'F4  '; 'F8  '; 'Fp2 '; 'AF7 '; 'AF3 '; 'AFz '; 'F1  '; 'F5  ';
    'FT7 '; 'FC3 '; 'FCz '; 'C1  '; 'C5  '; 'TP7 '; 'CP3 '; 'P1  '; 'P5  ';
    'PO7 '; 'PO3 '; 'POz '; 'PO4 '; 'PO8 '; 'P6  '; 'P2  '; 'CPz '; 'CP4 ';
    'TP8 '; 'C6  '; 'C2  '; 'FC4 '; 'FT8 '; 'F6  '; 'F2  '; 'AF4 '; 'AF8 '
];
%%
%for sub 7
% Define occipital and posterior channels
occipitalPosteriorChannels = {'O1', 'Oz', 'O2', 'P1', 'P2', 'P3', 'P4', 'P5', ...
'P6', 'P7', 'P8', 'Pz', 'PO3', 'PO4', 'PO7', 'PO8', 'POz'};

% Find indices of occipital/posterior channels
occipitalPosteriorIndices = [];
for i = 1:length(occipitalPosteriorChannels)
    for j = 1:size(channelNames, 1)
        % Get the current channel name and trim whitespace
        currentChannelName = strtrim(channelNames(j,:));
        
        % Compare with the target channel
        if strcmp(currentChannelName, occipitalPosteriorChannels{i})
            occipitalPosteriorIndices = [occipitalPosteriorIndices j];
            break;
        end
    end
end
channelNames1 = channelNames(occipitalPosteriorIndices, :);
%%
%subjectID = 'sub-06';
subjectID = 'sub-01';
%subjectID = 'sub-07';
%load averaged_gedbounds
load gedbounds
%maps = avgMaps;
%evecs = avgEvecs;
%evals = avgEvals;
avgEvecCorMat = evecCorMat;
freqbands = groupidx;

        %% Plot and save averaged figures if requested
        if saveFigures
            % Plot GED eigenvalues for averaged data
            % h0 = figure('visible', 'off');
            % plot(evals(2, :)/max(evals(2, :)), 'ks-', 'markersize', 10, 'markerfacecolor', 'm');
            % axis square;
            % set(gca, 'xlim', [0 30.5], 'FontSize', 12);
            % %title('GED eigenvalues (Averaged)');
            % xlabel('Component number','FontSize', 16), ylabel('Power ratio (\lambda)', 'FontSize', 16);
            % 
            % % Save eigenvalues figure
            % eigFilename = fullfile(outputDir, [subjectID '_averaged_eigenvalues.png']);
            % %saveas(h0, eigFilename);
            % print(h0, eigFilename, '-dpng', '-r300');
            % close(h0);
            
            % Correlation matrix and band boundaries
            h1 = figure('visible', 'off');
            colormap bone;
            
            imagesc(1-avgEvecCorMat), hold on;
            f2u = round(linspace(1, length(frex), 10));
            set(gca, 'clim', [.2 1], 'xtick', f2u, 'xticklabel', round(frex(f2u), 1), ...
                'ytick', f2u, 'yticklabel', round(frex(f2u), 1), 'FontSize', 12);
            xlabel('Frequency (Hz)','FontSize', 16), ylabel('Frequency (Hz)','FontSize', 16);
            axis square, axis xy;
            %title(['Eigenvectors correlation matrix RS ' subjectID ' (Averaged)']);
            
            for i = 1:max(freqbands)
                tbnds = frex(freqbands==i);
                tbnds = dsearchn(frex', tbnds([1 end])');
                
                % Box
                plot(tbnds, [1 1]*tbnds(1), 'm', 'linew', 2);
                plot(tbnds, [1 1]*tbnds(2), 'm', 'linew', 2);
                plot([1 1]*tbnds(1), tbnds, 'm', 'linew', 2);
                plot([1 1]*tbnds(2), tbnds, 'm', 'linew', 2);
            end
            
            % Save correlation matrix figure
            figFilename = fullfile(outputDir, [subjectID '_averaged_cormat.png']);
            %saveas(h1, figFilename, '-dpng', '-r300');
            print(h1, figFilename, '-dpng', '-r300');
            close(h1);
            
            % Plot the topoplots of eigenvectors
            h2 = figure('visible', 'off');
            set(h2, 'Position', [100, 100, 1200, 900]); % [left, bottom, width, height] in pixels
            for i = 1:min(max(freqbands), 12) % Limit to 12 subplots
                subplot(3, 4, i);
                m = pca(evecs(freqbands==i, :));
                % Get the first principal component to plot
                topoval = m(:, 1);
                % Pass the channelNames to topoplot_eeg63
                topoplot_eeg63(topoval, channelNames);
                title(['E-vecs: ' num2str(round(mean(frex(freqbands==i)), 2)) ' Hz']);
            end
            
            % Save topoplots figure
            figFilename = fullfile(outputDir, [subjectID '_averaged_topos.png']);
            %saveas(h2, figFilename, '-dpng', '-r300');
            print(h2, figFilename, '-dpng', '-r300');
            close(h2);


            h3 = figure;
            set(h3, 'Position', [100, 100, 1200, 900]); % [left, bottom, width, height] in pixels
            %title(['Maps for RS ' subjectID ' (Averaged)']);
            for i = 1:min(max(freqbands), 12) % Limit to 12 subplots
                subplot(3, 4, i);
                m = pca(maps(freqbands==i, :));
                % Get the first principal component to plot
                topoval = m(:, 1);
                % Pass the channelNames to topoplot_eeg63
                topoplot_eeg63(topoval, channelNames);
                title(['Maps: ' num2str(round(mean(frex(freqbands==i)), 2)) ' Hz']);
            end
            
            % Save topoplots figure
            figFilename = fullfile(outputDir, [subjectID '_averaged_maps.png']);
            %saveas(h3, figFilename, '-dpng', '-r300');
            print(h3, figFilename, '-dpng', '-r300');
            close(h3);
        end
            
