
clear

%% Parameters to set
dataDir = 'G:/Resting state/'; % data directory
outputDir = 'G:/Resting state/'; 
saveIndividualSessions = true; % Set to true to save results for each session
saveAveragedSessions = true;  
saveFigures = true;            
subjectList = {'sub-08'}; 
sessionList = {'ses-01', 'ses-02', 'ses-03','ses-04'};

%% This is used
% Frequency parameters
numfrex = 20;
lowfreq = 8;  % Hz
highfreq = 13; % Hz
frex = logspace(log10(lowfreq), log10(highfreq), numfrex);
%frex = linspace(8,13,numfrex);
stds = linspace(2, 5, numfrex);

%%
%channelNames = [ 'Fp1 '; 'F3 '; 'F7 '; 'FT9 '; 'FC5 '; 'FC1 '; 'C3 '; 'T7 '; 'TP9 '; 'CP5 '; 'CP1 '; 'Pz '; 'P3 '; 'P7 '; 'O1 '; 'Oz '; 'O2 '; 'P4 '; 'P8 '; 'TP10'; 'CP6 '; 'CP2 '; 'Cz '; 'C4 '; 'T8 '; 'FT10'; 'FC6 '; 'FC2 '; 'F4 '; 'F8 '; 'Fp2 '; 'AF7 '; 'AF3 '; 'AFz '; 'F1 '; 'F5 '; 'FT7 '; 'FC3 '; 'FCz '; 'C1 '; 'C5 '; 'TP7 '; 'CP3 '; 'P1 '; 'P5 '; 'PO7 '; 'PO3 '; 'POz '; 'PO4 '; 'PO8 '; 'P6 '; 'P2 '; 'CPz '; 'CP4 '; 'TP8 '; 'C6 '; 'C2 '; 'FC4 '; 'FT8 '; 'F6 '; 'F2 '; 'AF4 '; 'AF8 ' ];
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
        fprintf('Processing session %s\n', sessionID);
        
        % Load data file
        dataFilename = fullfile(dataDir, [subjectID '\' subjectID '\' sessionID '\data.mat']);
        load(dataFilename);
        data.nbchan = 17;
        %data.nbchan = 63;

        %data.raw_eeg_data = data.raw_eeg_data(1:63,:);
        data.raw_eeg_data = data.raw_eeg_data(occipitalPosteriorIndices, :);

        % Adjust dimensions if this is the first session (once we know data dimensions)
        if sesi == 1
            allSessionsMap = zeros(length(sessionList), numfrex, data.nbchan);
            allSessionsEvecs = zeros(length(sessionList), numfrex, data.nbchan);
            allSessionsEvals = zeros(length(sessionList), numfrex, data.nbchan);
        end
        % Define segments for epoching resting-state data
        onsets = data.sfre*2:2*data.sfre:data.pnts-data.sfre*4;
        snipn = 2*data.sfre;
        % Initialize session-specific variables
        [evals, evecs, maps] = deal(zeros(numfrex, data.nbchan));
        

        
        %Create R covariance matrix
        R = zeros(length(onsets), data.nbchan, data.nbchan);
        for segIdx = 1:length(onsets)
            snipdat = data.raw_eeg_data(:, onsets(segIdx):onsets(segIdx)+snipn);
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
        Rr = R*(1-gamma) + eye(data.nbchan)*gamma*mean(eig(R));
        
        %Loop over frequencies
        for fi = 1:numfrex
            % Filter data
            fdat = filterFGx(data.raw_eeg_data, data.sfre, frex(fi), stds(fi));
            
            % Compute S
            S = zeros(length(onsets), data.nbchan, data.nbchan);
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
        
        % Correlation matrices for clustering
        E = zscore(evecs, [], 2);
        evecCorMat = (E*E'/(data.nbchan-1)).^2;
        
        % Determine the optimal epsilon value
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
        
        % Average correlation coefficient within each cluster
        avecorcoef = zeros(max(freqbands), 2);
        for i = 1:max(freqbands)
            submat = evecCorMat(freqbands==i, freqbands==i);
            avecorcoef(i, 1) = mean(nonzeros(tril(submat, -1)));
            avecorcoef(i, 2) = mean(frex(freqbands==i));
        end
        
        % Store session results
        allSessionsMap(sesi, :, :) = maps;
        allSessionsEvecs(sesi, :, :) = evecs;
        allSessionsEvals(sesi, :, :) = evals;
        allSessionsCorMat{sesi} = evecCorMat;
        allSessionsFreqBands{sesi} = freqbands;
        allSessionsAveCorCoef{sesi} = avecorcoef;
        
        % Save individual session outputs if requested
        if saveIndividualSessions
            % Create output filename
            outfilename = fullfile(outputDir, [subjectID '\' subjectID '\' sessionID '\gedbounds.mat']);
            groupidx = freqbands; % This variable is used in original code's save
            
            % Create a diary file to save printed output
            diaryFilename = fullfile(outputDir, [subjectID '\' subjectID '\' sessionID '\freqbands.txt']);
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
            
            diary on;
            
            % Save the frequency band limits
            allSessionsFreqBandLimits{sesi} = freqbandLimits;
            
            % Save results
            save(outfilename, 'maps', 'evecs', 'evals', 'frex', 'evecCorMat', 'groupidx', ...
                'channelNames', 'avecorcoef', 'epsis', 'qvec', 'epsiidx', 'freqbandLimits');
            
            %Plot and save figures for individual session if requested
            if saveFigures
                % Plot GED eigenvalues
                % h0 = figure;
                % plot(evals(2, :)/max(evals(2, :)), 'ks-', 'markersize', 10, 'markerfacecolor', 'm');
                % axis square;
                % set(gca, 'xlim', [0 30.5]);
                % title('GED eigenvalues');
                % xlabel('Component number'), ylabel('Power ratio (\lambda)');
                % 
                % % Save eigenvalues figure
                % eigFilename = fullfile(outputDir, [subjectID '\' subjectID '\' sessionID '\eigenvalues.png']);
                % saveas(h0, eigFilename);
                
                
                % Correlation matrix and band boundaries
                h1 = figure;
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
                figFilename = fullfile(outputDir, [subjectID '\' subjectID '\' sessionID '\cormat.png']);
                saveas(h1, figFilename);
                %close(h1);
                
                % Plot the topoplots of eigenvectors - use test code pattern
                h2 = figure;
                %title(['E-vecs for RS ' subjectID ' ' sessionID]);
                for i = 1:min(max(freqbands), 12) % Limit to 12 subplots
                    subplot(3, 4, i);
                    m = pca(evecs(freqbands==i, :));
                    % Get the first principal component to plot
                    topoval = m(:, 1);
                    % Pass the channelNames directly to topoplot_eeg63
                    topoplot_eeg63(topoval, channelNames1);
                    title(['E-vecs: ' num2str(round(mean(frex(freqbands==i)), 2)) ' Hz']);
                end
                
                % Save topoplots figure
                figFilename = fullfile(outputDir, [subjectID '\' subjectID '\' sessionID '\evecstop.png']);
                saveas(h2, figFilename);
                %close(h2);
                
                h3 = figure;
                for i = 1:min(max(freqbands), 12) % Limit to 12 subplots
                    subplot(3, 4, i);
                    m = pca(maps(freqbands==i, :));
                    % Get the first principal component to plot
                    topoval = m(:, 1);
                    % Pass the channelNames directly to topoplot_eeg63
                    topoplot_eeg63(topoval, channelNames1);
                    title(['Maps: ' num2str(round(mean(frex(freqbands==i)), 2)) ' Hz']);

                    % Save topoplots figure
                figFilename = fullfile(outputDir, [subjectID '\' subjectID '\' sessionID '\mapstop.png']);
                saveas(h3, figFilename);
                %close(h3);

                end
            clear fdat S R Rr W L meanR meanS dists snipdat epsis qvec epsiidx newc cc MA M
            fprintf('cleared');
            end
        end
    end
    
    % Process averaged data (if requested)
    if saveAveragedSessions
        fprintf('  Computing averaged results for subject %s\n', subjectID);
        
        % Average maps, evecs, and evals across sessions
        avgMaps = squeeze(mean(allSessionsMap, 1));
        avgEvecs = squeeze(mean(allSessionsEvecs, 1));
        avgEvals = squeeze(mean(allSessionsEvals, 1));
        
        % Compute correlation matrix from averaged eigenvectors
        E = zscore(avgEvecs, [], 2);
        avgEvecCorMat = (E*E'/(data.nbchan-1)).^2;
        
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
        
        % Average correlation coefficient within each cluster
        avecorcoef = zeros(max(freqbands), 2);
        for i = 1:max(freqbands)
            submat = avgEvecCorMat(freqbands==i, freqbands==i);
            avecorcoef(i, 1) = mean(nonzeros(tril(submat, -1)));
            avecorcoef(i, 2) = mean(frex(freqbands==i));
        end
        
        % Save averaged outputs
        outfilename = fullfile(outputDir, [subjectID '\averaged_gedbounds.mat']);
        maps = avgMaps;
        evecs = avgEvecs;
        evals = avgEvals;
        evecCorMat = avgEvecCorMat;
        groupidx = freqbands;
        
        % Create a diary file to save printed output for averaged data
        diaryFilename = fullfile(outputDir, [subjectID '\averaged_freqbands.txt']);
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
        
        diary on;
        
        save(outfilename, 'maps', 'evecs', 'evals', 'frex', 'evecCorMat', 'groupidx', ...
            'channelNames', 'avecorcoef', 'epsis', 'qvec', 'epsiidx', 'freqbandLimits');
        
        % Plot and save averaged figures if requested
        if saveFigures
            % Plot GED eigenvalues for averaged data
            h0 = figure('visible', 'on');
            plot(evals(2, :)/max(evals(2, :)), 'ks-', 'markersize', 10, 'markerfacecolor', 'm');
            axis square;
            set(gca, 'xlim', [0 30.5]);
            title('GED eigenvalues (Averaged)');
            xlabel('Component number'), ylabel('Power ratio (\lambda)');
            
            % Save eigenvalues figure
            eigFilename = fullfile(outputDir, [subjectID '\averaged_eigenvalues.png']);
            saveas(h0, eigFilename);
            %close(h0);
            
            % Correlation matrix and band boundaries
            h1 = figure;
            %set(h2, 'Position', [100, 100, 1200, 900])
            colormap bone;
            
            imagesc(1-avgEvecCorMat), hold on;
            f2u = round(linspace(1, length(frex), 10));
            %set(gca, 'clim', [.2 1], 'xtick', f2u, 'xticklabel', round(frex(f2u), 1), ...
                %'ytick', f2u, 'yticklabel', round(frex(f2u), 1));
            %xlabel('Frequency (Hz)'), ylabel('Frequency (Hz)');
            set(gca,'clim',[.2 1],'xtick',f2u,'xticklabel',round(frex(f2u),1),'ytick',f2u,'yticklabel',round(frex(f2u),1),'FontSize', 12)
            xlabel('Frequency (Hz)','FontSize', 16), ylabel('Frequency (Hz)','FontSize', 16)
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
            figFilename = fullfile(outputDir, [subjectID '\averaged_cormat.png']);
            saveas(h1, figFilename);
            %print(h3, figFilename, '-dpng', '-r300');
            %close(h1);
            
            % Plot the topoplots of eigenvectors
            h2 = figure;
            set(h2, 'Position', [100, 100, 1200, 900])
            %title(['Maps for RS ' subjectID ' (Averaged)']);
            for i = 1:min(max(freqbands), 12) % Limit to 12 subplots
                subplot(3, 4, i);
                m = pca(evecs(freqbands==i, :));
                % Get the first principal component to plot
                topoval = m(:, 1);
                % Pass the channelNames to topoplot_eeg63
                topoplot_eeg63(topoval, channelNames1);
                title(['E-vecs: ' num2str(round(mean(frex(freqbands==i)), 2)) ' Hz']);
            end
            
            % Save topoplots figure
            figFilename = fullfile(outputDir, [subjectID '\averaged_evecs.png']);
            saveas(h2, figFilename);
            %print(h3, figFilename, '-dpng', '-r300');
           

            h3 = figure;
            set(h2, 'Position', [100, 100, 1200, 900])
            %title(['Maps for RS ' subjectID ' (Averaged)']);
            for i = 1:min(max(freqbands), 12) % Limit to 12 subplots
                subplot(3, 4, i);
                m = pca(maps(freqbands==i, :));
                % Get the first principal component to plot
                topoval = m(:, 1);
                % Pass the channelNames to topoplot_eeg63
                topoplot_eeg63(topoval, channelNames1);
                title(['Maps: ' num2str(round(mean(frex(freqbands==i)), 2)) ' Hz']);
            end
            
            % Save topoplots figure
            figFilename = fullfile(outputDir, [subjectID '\averaged_maps.png']);
            saveas(h3, figFilename);
            %print(h3, figFilename, '-dpng', '-r300');
            
        end


    end
end


