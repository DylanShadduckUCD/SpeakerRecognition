% Dylan Shadduck
% EEC 201 Speaker Recognition
% Winter 2021

clear
clc

% Setting the training and testing folders
train_folder = "C:\Users\Dylan\Documents\Winter 2021\EEC 201\SpeakerRecognition\Train\";
test_folder = "C:\Users\Dylan\Documents\Winter 2021\EEC 201\SpeakerRecognition\Test\";

% Setting the seed for random number generation for reproducible results
rng("default")

% Start by training the model
N = 256;
M = 100;
K = 20;
clusters = 4;
C = train_model(train_folder, N, M, K, clusters);

% Now let's test our model
predict_speaker(C, test_folder, train_folder, N, M, K)

% Plotting
generate_plots(train_folder);

%% Functions

function B = generate_plots(train_folder)
    % This function takes the input training data and displays several
    % plots along the process of converting an audio waveform into mel
    % frequency ceptstrum coefficients
    
    % Getting a list of all the .wav files in the folder
    filepattern = fullfile(train_folder, "*.wav");
    theFiles = dir(filepattern);
    
    % Create a variable to keep track of our figure count
    fig_count = 1;
    
    base_auido_plot = figure("name", "Test 1: Raw Audio");
    hold on
    fig_count = fig_count + 1;
    
    for k=1:12
        baseFileName = theFiles(k).name;
        split_name = split(baseFileName, ".");
        train_speaker = strip(string(split_name(1)), "left", "s");
        fullFileName = fullfile(theFiles(k).folder, baseFileName);
       
        % Read the .wav file
        [y, fs] = audioread(fullFileName);
        
        % plot the audio file
        t = (0:1/fs:(length(y) - 1)/fs);
        subplot(4,3,k)
        plot(t, y)
        title(strcat("Speaker ", train_speaker))
        xlabel("time (s)")
        ylabel("Volume")
    end
    hold off
    
    % Now let's plot the trimmed data
    
    % Generating trimmed audio figure
    trimmed_audio_plot = figure("name", "Test 1: Trimmed/Normalized Audio");
    hold on
    fig_count = fig_count + 1;
    
    for k=1:12
        baseFileName = theFiles(k).name;
        split_name = split(baseFileName, ".");
        train_speaker = strip(string(split_name(1)), "left", "s");
        fullFileName = fullfile(theFiles(k).folder, baseFileName);
       
        % Read the .wav file
        [y, fs] = audioread(fullFileName);
        
        % Normalizing the data
        y = y./max(abs(y));

        % Trimming the audio using minimum normalized volume of 0.05
        y = trim_silence(y, 0.05);
        
        % plot the audio file
        t = (0:1/fs:(length(y) - 1)/fs);
        subplot(4,3,k)
        plot(t, y)
        title(strcat("Speaker ", train_speaker))
        xlabel("time (s)")
        ylabel("Volume")
    end
    hold off
    
    % Moving on to plotting stft using different sizes of N and M
    % We will only need to demonstrate this for one signal
    % I will use speaker one as my default
    [y, fs] = audioread(strcat(train_folder, "s1.wav"));
    
    % normalize and trim
    y = trim_silence(y, 0.05);
    
    figure("name", "Test 2: STFT")
    fig_count = fig_count + 1;
    loop_count = 1;
    for N=[128, 256, 512]
        M = floor(N/3);
    
        % perform frame blocking
        block_matrix = frame_block(y, N, M);

        % Apply windowing to our framed signal
        windowed_block = matrix_hamm(block_matrix);
        windowT = transpose(windowed_block);

        % Find the fft of the windowed block
        block_fft = zeros(1+floor(N/2), length(windowed_block(:, 1)));

        % We only want to keep the positive portion of the fft
        for n=1:length(windowed_block(:, 1))
            % Computing the fft across the row
            row_fft = fft(windowT(:, n));
            block_fft(:, n) = row_fft(1:1+floor(N/2));
        end
        
        % Transpose our block fft back
        block_fft = transpose(block_fft);

        % Calculate the power of the fft
        power_fft = abs(block_fft).^2;
    
        subplot(1, 3, loop_count)
        stft(y,fs,'Window',hamming(N),'OverlapLength',M,'FFTLength',N, "FrequencyRange", "onesided")
        title(strcat("STFT N = ", string(N)))
        
        % Increment the loop count
        loop_count = loop_count + 1;
    end
    
    % Plotting the mel frequency banks
    m = mel_bank(fs, 256, 20, true);
    
    mel_fig = figure("name", "Test 3: Mel Filter Bank");
    hold on
    for n=1:20
        plot(linspace(0, floor(256/2), floor(256/2 + 1)), m(n,:))
    end
    hold off
    title("Mel Filters")
    
    % Getting data for a second user for comparison
    [y2, fs] = audioread(strcat(train_folder, "s6.wav"));
    y2 = y2./max(abs(y2));
    y2 = trim_silence(y2, 0.05);
    
    % Calculating mfcc and plotting the points for two users
    N = 256;
    M = 100;
    K = 20;
    mfcc_1 = get_mfcc(y, fs, N, M, K, false);
    mfcc_2 = get_mfcc(y2, fs, N, M, K, false);
    
    % Plotting the mfcc for two users
    mfcc_plot = figure("name", "Test 5: MFCC Comparison");
    fig_count = fig_count + 1;
    hold on
    scatter(mfcc_1(7, :), mfcc_1(15, :))
    scatter(mfcc_2(7, :), mfcc_2(15, :))
    legend("User 1", "User 6")
    xlabel("MFCC 7")
    ylabel("MFCC 15")
    hold off
    
    % Calculate the centroids for each using kmeans with K = 4
    [~, C1] = kmeans(mfcc_1', 4);
    [~, C2] = kmeans(mfcc_2', 4);
    
    % Plotting the same mfccs with centroids this time
    kmeans_plot = figure("name", "Test 6: MFCC With Centroids");
    fig_count = fig_count + 1;
    hold on
    scatter(mfcc_1(7, :), mfcc_1(15, :))
    scatter(C1(:, 7), C1(:,15), [], "rx")
    scatter(mfcc_2(7, :), mfcc_2(15, :))
    scatter(C2(:, 7), C2(:,15), [], "kx")
    legend("User 1", "Centroids User 1", "User 6", "Centroids User 6")
    xlabel("MFCC 7")
    ylabel("MFCC 15")
    hold off

end

function predict_speaker(trained_centroids, test_folder, train_folder, fft_size, overlap_len, mel_filt_num)
    % This function takes two inputs. The first is an N-D array of
    % centroids that have been trained on certain speakers in a training
    % data set. This function attempts to fit each speaker in a test data
    % set to the existing trained models and prints out which speaker from
    % the training set it thinks that the speaker in the test data set
    % belonds to
    %
    % This function is meant to be used with the train_model function
    % below and it assumes that each train file is named in the format 
    % sX.wav where X is the speaker number
    
    % Getting a list of all the .wav files in the folder
    filepattern = fullfile(test_folder, "*.wav");
    theFiles = dir(filepattern);
    
    % Define a minimum volume threshold for trimming the dead space in each
    % .wav file. I've found a minimum normalized volume of 0.05 works well
    vol_min = 0.05;
    
    for k=1:length(theFiles)
    
        baseFileName = theFiles(k).name;
        split_name = split(baseFileName, ".");
        test_speaker = strip(string(split_name(1)), "left", "s");
        fullFileName = fullfile(theFiles(k).folder, baseFileName);
           
        % Read the .wav file
        [y, fs] = audioread(fullFileName);

        % Normalizing the data
        y = y./max(abs(y));

        % Trimming the audio
        y = trim_silence(y, vol_min);

        % Calculate the mfcc coefficients
        mfcc = get_mfcc(y, fs, fft_size, overlap_len, mel_filt_num, false);
        
        % Now we need to check which of our trained speakers fits this data
        % the best
        centroid_shape = size(trained_centroids);
        num_speakers = centroid_shape(1);
        
        % Now we need to fit the mfcc data to all the trained speaker
        % centroids
        distances = zeros(1, num_speakers);
        
        for w=1:num_speakers
            
            % Extract the centroids for each individual speaker
            C = squeeze(trained_centroids(w, :, :));
            
            % Fitting our test data to the centroids and finding the distance
            distance = pdist2(C, mfcc', "Euclidean", "smallest", 1);

            % Appending this distance value to our distances array
            distances(w) = sum(distance);
        end
        
        % Find the index that corresponds to the smallest sum distance
        [~, indx] = min(distances);
        
        % Finding the speaker number from the filename
        trainFilePattern = fullfile(train_folder, "*.wav");
        trainFiles = dir(trainFilePattern);
        speaker_file_name = split(trainFiles(indx).name, ".");
        predict_speaker = strip(string(speaker_file_name(1)), "left", "s");
        
        % print out the speaker filename for now
        fprintf("Test file %s is predicted to be speaker: %s\n", test_speaker, predict_speaker);
    end
end

function C = train_model(train_folder, fft_size, overlap_len, mel_filt_num, num_clusters)
    % This function finds the mel frequency cepstrum coefficients for audio
    % files and then performs a kmeans clustering algorithm on those
    % coefficients for each .wav file in the training folder. The output of
    % this function is an N-D array with all the centroids for the clusters
    % so that the kmeans models produced here can be used for fitting new
    % data.
    
    % Getting a list of all the .wav files in the folder
    filepattern = fullfile(train_folder, "*.wav");
    theFiles = dir(filepattern);
    
    % Initialize the centroid array
    C = zeros(length(theFiles), num_clusters, mel_filt_num-1);
    
    % Define a minimum volume threshold for trimming the dead space in each
    % .wav file. I've found a minimum normalized volume of 0.05 works well
    vol_min = 0.05;
    
    for k=1:length(theFiles)
    
        baseFileName = theFiles(k).name;
        fullFileName = fullfile(theFiles(k).folder, baseFileName);
           
        % Read the .wav file
        [y, fs] = audioread(fullFileName);

        % Normalizing the data
        y = y./max(abs(y));

        % Trimming the audio
        y = trim_silence(y, vol_min);

        % Calculate the mfcc coefficients
        mfcc = get_mfcc(y, fs, fft_size, overlap_len, mel_filt_num, false);

        % Find the clusters for centroids number of groups using the kmeans
        % algorithm
        [~, centroids] = kmeans(mfcc', 4);

        % Append our centroids to our user centroid array
        C(k, :, :) = centroids;
    end
end


function y = trim_silence(audio_data, min_volume)
    % Minimum volume is set assuming that the audio volume has already been
    % normalized to one
    
    % Apply premphasis filter
    % This filtering balances the frequency spectrum since high and low
    % frequnecies are not perceived to have the same volume
    % Also avoids numerical issuses in fft and could improve snr
%     alpha = 0.97;
%     for n=1:length(audio_data)-1
%         audio_data(n+1) = audio_data(n+1) - alpha*audio_data(n);
%     end
    
    % Defining a boolean variable to identify if an audio sample was
    % originally stereo
    stereo = false;
    
    % Turning a stereo file into mono
    if length(audio_data(1, :)) == 2
        
        % This is a stereo sample
        stereo = true;
        
        % Using the first 1000 samples to find the average DC offset
        dc_offset = sum(audio_data(1:1000))/1000;
        audio_data = ((audio_data(:,1) + audio_data(:,2)) ./ 2) - dc_offset;
        
    end
    
    % Some audio samples have quite a bit of noise in the sections that we
    % want to trim
    % I can apply a median filter to the audio sample for trimming, but I
    % don't want to filter the output
    
    % Applying median filter over 500 samples
    if stereo
        min_volume = min_volume*1.75;
    end
    
    % Finding first index above our threshold value
    first_index = find(abs(audio_data) > min_volume, 1);
    
    % Finding last index above our threshold value
    flipped = flip(audio_data);
    last_index = length(audio_data) - find(abs(flipped) > min_volume, 1);
    
    % Trimming the data
    y = audio_data(first_index:last_index);
end

function mfcc_coeff = get_mfcc(data, samp_freq, fft_length, overlap_length, mel_filt_num, show_plot)
    % Given an input .wav file, this function returns an array of mel
    % frequency cepstrum coefficients for the specified fft length and the
    % overlap length. The mel_filt_num also specifies how many mel
    % frequency filters to use
    
    % Do frame blocking
    block_matrix = frame_block(data, fft_length, overlap_length);
    
    % Apply windowing to our framed signal
    windowed_block = matrix_hamm(block_matrix);
    
    % Find the fft of the windowed block
    block_fft = zeros(length(windowed_block(:, 1)), 1+floor(fft_length/2));

    % We only want to keep the positive portion of the fft
    for n=1:length(windowed_block(:, 1))
        % Computing the fft across the row
        row_fft = fft(windowed_block(n, :), fft_length, 2);
        block_fft(n, :) = row_fft(1:1+floor(fft_length/2));
    end

    % Calculate the power of the fft
    power_fft = abs(block_fft).^2;
    
    % Get the mel filters
    mel_filters = mel_bank(samp_freq, fft_length, mel_filt_num, true);
    
    % Matrix multiply the mel filters and our power fft
    y_filtered = mel_filters * power_fft';

    % We can ommit the first row since that only represents the DC component
    mfcc_coeff = dct(y_filtered);
    mfcc_coeff = mfcc_coeff(2:end, :);
    
    if show_plot
        
        % Plotting audio vs time
        t = (0:1/samp_freq:(length(data) - 1)/samp_freq);
        figure(1)
        plot(t,data)
        title("Audio vs time")
        xlabel("Time (s)")
        ylabel("Volume")
        
        % Look at the effect of the windowing
        figure(2)
        subplot(211)
        plot(linspace(0, fft_length-1, fft_length), block_matrix(floor(length(block_matrix(:, 1))/2), :))
        title("Block 52")
        subplot(212)
        plot(linspace(0, fft_length-1, fft_length), windowed_block(floor(length(block_matrix(:, 1))/2), :))
        title(strcat("Block", string(floor(length(block_matrix(:, 1))/2)), "after Windowing"))
        
        % Look at stft for our audio signal
        figure(3)
        stft(data, samp_freq, "Window", hamming(fft_length), "OverlapLength", overlap_length, "FFTLength", fft_length, "FrequencyRange", "Onesided")
        title(strcat("STFT"))
        
        % Plotting our mel filter bank
        figure(4)
        hold on
        for n=1:mel_filt_num
            plot(linspace(0, floor(fft_length/2), floor(fft_length/2 + 1)), mel_filters(n,:))
        end
        hold off
        title("Mel Filters")
        
    end
end


function m = mel_bank(samp_freq, fft_size, mel_filt_num, norm)
    % Returns mel spaced filter bank in frequency domain
    fmin = 0;
    fmax = samp_freq/2;
    
    % Find filter points and freqs arrays
    [filter_points, freqs] = mel_points(fmin, fmax, mel_filt_num, fft_size, samp_freq);
    
    % Initialize our filters matrix
    m = zeros(mel_filt_num, floor(fft_size/2 + 1));
    
    for n=1:mel_filt_num
        p1 = filter_points(n);
        p2 = filter_points(n+1);
        p3 = filter_points(n+2);
 
        m(n, p1:p2) = linspace(0, 1, p2 - p1 + 1);
        m(n, p2:p3) = linspace(1, 0, p3 - p2 + 1);
       
    end
    
    if norm
        norm_factor = 2 ./ (freqs(3:mel_filt_num+2) - freqs(1:mel_filt_num));
        m = m.*norm_factor';
    end

end

function [points, freqs] = mel_points(freq_min, freq_max, mel_filt_num, fft_size, samp_freq)
    % Transform freq_min and freq_max to mel domain
    mel_min = 2595*log10(1 + freq_min/700);
    mel_max = 2595*log10(1 + freq_max/700);
    
    % Find linearly spaced points between min and max
    mels = linspace(mel_min, mel_max, mel_filt_num+2);
    
    % Convert mel points back to frequency domain
    freqs = 700*(10.^(mels/2595) - 1);
    
    % Find filter points
    points = floor((fft_size + 1)/ samp_freq * freqs) + 1;
   
end

function B = matrix_hamm(block_matrix)
    % This function takes a 2D block matrix as the input
    % Where each row of the matrix is of size N
    % Each row is then multiplied by a hamming window
    
    [rows, N] = size(block_matrix);
    n_array = linspace(0, N-1, N);
    hamm = 0.54 -  0.46*cos(2*pi*n_array/(N-1));
    
    % Debuging
    %fprintf("Hamm length: %d\n", length(hamm));
    %fprintf("Block row length: %d\n", length(block_matrix(1, :)));
    
    % Initialize our output vector
    B = zeros(size(block_matrix));
    
    for k=1:rows
        B(k, :) = block_matrix(k, :).*hamm;
    end
    
end

function B = frame_block(data, N, M)
    % This function takes a 1d array and reshapes it into
    % blocks of length N with M number of points overlapping
    % between each block
    

    num_rows = length((1:N-M:length(data)-N));
    B = zeros(num_rows, N);
    current_row = 1;
    
    for n=1:N-M:length(data)-N
        block = data(n:n+N-1);
        B(current_row, :) = block;
        
        current_row = current_row + 1;
    end
end