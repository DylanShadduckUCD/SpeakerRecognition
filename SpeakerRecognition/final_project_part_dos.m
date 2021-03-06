% Final Project Second Attemp (for tidiness)
% Dylan Shadduck
% EEC 201

clear
clc
%% Getting all the training files
% Windows
folder = "C:\Users\Dylan\Documents\Winter 2021\EEC 201\Train\";
% Mac
%folder = "/Users/Dylan/Desktop/Train/";

% Getting a list of all the .wav files in the folder
filepattern = fullfile(folder, "*.wav");
theFiles = dir(filepattern);

% Initialize the centroid array
centroids = 4;
K = 20;
user_centroids = zeros(length(theFiles), centroids, K-1);

figure(1)
for k=1:length(theFiles)
    
    baseFileName = theFiles(k).name;
    % fprintf("File: %s\n", baseFileName)
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    
    [y, fs] = audioread(fullFileName);
    
    % Normalizing the data
    y = y./max(abs(y));
    
    if k == 11
        %sound(y, fs);
    end
    
    % Trimming the audio with normalized minimum volume of 0.05
    y = trim_silence(y, 0.05);
        
    % Plotting audio vs time
    t = (0:1/fs:(length(y) - 1)/fs);
    subplot(4,3,k)
    plot(t,y)
    title(strcat("Speaker ", string(k)))
    xlabel("Time (s)")
    ylabel("Volume")
    
    % Get the mfcc for the data
    N = 256;
    M = 100;
    K = 20;
    
    % Calculate the mfcc coefficients
    mfcc = get_mfcc(y, fs, N, M, K, false);
    
    % Find the clusters for centroids number of groups using the kmeans
    % algorithm
    [idx, C, sumD, D] = kmeans(mfcc', 4);
    
    % Append our centroids to our user centroid array
    user_centroids(k, :, :) = C;
    
end

% Could use variable kmeans clustering so that each speaker mfcc has a
% minimum threshold for acceptable euclidean distance error. 
% [idx, centroids, sumD, D] = kmeans(mfcc_1', 4);

% figure(2)
% hold on
% scatter(mfcc_1(7, :), mfcc_1(15, :))
% scatter(centroids(:, 7), centroids(:,15), [], "x")
% scatter(mfcc_2(7, :), mfcc_2(15, :))
% legend("User 1", "User1 centroids", "User 4")
% xlabel("MFCC 7")
% ylabel("MFCC 15")
% hold off


%% Functions

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