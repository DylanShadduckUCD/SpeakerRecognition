% Dylan Shadduck
% EEC 201
% Final Project - Speaker Recognition

clear
clc

%% Reading in the speaker data

% Folder for all training data
folder = "C:\Users\Dylan\Documents\Winter 2021\EEC 201\Train\";

% Reading in the data for each user individually and normalizing volume
file_1 = strcat(folder,"s",string(1),".wav");
[y1, fs] = audioread(file_1);
y1 = y1./max(y1);

% User 2
file_2 = strcat(folder,"s",string(2),".wav");
[y2, fs] = audioread(file_2);
y2 = y2./max(y2);

% User 3
file_3 = strcat(folder,"s",string(3),".wav");
[y3, fs] = audioread(file_3);
y3 = y3./max(y3);

% User 4
file_4 = strcat(folder,"s",string(4),".wav");
[y4, fs] = audioread(file_4);
y4 = y4./max(y4);

% User 5
file_5 = strcat(folder,"s",string(5),".wav");
[y5, fs] = audioread(file_5);
y5 = y5./max(y5);

% User 6
file_6 = strcat(folder,"s",string(6),".wav");
[y6, fs] = audioread(file_6);
y6 = y6./max(y6);

% User 7
file_7 = strcat(folder,"s",string(6),".wav");
[y7, fs] = audioread(file_7);
y7 = y7./max(7);

% User 8
file_8 = strcat(folder,"s",string(6),".wav");
[y8, fs] = audioread(file_8);
y8 = y8./max(y8);

% User 9
file_9 = strcat(folder,"s",string(6),".wav");
[y9, fs] = audioread(file_9);
y9 = y9./max(y9);

% User 10
file_10 = strcat(folder,"s",string(6),".wav");
[y10, fs] = audioread(file_10);
y10 = y10./max(y10);

% User 11
file_11 = strcat(folder,"s",string(6),".wav");
[y11, fs] = audioread(file_11);
y11 = y11./max(y11);

% play back the audio from this file
%sound(y1, fs)

% Plot y1 vs time
t = (0:1/fs:(length(y1) - 1)/fs);

figure(1)
plot(t,y1)
title("Speaker 1")
xlabel("Time (s)")
ylabel("Volume")

% TODO: Look into trimming each audio file so that we don't have time when
% the speaker isn't saying anything

%% Frame Blocking

% N is the length of our frame
% M is the number of samples we are overlapping
N = 256;
M = 100;

block_matrix = frame_block(y1, N, M);

%% Windowing

% We want to apply a hamming window to each row of N samples in our
% blocking matrix

windowed_block = matrix_hamm(block_matrix);

% Look at the effect of the windowing
figure(2)
subplot(211)
plot(linspace(0, N-1, N), block_matrix(52, :))
title("Block 52")
subplot(212)
plot(linspace(0, N-1, N), windowed_block(52, :))
title("Block 52 after Windowing")

%% Performing FFT of our windowed block matrix
% The inputs to the fft function are the data, fft length, and axis
% The number 2 for the axis ensures that the fft is computed along the rows
% not the columns of our data

% Using the stft function for plotting
figure(3)
stft(y1, fs, "Window", hamming(N), "OverlapLength", M, "FFTLength", N, "FrequencyRange", "Onesided")
title("STFT User 1")

block_fft = zeros(length(windowed_block(:, 1)), 1+floor(N/2));

% We only want to keep the positive portion of the fft
for n=1:length(windowed_block(:, 1))
    % Computing the fft across the row
    row_fft = fft(windowed_block(n, :), N, 2);
    block_fft(n, :) = row_fft(1:1+floor(N/2));
end

% Calculate the power of the fft
power_fft = abs(block_fft).^2;


%% Mel Frequency Wrapping

% K is the number of mel filters we wish to use
K = 10;

mel_filters = mel_bank(fs, N, K, true);

figure(4)
hold on
for n=1:K
    plot(linspace(0, floor(N/2), floor(N/2 + 1)), mel_filters(n,:))
end
hold off
title("Mel Filters")

% Find dot product between filters and power of the fft
s1_filtered = mel_filters * power_fft';

%% Compute the Ceptum Spectrum Coeffients


%% Functions
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