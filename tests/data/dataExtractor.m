fileName = 'small.tif';

a = double(imread(fileName, 1));
b = double(imread(fileName, 2));
sz = length(a); % assume square frames

csvwrite('diff_matlab_2-1.csv', b-a)

pa = fftTrans(a, 'fft_matlab_f1.csv');
pb = fftTrans(b, 'fft_matlab_f2.csv');
pc = fftTrans(b-a, 'fft_matlab_f(2-1).csv');

% Find FFT on each frame before subtraction
pd = fft2(b)-fft2(a);
pd = abs(pd).^2/sz^2;
pd = fftshift(pd);

% Write to CSV with extra precision
dlmwrite('fft_matlab_f2-f1.csv', pd, 'precision', 9)
dlmwrite('fft_matlab_f(2-1).csv', pc, 'precision', 9)

% fft2 is linear; normalisation is not.
% csvwrite('fft_matlab_f2-f1_bad.jpg', pb-pa)

% F-then-S vs S-then-F
disp(['Biggest difference: ', num2str(max(max(abs(pd-pc))))])

function [fft] = fftTrans(A, n)
    sz = length(A);

    fft = fft2(A);
    fft = abs(fft).^2/sz^2;
    fft = fftshift(fft);

    csvwrite(n, fft)
end
