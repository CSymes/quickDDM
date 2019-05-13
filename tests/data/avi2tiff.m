f = 'small.avi';
outputFileName = 'out.tif';

img = read(VideoReader(f));

for K=1:length(img(1, 1, :))
   imwrite(img(:, :, K), outputFileName, 'WriteMode', 'append');
end