f = 'black2.avi';

fb = ones(1024, 1024);

v = VideoWriter(f, 'Grayscale AVI');
v.FrameRate = 100;
open(v)

for i = 1:10
    writeVideo(v, fb);
end

close(v)
