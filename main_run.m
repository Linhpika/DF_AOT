close all; clc;
im_rgb = im2double(imread('./DRIVE/Training/images/40_training.tif'));

im_mask = im_rgb(:,:,3) > (20/255); % Extract Blue channel
im_mask = double(imerode(im_mask, strel('disk',3)));

figure
subplot(2,2,1),imshow(im_rgb),title('General image');
subplot(2,2,2),imshow(im_mask),title('Mask after erosion');

im_Blue = im_rgb(:,:,3);
subplot(2,2,3),imshow(im_Blue),title('Blue Channel')

% CLAHE
im_enh = adapthisteq(im_Blue,'numTiles',[8 8],'nBins',128);
subplot(2,2,4),imshow(im_enh),title('CLAHE enhancement')
saveas(gcf,'./Result/Blue_channel_Extraction_Process/Training/20/process.png');
% replace_black_rin
[im_enh1, mean_val] = replace_black_ring(im_enh,im_mask);

% create negativ of im_enh1
im_gray = imcomplement(im_enh1);  
figure
subplot(2,2,1),imshow(im_gray),title('Gray Image')

% top-hat transform
se = strel('disk',10);
im_top = imtophat(im_gray,se);
subplot(2,2,2),imshow(im_top),title('After top-hat')

% OTSU
level = graythresh(im_top);
im_thre = imbinarize(im_top,level) & im_mask;
subplot(2,2,3), imshow(im_thre),title('Otsu threhsolding')

% Remove small pixels
im_rmpix = bwareaopen(im_thre,100,8);
subplot(2,2,4), imshow(im_rmpix),title('Remove small pixels')
saveas(gcf,'./Result/Blue_channel_Extraction_Process/Training/20/extraction.png');

[im_sel] = vessel_point_selected(im_gray,im_rmpix,mean_val);
figure
subplot(1,3,1),imshow(im_sel),title('Thick vessel extraction')

im_thin_vess = MatchFilterWithGaussDerivative(im_enh, 1, 4, 12, im_mask, 2.3, 30);
subplot(1,3,2), imshow(im_thin_vess),title('Thin vessel extraction')

[im_final] = combine_thin_vessel(im_thin_vess,im_sel);
subplot(1,3,3),imshow(im_final),title('Final image')
saveas(gcf,'./Result/Blue_channel_Extraction_Process/Training/20/post process.png');
g_truth = imread('./DRIVE/Training/1st_manual/40_manual1.gif');

[Se, Sp, Acc, FPPI] = performance_measure(im_final,g_truth);
%fprintf('Sensitivity = %.2f\n', Se);
%fprintf('Specificiy = %.2f\n', Sp);
%fprintf('Accuracy = %.2f\n', Acc);
%fprintf('FPPI = %.2f\n', FPPI);
fid = fopen('./Result/Blue_channel_Extraction_Process/Training/20/performance_measures.txt', 'w');
fprintf(fid, 'Sensitivity = %.2f\n', Se);
fprintf(fid, 'Specificity = %.2f\n', Sp);
fprintf(fid, 'Accuracy = %.2f\n', Acc);
fprintf(fid, 'FPPI = %.2f\n', FPPI);
fclose(fid);


g_truth = imbinarize(g_truth);
dice = 2*sum(sum((im_final) .* g_truth))/(sum(sum(im_final))+ sum(sum(g_truth)));

mixed = zeros(size(im_final, 1), size(im_final, 2), 3);

for i = 1 : size(im_final, 1)
    for j = 1 : size(im_final, 2)
        if im_final(i, j) == g_truth(i, j) && im_final(i, j) == 1
            mixed(i, j, 1) = 0;
            mixed(i, j, 2) = 1;
            mixed(i, j, 3) = 0;
        elseif im_final(i, j) == g_truth(i, j) && im_final(i, j) == 0
            mixed(i, j, 1) = 0;
            mixed(i, j, 2) = 0;
            mixed(i, j, 3) = 0;
        elseif im_final(i, j) ~= g_truth(i, j) && im_final(i, j) == 0
            mixed(i, j, 1) = 1;
            mixed(i, j, 2) = 0;
            mixed(i, j, 3) = 1;
        elseif im_final(i, j) ~= g_truth(i, j) && im_final(i, j) == 1
            mixed(i, j, 1) = 1;
            mixed(i, j, 2) = 0;
            mixed(i, j, 3) = 0;
        end
    end
end
figure, subplot(1,1,1),imshow(mixed, []),title('Mixed')
saveas(gcf,'./Result/Blue_channel_Extraction_Process/Training/20/evaluate.png');
