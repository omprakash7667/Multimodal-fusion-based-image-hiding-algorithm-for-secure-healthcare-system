%                      
%
% ----------------- Code for "Multimodal fusion-based image hiding algorithm for secure healthcare system" in IEEE Intelligent System -----------------
%
% O. P. Singh, A. K. Singh, and H. Zhou , "Multimodal fusion-based image hiding algorithm for secure healthcare system," 2022, IEEE Intelligent System,
%
for any query, you can contact me at omprakash7667@gmail.com
% -------------------------------------------------------------------------

clc; 
clear all;
close all;
SF=0.1;
img=imread('House.bmp');
img=imresize(img,[512 512]);
figure(1),imshow(img,[]);title('Original RGB Image');
  
Watermark=imread('fused.png');
Watermark=double(Watermark);
figure(2), imshow(real(Watermark),[]);title('Watermark Image');

%%%%%%%%%PCA PART
img1=img(:,:,1);
[m, n]=size(img1);
img2=img(:,:,2);
img3=img(:,:,3);

%to get elements along rows 
temp1=reshape(img1',m*n,1);
temp2=reshape(img2',m*n,1);
temp3=reshape(img3',m*n,1);

I=[temp1 temp2 temp3]; 

%to get mean
m1=mean(I,2);

%subtract mean

for i=1:3
    I1(:,i)=(double(I(:,i))-m1);
end

%Find the covariance matrix and eigen vectors
a1=double(I1);
a=a1';
covv =1/(m-1)*(a*a');

[eigenvec, eigenvalue]=eig(covv);
eigenvalue1 = diag(eigenvalue);
[egn,index]=sort(-1*eigenvalue1);
eigenvalue1=eigenvalue1(index);
eigenvec1=eigenvec(:,index);

pcaoutput=a1*eigenvec1;

ima=reshape(pcaoutput(:,3)',m,n);  % taking the 3rd component

ima=ima';
imshow(uint8(ima));
Host_image=ima; %host image = 3rd component after pca
imwrite(uint8(Host_image),'PCA_host.jpg');
[Hrow,Hcol]=size(Host_image);
 
%%%%%%%% PCA END


%%%%%%%%%%  FRFT

angles=[0.7 0.8];
Transformed=frft2(Host_image,angles);
    
%Watermark Embedding
    
 [u ,s, v]=svd(Transformed);
 [u1,s1,v1]=svd(Watermark);
 [x, y]=size(Watermark);
    for i=1:x
       for j=1:y
          s(i,j) =s(i,j) + SF * s1(i,j);
       end 
   end 
   
Wimg =u* s* v';  
IT=frft2((Wimg),-angles);
Watermarked=IT;
    
 %%%%%%Inverse PCA
    
    t1=reshape(Watermarked',Hrow*Hcol,1);
    t2=pcaoutput;
    t2(:,3)=real(t1);  %do the changes here 
    V_inv=inv(eigenvec1);
    original=t2*V_inv;
    for i=1:3
    I2(:,i)=(double(original(:,i))+m1);
    end
    I2=round(I2);
    img6=reshape(I2(:,1)',m,n);
img6=img6';
img7=reshape(I2(:,2)',m,n);
img7=img7';
img8=reshape(I2(:,3)',m,n);
img8=img8';
back_to_original_img = cat(3, img6, img7, img8);
figure(4), imshow(uint8(back_to_original_img)); title('RGB Watermarked Image'); 
psnr=psnr(uint8(img),uint8(back_to_original_img));
SSIM=ssim(uint8(img),uint8(back_to_original_img));
%%%%%%% temp code
Coef=corr2(real(Watermarked),real(Host_image));
disp('Host Image to Watermarked Image corelation');
disp(Coef);

%%%%%%%%

%%%%%%% WATERMARK EXTRACTION 
key1 = u1;   %SVD OF WATERMARK 
key2 = v1;
IT_new=frft2(IT,angles);    %FRFT of Watermarked Image


[ExtractedWatermark]=svdExt_New(Transformed,IT_new,key1,key2,SF);
figure(6),imshow(real(ExtractedWatermark),[]);title('Extracted Watermark');
 nc=nc(Watermark,ExtractedWatermark); 
disp('Detect With No Attack');
CoCoef(Host_image,Watermarked,ExtractedWatermark,Watermark);

%%%%%%%%%%%%%%%%%%%

Please cite this article
