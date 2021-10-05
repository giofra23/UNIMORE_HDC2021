function main_new( inputFolder,outputFolder,categoryNbr )
noise_level = categoryNbr;
f_names = dir(inputFolder);
f_names = f_names(3:end,:);
% If "Error using imread>get_format_info (line 545)
%    Unable to determine the file format."
% occurs, switch 3 with 4 in f_names.

addpath(genpath('Tools'));
seefig = 0;

for i=1:length(f_names)
fprintf('\n Image %g',i);
gn = double(imread([ inputFolder f_names(i).name]))/65535*255;
%obj = double(imread([  rad_nome end_filename]))/65535*255;
if seefig, figure(1), imshow(gn,[]);title('Data');end
gn0 = gn;

NIT     = 50;

switch noise_level 
    case 0
        Npsf = 3; ry = 3; taperdim = 60; tapersigma = 20; sigma = 10;
        scale = 0.5;
    case 1
        Npsf = 7; ry = 7;taperdim = 60; tapersigma = 20; sigma = 10;
        scale = 0.5;
    case 2
        Npsf = 13; ry = 13;taperdim = 60; tapersigma = 20; sigma = 8;
        scale = 0.5;
    case 3
        Npsf = 17; ry = 17; taperdim = 60; tapersigma = 20; sigma = 5;
        scale = 0.5;
    case 4
        Npsf = 9; ry   = 9;taperdim = 60; tapersigma = 20; sigma = 0.8;
        scale = 0.25;
    case 5
        Npsf = 17; ry   = 17;taperdim = 60; tapersigma = 20; sigma = 0.3;
        scale = 0.25;
    case 6
        Npsf = 23; ry   = 23;taperdim = 60; tapersigma = 20; sigma = 0.3;
        scale = 0.25;
    case 7
        Npsf = 27; ry   = 27;taperdim = 60; tapersigma = 20; sigma = 0.3;
        scale = 0.25;
    case 8
        Npsf = 31; ry   = 31;taperdim = 60; tapersigma = 20; sigma = 0.3;
        scale = 0.25;
    case 9
        Npsf = 19; ry   = 19;taperdim = 60; tapersigma = 20; sigma = 0.3;
        scale = 0.125;
    otherwise
        Npsf = 21; ry   = 21;taperdim = 50; tapersigma = 30; sigma = 0.3;
        scale = 0.125;
end

gn = imresize(gn,scale); 
if seefig,figure(2), imshow(gn,[]);title(['Resized data (scale:' num2str(scale) ')'] );end
%obj = imresize(obj,scale); 
%if vedifig,figure(3), imshow(obj,[]);title(['Resized obj (scale:' num2str(scale) ')'] );end
if seefig,figure(4), imshow(log10(abs(fftshift(fft2(fftshift(gn))))),[]); title('FT of data');end
gn  = edgetaper(gn,fspecial('gaussian',taperdim,tapersigma));

[n,m]  = size(gn);
boundary = 'reflective';%'reflective';
if strcmp(boundary,'reflective')
    %Reflective conditions
    z      = zeros(n,1); zt = zeros(1,m);
    grad1  = @(y)[diff(y,1,2), z];
    grad2  = @(y)[diff(y); zt];  %diff(X)=[X(2:n,:) - X(1:n-1,:)]
    div    = @(x1,x2)([-x1(:,1), -diff(x1(:,1:m-1),1,2), x1(:,m-1)] +...
        [-x2(1,:);-diff(x2(1:n-1,:)); x2(n-1,:)]);
else
    %Dirichlet (zero) conditions
    grad1  = @(y)[diff(y,1,2), -y(:,end)];
    grad2  = @(y)[diff(y); -y(end,:)];  %diff(X)=[X(2:n,:) - X(1:n-1,:)]
    div    = @(x1,x2)([-x1(:,1), -diff(x1(:,1:m),1,2)] +...
        [-x2(1,:);-diff(x2(1:n,:));]);
end

x0 = gn; 
x0(x0 < 0) = 0;
x0(x0>255) = 255;
xmin = 0; xmax = 255; 
delta   = 0.001;
verbose = 0;
M       = 1;
tol     = 1e-8;

%%%%% stima del parametro di regolarizzazione per i
dx1 = grad1(gn); dx2 = grad2(gn); 
densq = sqrt(dx1.^2 + dx2.^2+delta^2); 
TV = sum(densq(:))-n*m*delta;
rho = 0.7*TV/(sigma^2*n*m); %valore stimato del parametro di regolarizzazione


%%%%%%%%%%%%%%%%%%%% TVdelta (Total Variation smoothed)
%Npsf = 3;
%ry   = 3;

npad = (Npsf-ry)/2;

r    = floor(Npsf/2);
% [X,Y] = meshgrid([-r:r]);
% ij = find(X.^2+Y.^2< r^2);
% P(ij) = 1;
% P = P/sum(P(:));

P = fspecial('disk',r);
P = imresize(P,[ry,Npsf]);
P = padarray(P,npad)/sum(P(:));
if seefig,figure(5), imagesc(P);axis square; title('PSF');end

if strcmp(boundary,'reflective') 
    A =  @(x)(reshape(mex_convolution(x(:),P,n,m),n,m));%@(x)(conv2(x,P,'same'));
    AT = @(x)(reshape(mex_convolution_transpose(x(:),P,n,m),n,m));%@(x)(conv2(x,rot90(P,2),'same'));
else
    A =  @(x)(conv2(x,P,'same'));
    AT = @(x)(conv2(x,rot90(P,2),'same'));
end

nF = [];
tic
[xTV,TimeCostTV,PrimalSGP,alpha_vecTV,errTV] = ...
    SGP_TVdelta(A,AT,gn, rho, delta, xmin,xmax, grad1, grad2, div, NIT, tol, verbose, M, [],nF);
toc
if seefig,figure(6), imshow(xTV,[]); title('Immagine ricostruita con deblurring (SGP)');end
rec = xTV;

rec = imresize(rec,1/scale);
if seefig,figure(7), imshow(rec,[]); title('final output');end

outfilename = [outputFolder f_names(i).name(1:end-3) 'png'];  
%outimagename = [cartella_out  rad_nome num_file '_2.png'];
I = cat(3, uint8(rec),uint8(rec),uint8(rec));
imwrite(I, outfilename);
end

end