function main( inputFolder,outputFolder,categoryNbr )
noise_level = categoryNbr;
f_names = dir(inputFolder);
f_names = f_names(3:end,:);
% If "Error using imread>get_format_info (line 545)
%    Unable to determine the file format."
% occurs, switch 3 with 4 in f_names.

for i=1:length(f_names)

   gn = double(imread([inputFolder f_names(i).name]))/65535*255;
   %imshow(fid,[]);
   sigma = 13;%13;%25; %livello di rumore gaussiano (deviazione standard)
%gn    = obj + sigma*randn(size(obj));

%figure, imshow(obj,[]);title('Immagine Originale');
%figure, imshow(gn,[]);title('Immagine con rumore');

[n,m]  = size(gn);
grad1  = @(y)[diff(y,1,2), -y(:,end)];
grad2  = @(y)[diff(y); -y(end,:)];  %diff(X)=[X(2:n,:) - X(1:n-1,:)]
div    = @(x1,x2)([-x1(:,1), -diff(x1(:,1:m),1,2)] +...
        [-x2(1,:);-diff(x2(1:n,:));]);
x0 = gn; 
x0(x0 < 0) = 0;
x0(x0>255) = 255;
xmin = 0; xmax = 255; 
delta   = 0.1;
verbose = 1;
M       = 1;
NIT     = 200;
tol     = 1e-8;
%%%%% stima del parametro di regolarizzazione per i
dx1 = grad1(gn); dx2 = grad2(gn); 
densq = sqrt(dx1.^2 + dx2.^2+delta^2); 
TV = sum(densq(:))-n*m*delta;
rho = 0.7*TV/(sigma^2*n*m); %valore stimato del parametro di regolarizzazione
%%%%%%%%%%%%%%%%%%%% TVdelta (Total Variation smoothed)

switch noise_level
    case 0
        Npsf = 7;
    case 1
        Npsf = 7;
    case 2
        Npsf = 9;
    case 3
        Npsf = 9;
    case 4 
        Npsf = 15;
    case 5 
        Npsf = 23;
    case 6
        Npsf = 57;
    otherwise 
        Npsf  = 95;
        
end
        
P    = zeros(Npsf);
r    = floor(Npsf/2);
[X,Y] = meshgrid([-r:r]);
ij = find(X.^2+Y.^2<= r^2);
P(ij) = 1;
P = P/sum(P(:));
%figure, imagesc(P);axis square;

A = @(x)(conv2(x,P,'same'));
AT = @(x)(conv2(x,rot90(P,2),'same'));


tic
[xTV,TimeCostTV,PrimalSGP,alpha_vecTV,errTV] = ...
    SGP_TVdelta(A,AT,gn, rho, delta, xmin,xmax, grad1, grad2, div, NIT, tol, verbose, M, []);
toc
%figure, imshow(xTV,[]); title('Immagine ricostruita con deblurring (SGP)');
rec = xTV;
A = @(x)(x);
AT =@(x)(x);

outfilename = [outputFolder f_names(i).name(1:end-3) 'png'];  
%outimagename = [cartella_out  rad_nome num_file '_2.png'];
I = cat(3, uint8(rec),uint8(rec),uint8(rec));
imwrite(I, outfilename);
end

% main('./inputFolder/','./outputFolder/',2)

