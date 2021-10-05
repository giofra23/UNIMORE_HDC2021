function [x,TimeCost,Primal,alpha_vec,err,varargout] = ...
    SGP_TVdelta(A, AT, gn, rho, delta, xmin,xmax, grad1, grad2, div, NIT, tol, verbose, M, obj,nF)

nobj = length(obj);
err = [];
for i = 1:nobj
    normobj(i) = norm(obj{i}(:));
    err{i}     = zeros(NIT+1,1); %arrays to store errors per iteration
end
tolf = 0; 

Primal = zeros(NIT+1,1); 
phi_xy = zeros(NIT+1,1); 
TimeCost = zeros(NIT+1,1);
kkt_vec = zeros(NIT,1);

[n,m] = size(gn);


gamma = 1e-4;                   % parametri linesearch
beta = 0.4;
%M = 1;                          % algoritmo monotono(M=1) /nonmonotono(M>1)
Fold = -1e30 * ones(M, 1);
alpha_min = 1e-5;               %  parametri steplength
alpha_max = 1e3;
Malpha = 3;                     % memoria su alfaBB
tau = 0.5;
Valpha = 1e30 * ones(Malpha,1);
iter = 1;
soglia_X_low = 1e-10;
soglia_X_upp = 1e10; % soglia sulla matrice di scaling
alpha = 1.3;                    % alpha iniziale
d2 = delta^2;
% initial point
x = min(max(gn,xmin),xmax);

% gradiente e funzione obiettivo nella x corrente
dx1     = grad1(x);
dx2     = grad2(x);
densq   = sqrt(dx1.^2 + dx2.^2 + d2);
TV      = sum(densq(:));
g_TV    = div(dx1./densq,dx2./densq);
V_R     = x.*(2./densq + 1./[densq(n,:); densq(1:n-1,:)] + 1./[densq(:,m) densq(:,1:m-1)]);
Ax      = A(x);
fv = 0.5*rho*norm(Ax(:)-gn(:))^2 + TV; %% funzione obiettivo nella x corrente
ATAx = AT(Ax);
ATg  = AT(gn);
g_KL = ATAx-ATg; 
g = rho*g_KL + g_TV; %% gradiente

Primal(1) = fv;
alpha_vec(1) = alpha;
for i=1:nobj
    err{i}(1) = norm(obj{i}(:)-x(:))/normobj(i);
end

if verbose
    fprintf('\nInitial: Primal=%8.3e', Primal(1));
    for i=1:nobj
        fprintf(' err{%g} %g', i, err{i}(1));
    end
end

TimeCost(1)=0;
t0 = cputime;                %Start CPU clock

%% matrice di scaling
X =  x./(rho*ATAx + V_R + sqrt(eps)); 
X(X<soglia_X_low) = soglia_X_low;
X(X>soglia_X_upp) = soglia_X_upp;
D = 1./X; 


for itr = 1:NIT
    if ~isempty(nF)
        figure(nF); imshow(x,[]);title(['iter n.' num2str(itr)]); pause(0.1);
    end
    Valpha(1:Malpha-1) = Valpha(2:Malpha);

    Fold(1:M-1) = Fold(2:M);
    Fold(M) = fv;

    % Step 2.1
    y = x - alpha*X.*g;
    %y = x - alpha*g;

    % proiezione
    y = min(max(y,xmin),xmax);
    
    d = y - x;
    Ad = A(d);
    gd = sum(dot(d,g));
    fold = fv;
    lam = 1;
    %d_tf = d;
    %step 2.2  linesearch
    fcontinue = 1;
    
    fr = max(Fold);

    while fcontinue
        xplus = x   + lam*d;
        Axplus = Ax + lam*Ad;
        %Update the objective function value
        dx1     = grad1(xplus);
        dx2     = grad2(xplus);
        densq   = sqrt(dx1.^2 + dx2.^2 + d2);
        TV      = sum(densq(:));
        
        fv = 0.5*rho*norm(Axplus(:)-gn(:))^2 + TV;   %% funzione obiettivo calcolata in xplus
        
        % Step 2.3
        if ( fv <= fr + gamma * lam * gd | lam<1e-12)

            x       = xplus; clear xplus;
            sk      = lam*d; % differenza tra le iterate
            Ax      = Axplus;
            g_TV    = div(dx1./densq,dx2./densq);
            %V_R     = x.*(2./densq + 1./[densq(n,:);densq(1:n-1,:)] + 1./[densq(:,m) densq(:,1:m-1)]);
            ATAx    = AT(Ax);
            g_KL    = ATAx - ATg;
            gtemp   = rho*g_KL + g_TV;
            yk      = gtemp - g; % differenza del nuovo e vecchio gradiente
            g = gtemp; clear gtemp;
            fcontinue = 0;
        else
            lam = lam * beta;
        end
    end
     %fprintf('\nsk %g yk %g ',norm(sk(:)), norm(yk(:)));
    % Step 3
    %% matrice di scaling
    V_R     = x.*(2./densq + 1./[densq(n,:); densq(1:n-1,:)] + 1./[densq(:,m) densq(:,1:m-1)]);
    X       = x./(rho*ATAx + V_R + sqrt(eps));
    X(X<soglia_X_low) = soglia_X_low;
    X(X>soglia_X_upp) = soglia_X_upp;
    D = 1./X;
    
    
    sk2 = sk.*D; yk2 = yk.*X;
    %sk2 = sk; yk2 = yk;
 
    bk = sum(dot(sk2,yk));  ck = sum(dot(yk2,sk));
    %fprintf('\n      sk2 %g yk2 %g bk %g ck %g',norm(sk2(:)), norm(yk2(:)), bk,ck);
    if (bk <= 0)
        alpha1 = alpha_max;
    else
        alpha1BB = sum(dot(sk2,sk2))/bk;
        alpha1 = min(alpha_max, max(alpha_min, alpha1BB));
    end
    if (ck <= 0)
        alpha2 = alpha_max;
    else
        alpha2BB = ck/sum(dot(yk2,yk2));
        alpha2 = min(alpha_max, max(alpha_min, alpha2BB));
    end

    Valpha(Malpha) = alpha2;

    if (alpha2/alpha1 < tau)
        alpha = min(Valpha);
        tau = tau*0.9;
    else
        alpha = alpha1;
        tau = tau*1.1;
    end
    Primal(itr + 1)   = fv;
    TimeCost(itr + 1) = cputime-t0;
    alpha_vec(itr + 1) = alpha;
    
    for i=1:nobj
        err{i}(itr + 1) = norm(obj{i}(:)-x(:))/normobj(i);
    end
    
    if verbose
        fprintf('\n%4d): f(x)=%g  alpha %g norm d %g',  itr, ...
            Primal(itr+1), alpha, norm(d(:)) );
        for i=1:nobj
            fprintf(' err{%g} %g', i, err{i}(itr + 1));
        end
    end
    kkt_vec(itr) = norm(sk(:))/norm(x(:));
    if tolf & fv <= tolf
        break;
    end
    if (kkt_vec(itr) < tol)
        break;
    end
end
%end of the main loop
Primal(itr+2:end) = [];
for i = 1:nobj
    err{i}(itr + 2:end) = [];
end
alpha_vec(itr+2:end) = [];
TimeCost(itr+2:end) = [];
if nargout == 6
    varargout{1} = kkt_vec(1:itr);
end
if verbose
    fprintf('\n');
end


