function [H,P]=AALM(DataSet,options)

%% Approximated augmented Lagrangian method for solving the following problem
%
%             min |H-HZ|_F+ eta |Z|_F + lambda tr(P^T(Sb-nu St)P)
%             s.t. P^TX=H, P^TP=I，diag(Z) = 0.
%
%% --------------------------------parameter--------------------------------
d = options.reduce_dimension;
rho = options.rho;
epso = options.epso;
tau = options.tau;
mu = options.mu;
decrease = options.decrease;
max_iter = options.max_iter;
inn_max_iter = options.inn_max_iter;
eta = 0.01; lambda = 1; 
error = options.error;



%% ------------------------------- initialization-------------------------------------------------
X = DataSet.X; y = DataSet.y;
[r,n] = size(X);
[X, ~] = data_normalization(X, y, 'std');   %data normlization
for i=1:n
    X(:,i)=X(:,i)/norm(X(:,i));
end

[M,~,~] = gene_cmatrix(X',y); % generate matrix :Sw-nu*Sb

[P0] = LDA(X', y',d); 
[U,~,V] = svd(P0,'econ');P0 = U*V';
J0 = zeros(r,d);H0 = zeros(d,n);Z0 = eye(n);
Y10 = zeros(d,n); Y20 = zeros(r,d);
R0 = norm(P0'*X-H0,'fro')^2; 


 k=1;
while(1)
 
    [H,Z,P,J] = PAM(X,M,y,H0,Z0,P0,J0,Y10,Y20,rho,epso,lambda,eta,inn_max_iter);
    
    Y1 = Y10+rho*(P'*X-H);
    Y2 = Y20+rho*(P-J);
    
    R = norm(P'*X-H,'fro')^2;
    if(R>tau*R0)
        rho = rho*mu;
    end
    R0 = R;
    
    if(R<error || k>max_iter)
        break;
    end
    
    
    k=k+1;
    epso = epso*decrease;
    H0 = H; Z0 = Z; P0 = P; J0 = J;
    Y10 = Y1; Y20 = Y2;

end


