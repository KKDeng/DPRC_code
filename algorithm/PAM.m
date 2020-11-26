function [H,Z,P,J] = PAM(X,M,y,H0,Z,P0,J0,Y10,Y20,rho,epso,lambda,eta,inn_max_iter)
%% Proximal alternating method for solving subproblem of Augmented Lagrangian method


[r,n] = size(X); XXT = X*X';
s1 = 1; s2 = 1; s4 = 1;
m=1;

while(1)
    %H子问题
    
    A = (eye(n)-Z)*(eye(n)-Z)'+(rho +s1)*eye(n); B = rho*(P0'*X+1/rho*Y10)+s1*H0;
    temp = A'\B'; H = temp';
    
    
    %P子问题
    
    A = (lambda*M+rho*XXT+(rho+s2)*eye(r)); B = rho*X*(H-1/rho*Y10)'+rho*(J0-1/rho*Y20)+ s2*P0;
    P = A\B;
    
    
    % Z-subproblem
    Z = zeros(n,n);
    for i=1:n
        index_i = (y==y(i)); index_i(i)=0;
        Z(index_i,i) = (H(:,index_i)'*H(:,index_i)+2*eta*eye(sum(index_i)))\(H(:,index_i)'*H(:,i));
        
    end
    
    
    %J-subproblem
    [U,~,V] = svd(( rho*P+Y20+s4*J0)/(rho+s4),'econ');
    J = U*V';
    
    
    
    if(norm(P-P0,'fro')<epso || m>inn_max_iter)
        break;
    end
    H0 = H; P0 = P; J0 = J;
    m=m+1;
end



