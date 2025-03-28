function [GAMMA2,kappa2] = ComputeGamma2(delta_0,h,k_2_1,k_2_2,k_2_3,e_2_2,e_2_3,GAMMA1,K1,B1,A1)

P2 = k_2_2+h*delta_0*abs(K1'*B1)/(2*e_2_2);
C1 = abs(K1'*B1);
C2 = abs((K1'-h*K1'*A1)*B1-h*(k_2_1+P2)*K1'*B1);

GAMMA2 = GAMMA1+C1*h*delta_0*e_2_2/2+C2*delta_0*e_2_3/2;
[kappa2,~] = min([k_2_1,k_2_2,k_2_3]);