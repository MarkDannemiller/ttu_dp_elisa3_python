function [GAMMA1,kappa1] = ComputeGamma1(delta_0,h,k_1_1,k_1_2,k_1_3,e_1_1,e_1_2,e_1_3)


p_1 = k_1_1+h*delta_0/(2*e_1_1);
q_1 = k_1_2+abs(1-k_1_1*h-(h^2*delta_0)/(2*e_1_1))*delta_0/(2*e_1_2);
GAMMA1 = h*delta_0*e_1_1/2+abs(1-k_1_1*h-h^2*delta_0/(2*e_1_1))*delta_0*e_1_2/2+abs(h+q_1*k_1_1*h+q_1*h^2*delta_0/(2*e_1_1)-p_1-q_1)*delta_0*e_1_3/2;
[kappa1,~] = min([k_1_1,k_1_2,k_1_3]);