function [K1,B1,A1] = ComputeClosedDynamicsVeh1(delta_0,h,k_1_1,k_1_2,k_1_3,e_1_1,e_1_2,e_1_3)

p_1 = k_1_1+h*delta_0/(2*e_1_1);
q_1 = k_1_2+abs(1-k_1_1*h-(h*h*delta_0)/(2*e_1_1))*delta_0/(2*e_1_2);

a11 = -p_1;
a12 = 1;
a13 = 0;
% a21 = -1+q_1*k_1_1+q_1*h*delta_0/(2*e_1_1)-(k_1_1+h*delta_0/(2*e_1_1))*(p_1+q_1+k_1_1+h*delta_0/(2*e_1_1));
a21 = -1;
% a22 = p_1+q_1+k_1_1+h*delta_0/(2*e_1_1);
a22 = -q_1;
a23 = -1;
a31 = 0;
a32 = 1;
% a33 = -k_1_3-abs(h-q_1*k_1_1*h-q_1*h^2*delta_0/(2*e_1_1)-p_1+q_1)*delta_0/(2*e_1_1);
a33 = -k_1_3-abs(h+q_1*k_1_1*h+(q_1*h^2*delta_0)/(2*e_1_1)-p_1-q_1)*delta_0/(2*e_1_3);
b1 = -h;
% b2 = 1-k_1_1-h^2*delta_0/(2*e_1_1);
b2 = 1-k_1_1*h-(h*h*delta_0)/(2*e_1_1);
% b3 = h-q_1*k_1_1*h-q_1*h^2*delta_0/(2*e_1_1)-p_1+q_1;
b3 = h+q_1*k_1_1*h+q_1*h*h*delta_0/(2*e_1_1)-p_1-q_1;

k1 = 1-p_1*p_1;
k2 = p_1+q_1;
k3 = 1;

A1 = [a11,a12,a13;a21,a22,a23;a31,a32,a33];
B1 = [b1;b2;b3];
K1 = [k1;k2;k3];