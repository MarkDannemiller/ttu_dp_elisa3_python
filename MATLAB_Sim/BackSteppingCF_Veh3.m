function [u,e_x,e_v,A3,B3,K3] = BackSteppingCF_Veh3(X,x_veh2,v_veh2,A1,B1,K1,ZX1,A2,B2,K2,ZX2)

m = 5760;%mass
Af = 7.5;%Effective frontal area
rho = 1.206;%air mass density
Cd = 0.51;%aero drag force coefficient
Cr = 0.0041*9.8;%rolling resistance coefficient
Tau = 0.02;%powertrain response time lag
L = 5; %vehicle length

fi = -(X(3)+Af*rho*Cd*X(2)^2/(2*m)+Cr)/Tau-Af*rho*Cd*X(2)*X(3)/m;
gi = 1/(m*Tau);

h = 1; %desired time gap

e_x = x_veh2-X(1)-L-h*X(2); %gap error 
e_v = v_veh2-X(2); %speed error
a = X(3);

delta_0 = 3;

k_3_1 = 30.0;
k_3_2 = 30.0;
k_3_3 = 30.0;
eps_3_2 = 0.01;
eps_3_3 = 0.01;
% 

M_3_2 = K2'-h*K2'*A2;
M_3_1 = K1'-h*K1'*A1-h*(K1'-h*K1'*A1)*A1;

Z7 = e_x-h*e_v;
e_v_bar = h*K2'*ZX2-k_3_1*Z7;
Z8 = e_v-e_v_bar;
P3 = k_3_2+abs(K2'*B2+(K1'-h*K1'*A1)*B1)*(h*delta_0/(2*eps_3_2));
a_bar = (1-k_3_1^2)*Z7 +(k_3_1+P3)*Z8 +M_3_1*ZX1 + M_3_2*ZX2;
Z9 = a-a_bar;



C1 = -((2-k_3_1^2)*k_3_1+P3);
C2 = 2-k_3_1^2-(k_3_1+P3)*P3;
C3 = -k_3_1-P3-k_3_3-abs(M_3_2*B2+M_3_1*B1-(k_3_1+P3)*(h*K2'*B2+h*(K1'-h*K1'*A1)*B1))*(delta_0/(2*eps_3_3));

u = (-fi+C1*Z7+C2*Z8+C3*Z9+M_3_2*A2*ZX2+M_3_1*A1*ZX1)/gi;
% 
% u=1000;

A3 = [];
B3 = [];
K3 = [];

