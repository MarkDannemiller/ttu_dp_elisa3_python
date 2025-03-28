function [u,e_x,e_v,A2,B2,K2,Z2,GAMMA2] = BackSteppingCF_Veh2(X,x_veh1,v_veh1,A1,B1,K1,ZX1)
 
% m = 5760;%mass

m = 1000;%mass
Af = 7.5;%Effective frontal area
rho = 1.206;%air mass density
Cd = 0.51;%aero drag force coefficient
Cr = 0.0041*9.8;%rolling resistance coefficient
Tau = 0.02;%powertrain response time lag
L = 5; %vehicle length

fi = -(X(3)+Af*rho*Cd*X(2)^2/(2*m)+Cr)/Tau-Af*rho*Cd*X(2)*X(3)/m;
gi = 1/(m*Tau);

h = 1; %desired time gap

e_x = x_veh1-X(1)-L-h*X(2); %gap error 
e_v = v_veh1-X(2); %speed error
a = X(3);

delta_0 = 4.5;

k_2_1 = 0.1;
k_2_2 = 0.1;
k_2_3 = 0.1;
eps_2_2 = 0.1;
eps_2_3 = 0.01;

P2 = k_2_2+h*abs(K1'*B1)*delta_0/(2*eps_2_2);
Q2 = k_2_3+abs((k_2_1+P2)*h*K1'*B1-(K1'-h*K1'*A1)*B1)*delta_0/(2*eps_2_3);

z4 = e_x-h*e_v;
e_v_bar = h*K1'*ZX1-k_2_1*z4;
z5 = e_v-e_v_bar;
% a_bar = z4+k_2_1*e_v+(K1'-h*K1'*A1-k_2_1*h*K1')*ZX1+(k_2_2+abs(h*K1'*B1)*delta_0/(2*eps_2_2))*z5;
a_bar = (1-k_2_1^2)*z4+(k_2_1+P2)*z5+(K1'-h*K1'*A1)*ZX1;
z6 = a-a_bar;

% C1 = 1+k_2_1*(k_2_2+abs(h*K1'*B1)*delta_0/(2*eps_2_2));
% C2 = -(k_2_1+k_2_2+abs(h*K1'*B1)*delta_0/(2*eps_2_2));
% C3 = (-h*K1'-k_2_1*(k_2_2+abs(h*K1'*B1)*delta_0)/(2*eps_2_2)*h*K1'+(k_2_1+k_2_2+(abs(h*K1'*B1)*delta_0)/(2*eps_2_2))*K1'+(K1'-h*K1'*A1-k_2_1*h*K1')*A1-(k_2_2+(abs(h*K1'*B1)*delta_0)/(2*eps_2_2))*h*K1'*A1)*ZX1;
% C4 = 1;
% C5 = -k_2_3 - abs((-K1'+h*K1'*A1+(k_2_1+k_2_2+(abs(h*K1'*B1)*delta_0)/(2*eps_2_2))*h*K1')*B1)*delta_0/(2*eps_2_3);
% u = (-fi+C1*e_v+C2*a+C3+C4*z5+C5*z6)/gi;


C1 = -((2-k_2_1^2)*k_2_1+P2);
C2 = (2-k_2_1^2-(k_2_1+P2)*P2);
C3 = -(k_2_1+P2+Q2);
u = (-fi+C1*z4+C2*z5+C3*z6+(K1'-h*K1'*A1)*A1*ZX1)/gi;

a11 = -k_2_1;
a12 = 1;
a13 = 0;
a21 = -1;
a22 = -P2;
a23 = -1;
a31 = 0;
a32 = 1;
a33 = -Q2;
b1 = 0;
b2 = -h*K1'*B1;
b3 = (k_2_1+P2)*h*K1'*B1-(K1'-h*K1'*A1)*B1;


A2 = [a11,a12,a13;a21,a22,a23;a31,a32,a33];
B2 = [b1;b2;b3];
K2 = [1-k_2_1^2;k_2_1+P2;1];
Z2 = [z4;z5;z6];
[GAMMA2,kappa2] = ComputeGamma2(delta_0,h,k_2_1,k_2_2,k_2_3,0,eps_2_2,eps_2_3,0,K1,B1,A1);
