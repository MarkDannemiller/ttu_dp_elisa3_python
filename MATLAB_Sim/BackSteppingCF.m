function u = BackSteppingCF(X,t,Veh0_Time_Step,Veh0_Pos,Veh0_Spd)
 
P_t = interp1(Veh0_Time_Step,Veh0_Pos,t); %target vehicle position
V_t = interp1(Veh0_Time_Step,Veh0_Spd,t); %target vehicle speed

m = 5760;%mass
Af = 7.5;%Effective frontal area
rho = 1.206;%air mass density
Cd = 0.51;%aero drag force coefficient
Cr = 0.0041*9.8;%rolling resistance coefficient
Tau = 0.05;%powertrain response time lag
L = 5; %vehicle length

fi = -(X(3)+Af*rho*Cd*X(2)^2/(2*m)+Cr)/Tau-Af*rho*Cd*X(2)*X(3)/m;
gi = 1/(m*Tau);

h = 1; %desired time gap

e_x = P_t-X(1)-L-h*X(2); %gap error 
e_v = V_t-X(2); %speed error
a = X(3);

delta_0 = 2;
k_1_1 = 0.01;
k_1_2 = 0.01;
k_1_3 = 0.01;
e_1_1 = 1.0;
e_1_2 = 1.0;
e_1_3 = 1.0;

p_1 = k_1_1+h*delta_0/(2*e_1_1);
q_1 = k_1_2*abs(1-k_1_1*h-h^2*delta_0/(2*e_1_1))*delta_0/(2*e_1_2);
r_1 = k_1_3+abs(h-q_1*k_1_1*h-q_1*h^2*delta_0/(2*e_1_1)-p_1+q_1)*delta_0/(2*e_1_3);

Z_1_1 = e_x-h*e_v;
Z_1_2 = e_v+(k_1_1+h*delta_0/(2*e_1_1))*Z_1_1;
Z_1_3 = a - Z_1_1-p_1*e_v+q_1*Z_1_2;

% u = (-fi+(k_1_1+h*delta_0/2*e_1_1)*Z_1_1-(-2+q_1*k_1_1+q_1*h*delta_0/(2*e_1_1))*e_v-(p_1-q_1)*a-r_1*Z_1_3)/gi;
u = (-fi+(k_1_1+h*delta_0/2*e_1_1)*Z_1_1+(2+q_1*k_1_1+q_1*h*delta_0/(2*e_1_1))*e_v-(p_1+q_1)*a-r_1*Z_1_3)/gi;
