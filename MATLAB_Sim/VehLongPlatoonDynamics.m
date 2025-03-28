function dydt = VehLongPlatoonDynamics(t,y,Veh0_Time_Step,Veh0_Pos,Veh0_Spd)

%
% m = 5760;%mass
m = 1000;%mass
Af = 7.5;%Effective frontal area
rho = 1.206;%air mass density
Cd = 0.51;%aero drag force coefficient
Cr = 0.0041*9.8;%rolling resistance coefficient
Tau = 0.02;%powertrain response time lag

dydt = zeros(3,1);
x1 = y(1);
v1 = y(2);
a1 = y(3);
% x2 = y(4);
% v2 = y(5);
% a2 = y(6);
% x3 = y(7);
% v3 = y(8);
% a3 = y(9);

f1 = -(a1+Af*rho*Cd*v1^2/(2*m)+Cr)/Tau-Af*rho*Cd*v1*a1/m;
g1 = 1/(m*Tau);
% f2 = -(a2+Af*rho*Cd*v2^2/(2*m)+Cr)/Tau-Af*rho*Cd*v2*a2/m;
% g2 = 1/(m*Tau);
% f3 = -(a3+Af*rho*Cd*v3^2/(2*m)+Cr)/Tau-Af*rho*Cd*v3*a3/m;
% g3 = 1/(m*Tau);

X1 = [x1;v1;a1];
% X2 = [x2;v2;a2];
% X3 = [x3;v3;a3];
                                             
[u1,~,~,A1,B1,K1,Z1,~] = BackSteppingCF_Veh1(X1,t,Veh0_Time_Step,Veh0_Pos,Veh0_Spd);

% [u2,~,~,A2,B2,K2,Z2,~] = BackSteppingCF_Veh2(X2,x1,v1,A1,B1,K1,Z1);
% [u3,e_x3,e_v3,A3,B3,K3] = BackSteppingCF_Veh3(X3,x2,v2,A1,B1,K1,Z1,A2,B2,K2,Z2);


% if(u3>5000)
%     u3 = 5000;
% end
% 
% if(u3<-5000)
%     u3 = -5000;
% end


% if(u3>50000)
%     u3 = 50000;
% end
% 
% if(u3<-50000)
%     u3 = -50000;
% end

dydt(1) = y(2);
dydt(2) = y(3);
dydt(3) = f1+g1*u1;
% dydt(4) = y(5);
% dydt(5) = y(6);
% dydt(6) = f2+g2*u2;
% dydt(7) = y(8);
% dydt(8) = y(9);
% dydt(9) = f3+g3*u3;

