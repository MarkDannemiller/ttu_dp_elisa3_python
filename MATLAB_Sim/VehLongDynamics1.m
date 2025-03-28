function dydt = VehLongDynamics1(t,y,Veh0_Time_Step,Veh0_Pos,Veh0_Spd)

%
m = 5760;%mass
Af = 7.5;%Effective frontal area
rho = 1.206;%air mass density
Cd = 0.51;%aero drag force coefficient
Cr = 0.0041*9.8;%rolling resistance coefficient
Tau = 0.05;%powertrain response time lag

dydt = zeros(3,1);

f1 = -(y(3)+Af*rho*Cd*y(2)^2/(2*m)+Cr)/Tau-Af*rho*Cd*y(2)*y(3)/m;
g1 = 1/(m*Tau);

X = [y(1);y(2);y(3)]; %x/v/a

[u,~,~]  = BackSteppingCF_Veh1(X,t,Veh0_Time_Step,Veh0_Pos,Veh0_Spd);

% if(u>1000)
%     ui = 1000;
% else
%     ui = u;
% end
% 
% if(u<-1000)
%     ui = -1000;
% else
%     ui = u;
% end

u1 = u;

dydt(1) = y(2);
dydt(2) = y(3);
dydt(3) = f1+g1*u1;

