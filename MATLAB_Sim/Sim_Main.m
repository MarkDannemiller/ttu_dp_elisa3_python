close all
clear all
clc

load('V0_Spd_Pos.mat'); %Veh0_Time_Step/Veh0_Spd/Veh0_Pos

% Veh0_Spd = Veh0_Spd/3.6;

Veh0_Accel = (Veh0_Spd(2:end)-Veh0_Spd(1:end-1))/0.01;
Veh0_Accel = [Veh0_Accel;Veh0_Accel(end)];
L = 5;
h = 1;

figure()
plot(Veh0_Time_Step,Veh0_Accel)
xlabel('Time(s)','FontSize',30)
ylabel('Acceleration(m/s^2)','FontSize',30)
set(gca,'FontSize',30)
%% Simulation

tspan = 0:0.1:Veh0_Time_Step(end);
% y0 = [60;60/3.6;0;40;60/3.6;0;20;60/3.6;0]; %x1(0) / v1(0) / a1(0) / x2(0) / v2(0) /a2(0) / x3(0) / v3(0) /a3(0)
% y0 = [60;60/3.6;0;40;60/3.6;0]; %x1(0) / v1(0) / a1(0) / x2(0) / v2(0) /a2(0) 
% y0 = [60;60/3.6;0;20;60/3.6;0]; %x1(0) / v1(0) / a1(0) / x2(0) / v2(0) /a2(0) 
y0 = [60;60/3.6;0]; %x1(0) / v1(0) / a1(0) / x2(0) / v2(0) /a2(0) 


% opts = odeset('RelTol',1e-1,'AbsTol',1e-1);
[t,y] = ode45(@(t,y) VehLongPlatoonDynamics(t,y,Veh0_Time_Step,Veh0_Pos,Veh0_Spd),tspan,y0);

Veh1_Pos = y(:,1);
Veh1_Spd = y(:,2);
Veh1_Accel = y(:,3);
% Veh2_Pos = y(:,4);
% Veh2_Spd = y(:,5);
% Veh2_Accel = y(:,6);
% Veh3_Pos = y(:,7);
% Veh3_Spd = y(:,8);
% Veh3_Accel = y(:,9);

e_x1_t = zeros(length(t),1);
e_v1_t = zeros(length(t),1);
% e_x2_t = Veh1_Pos-Veh2_Pos-L*ones(length(Veh2_Spd),1)-h*Veh2_Spd;
% e_v2_t = Veh1_Spd - Veh2_Spd;
% e_x3_t = Veh2_Pos-Veh3_Pos-L*ones(length(Veh3_Spd),1)-h*Veh3_Spd;
% e_v3_t = Veh2_Spd - Veh3_Spd;

u1_t = zeros(length(t),1);
% u2_t = zeros(length(t),1);
% u3_t = zeros(length(t),1);

V1_t = zeros(length(t),1);
% V2_t = zeros(length(t),1);
a1_est_t = zeros(length(t),1);
for i =1:1:length(t)
   t_s = t(i); 
   X1_t = y(i,:);
   [u1,e_x,e_v,A1,B1,K1,ZX1,GAMMA1] = BackSteppingCF_Veh1(X1_t,t_s,Veh0_Time_Step,Veh0_Pos,Veh0_Spd);
%    X2 = [Veh2_Pos(i);Veh2_Spd(i);Veh2_Accel(i)];
%    [u2,~,~,~,~,~,ZX2,GAMMA2] = BackSteppingCF_Veh2(X2,Veh1_Pos(i),Veh1_Spd(i),A1,B1,K1,ZX1);
   a1_est_t = K1'*ZX1;
   
   V1_t(i) = norm(ZX1);
%    V2_t(i) = norm(ZX2);
   e_x1_t(i) = e_x;
   e_v1_t(i) = e_v;
  
   u1_t(i) = u1;
%    u2_t(i) = u2;
end

%%
figure()
plot(Veh0_Time_Step,Veh0_Pos,'LineWidth',2)
hold on
plot(t,Veh1_Pos,'LineWidth',2)
% plot(t,Veh2_Pos,'LineWidth',2)
% plot(t,Veh3_Pos,'LineWidth',2)
legend('Veh0','Veh1','Veh2','Veh3')
xlabel('Time(sec)','FontSize',30)
ylabel('Position(m)','FontSize',30)
set(gca,'FontSize',30)

figure()
plot(Veh0_Time_Step,Veh0_Spd,'LineWidth',2)
hold on
plot(t,Veh1_Spd,'LineWidth',2)
% plot(t,Veh2_Spd,'LineWidth',2)
% plot(t,Veh3_Spd,'LineWidth',2)
xlabel('Time(sec)','FontSize',30)
ylabel('Speed(m/s)','FontSize',30)
legend('Veh0','Veh1','Veh2','Veh3')
set(gca,'FontSize',30)

figure()
plot(t,e_x1_t,'LineWidth',2)
hold on
% plot(t,e_x2_t,'LineWidth',2)
% plot(t,e_x3_t,'LineWidth',2)
xlabel('Time(sec)','FontSize',30)
ylabel('Gap error(m)','FontSize',30)
legend('Veh0-Veh1','Veh1-Veh2','Veh2-Veh3')
set(gca,'FontSize',30)

figure()
plot(t,e_v1_t,'LineWidth',2);
hold on
% plot(t,e_v2_t,'LineWidth',2);
% plot(t,e_v3_t,'LineWidth',2);
xlabel('Time(sec)','FontSize',30)
ylabel('Speed error(m/s)','FontSize',30)
legend('Veh0-Veh1','Veh1-Veh2','Veh2-Veh3')
set(gca,'FontSize',30)

figure()
plot(t,u1_t,'LineWidth',2);
hold on
% plot(t,u2_t,'LineWidth',2);
xlabel('Time(sec)','FontSize',30)
ylabel('ControlInput','FontSize',30)
legend('Veh1','Veh2')
set(gca,'FontSize',30)
ylim([-20000,20000])

%%
figure()
plot(t,V1_t/2)
hold on
% plot(t,V2_t/2)
V1_bound = (V1_t(1)/2-GAMMA1/(2*kappa1))*exp(-0.2*t)+GAMMA1/(2*kappa1);
% V2_bound = (V2_t(1)/2-GAMMA2/0.2)*exp(-0.1*t)+GAMMA2/0.1;
plot(t,V1_bound,'-.')
% plot(t,V2_bound,'-.')
legend('V1','V2','V1bound','V2bound')



