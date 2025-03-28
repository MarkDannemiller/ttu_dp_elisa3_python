clear all;
clc;

load('V0_Spd_Pos.mat'); %Veh0_Time_Step/Veh0_Spd/Veh0_Pos


Veh0_Spd = Veh0_Spd/3.6;
%% Simulate first vehicle only

tspan = 0:0.1:70;
y0 = [60,60/3.6,0]; %x0 / v0 / a0

% opts = odeset('RelTol',1e-1,'AbsTol',1e-1);
[t,y] = ode45(@(t,y) VehLongDynamics1(t,y,Veh0_Time_Step,Veh0_Pos,Veh0_Spd),tspan,y0);

Veh1_Pos = y(:,1);
Veh1_Veh = y(:,2);
Veh1_Accel = y(:,3);

e_x_t = zeros(length(t),1);
e_v_t = zeros(length(t),1);

for i =1:1:length(t)
   t_s = t(i); 
   X_s = y(i,:);
   [u,e_x,e_v] = BackSteppingCF_Veh1(X_s,t_s,Veh0_Time_Step,Veh0_Pos,Veh0_Spd);
   e_x_t(i) = e_x;
   e_v_t(i) = e_v;
end


figure()
plot(t,Veh1_Pos,'LineWidth',2)
hold on
plot(Veh0_Time_Step,Veh0_Pos,'LineWidth',2)
xlabel('Time(sec)','FontSize',30)
ylabel('Position(m)','FontSize',30)
set(gca,'FontSize',30)

figure()
plot(t,Veh1_Veh,'LineWidth',2)
hold on
plot(Veh0_Time_Step,Veh0_Spd,'LineWidth',2)
xlabel('Time(sec)','FontSize',30)
ylabel('Speed(m/s)','FontSize',30)
set(gca,'FontSize',30)

figure()
plot(t,e_x_t,'LineWidth',2)
xlabel('Time(sec)','FontSize',30)
ylabel('Gap error(m)','FontSize',30)
set(gca,'FontSize',30)

figure()
plot(t,e_v_t,'LineWidth',2);
xlabel('Time(sec)','FontSize',30)
ylabel('Speed error(m/s)','FontSize',30)


%%
set(gca,'FontSize',30)