function dydt = VehLongDynamics(t,y)


m = 5760;%mass
Af = 7.5;%Effective frontal area
rho = 1.206;%air mass density
Cd = 0.51;%aero drag force coefficient
Cr = 0.0041*9.8;%rolling resistance coefficient
Tau = 0.05;%powertrain response time lag

dydt = zeros(3,1);

fi = -(y(3)+Af*rho*Cd*y(2)^2/(2*m)+Cr)/tau-Af*rho*Cd*y(2)*y(3)/m;
gi = 1/(m*tau);
u = BackSteppingCF(y);

dydt(1) = y(2);
dydt(2) = y(3);
dydt(3) = fi+gi*ui;

dydt = [y(2); (1-y(1)^2)*y(2)-y(1)];