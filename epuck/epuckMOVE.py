from unifr_api_epuck import wrapper

ip_addr = '192.168.1.188'
r = wrapper.get_robot(ip_addr)

r.set_speed(2, -2)        #sets the speed of the wheels
r.init_sensors()      #initiates the proximity sensor

#infinite loop
while r.go_on():
    print(r.get_prox()) #prints the proximity sensor values on the terminal

    #inserts some more code here to control your robot

r.clean_up()