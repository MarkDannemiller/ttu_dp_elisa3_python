# pip3 install libusb_package
# pip3 install pyusb

import os
import platform
import libusb_package
import usb
import usb.core
import threading
import time
import traceback
import sys
from struct import pack, unpack

USB_VID = 0x1915
USB_PID = 0x0101

PAYLOAD_SIZE = 13
ADDR_SIZE = 2
ROBOT_PACKET_SIZE = (PAYLOAD_SIZE+ADDR_SIZE)
PACKETS_SIZE = 64
UNUSED_BYTES = 3


def _find_devices():
    """
    Returns a list of Elisa-3 radio module devices currently connected to the computer
    """
    ret = []

    print('Looking for devices....')

    if os.name == 'nt':
        import usb.backend.libusb0 as libusb0

        backend = libusb0.get_backend()
    else:
        backend = libusb_package.get_libusb1_backend()

    devices = usb.core.find(idVendor=USB_VID, idProduct=USB_PID, find_all=1, backend=backend)
    
    if devices:
        for d in devices:
            print(d)
            ret.append(d)

    return ret


class Elisa3(threading.Thread):
    """ Used for communication with the Elisa-3 """

    # The usb buffer between the pc and the base-station is 64 bytes.
    # Each packet exchanged with the bast-station must contain as the
    # first byte the "command id" that at the moment can be either
    # "change robot state" (0x27) or "goto base-station bootloader" (0x28).
    # In order to optimize the throughput the packet exchanged with the radio
    # base-station contains informations to send to four different robots
    # simultaneously.
    # Each robot must be identified by a 2 byte address, thus we have:
    # 64 - 1 - 2*4 = 55 / 4 = 13 bytes usable for the payload of each robot.
    #
    # Payload content for each robot:
    # --------------------------------------------------------------------------
    # R | B | G | IR/flags | Right | Left | Leds | ...remaining 6 bytes not used
    # --------------------------------------------------------------------------
    #
    # * R, B, G: values from 0 (OFF) to 100 (ON max power)
    # * IR/flags:
    #   - first two bits are dedicated to the IRs:
    #     0x00 => all IRs off
    #     0x01 => back IR on
    #     0x02 => front IRs on
    #     0x03 => all IRs on
    #   - third bit is used for enabling/disablng IR remote control (0=>diabled, 1=>enabled)
    #   - fourth bit is used for sleep (1 => go to sleep for 1 minute)
    #   - fifth bit is used to calibrate all sensors (proximity, ground, accelerometer)
    #   - sixth bits is reserved (used by radio station)
    #   - seventh bit is used for enabling/disabling onboard obstacle avoidance
    #   - eight bit is used for enabling/disabling onboard cliff avoidance
    # * Right, Left: speed (in percentage); MSBit indicate direction: 1=forward, 0=backward; values from 0 to 100
    # * Leds: each bit define whether the corresponding led is turned on (1) or off(0); e.g. if bit0=1 then led0=on
    # * remaining bytes free to be used
    #
    # Overhead content :
    # - command: 1 byte, indicates which command the packet refer to
    # - address: 2 bytes per robot

    def __init__(self, robot_addr, devid=0):
        """ Create object and scan for USB dongle if no device is supplied """
        self.dev = None
        self.handle = None

        # Packet content
        self.payloadId = 0
        self.currPacketId = 0
        self.RX_buffer = bytearray(64)
        self.TX_buffer = bytearray(PACKETS_SIZE-UNUSED_BYTES)
        self.TX_buffer[0] = 0x27

        # Robots state
        self.robotAddress = [0] * 100
        self.leftSpeed = bytearray(100)
        self.rightSpeed = bytearray(100)
        self.redLed = bytearray(100)
        self.greenLed = bytearray(100)
        self.blueLed = bytearray(100)
        self.smallLeds = bytearray(100)
        self.batteryAdc = [0] * 100
        self.batteryPercent = [0] * 100
        self.accX = [0] * 100
        self.accY = [0] * 100
        self.accZ = [0] * 100
        self.selector = bytearray(100)
        self.tvRemote = bytearray(100)
        self.flagsRX = bytearray(100)
        self.flagsRX = bytearray(100)
        self.robTheta = [0] * 100
        self.robXPos = [0] * 100
        self.robYPos = [0] * 100
        self.leftMotSteps = [0] * 100
        self.rightMotSteps = [0] * 100
        self.sleepEnabledFlag = bytearray(100)
        self.proxValue = [[0 for x in range(8)] for y in range(100)]
        self.proxAmbientValue = [[0 for x in range(8)] for y in range(100)]
        self.groundValue = [[0 for x in range(4)] for y in range(100)]
        self.groundAmbientValue = [[0 for x in range(4)] for y in range(100)]
        self.flagsTX = [[0 for x in range(2)] for y in range(100)]        

        devices = _find_devices()
        if not devices:
            print("Cannot find base station!");

        try:
            self.dev = devices[devid]
        except Exception:
            self.dev = None

        self.currNumRobots = len(robot_addr)
        print("num of robots = " + str(self.currNumRobots))

        self.robotAddress[0:self.currNumRobots] = robot_addr
        print("robot addresses: " + str(self.robotAddress[0:self.currNumRobots]) + " (len=" + str(len(self.robotAddress)) + ")")

        # try:  # configuration might already be confgiured by composite VCP, try claim interface
        #     usb.util.claim_interface(self.dev, 0)
        # except Exception:
        #     try:
        #         self.dev.set_configuration()  # it was not, then set configuration
        #     except Exception:
        #         if self.dev:
        #             if platform.system() == 'Linux':
        #                 self.dev.reset()
        #                 self.dev.set_configuration()

        self.handle = self.dev

        threading.Thread.__init__(self)

    def close(self):
        if self.dev:
            usb.util.dispose_resources(self.dev)

        self.handle = None
        self.dev = None

    def speed(self, value):
        if(value <= 127):
            return (value|0x80)
        else:
            return ((256-value)&0x7F)

    def send_packet(self, dataOut):
        """ Send a packet and receive the ack from the radio dongle
            The ack contains information about the packet transmission
            and a data payload if the ack packet contained any """
        try:
            self.handle.write(endpoint=1, data=dataOut, timeout=20)
        except usb.USBError:
            pass

    def receive_packet(self):
        dataIn = ()
        try:
            dataIn = self.handle.read(0x81, 64, timeout=20)
        except usb.USBError as e:
            try:
                if e.backend_error_code == -7 or e.backend_error_code == -116:
                    # Normal, the read was empty
                    pass
                else:
                    raise IOError('Crazyflie disconnected')
            except AttributeError:
                # pyusb < 1.0 doesn't implement getting the underlying error
                # number and it seems as if it's not possible to detect
                # if the cable is disconnected. So this detection is not
                # supported, but the "normal" case will work.
                pass

        return dataIn

    def transfer_data(self):
        """ Send a packet and receive the ack from the radio dongle
            The ack contains information about the packet transmission
            and a data payload if the ack packet contained any """

        try:
            
            #print("addresses: " + str(self.robotAddress[self.currPacketId*4+0]) + ","  + str(self.robotAddress[self.currPacketId*4+1]) + "," + str(self.robotAddress[self.currPacketId*4+2]) + "," + str(self.robotAddress[self.currPacketId*4+3]))
            #print("currPacketId = " + str(self.currPacketId))

            # First robot
            self.TX_buffer[(0*ROBOT_PACKET_SIZE)+1] = self.redLed[self.currPacketId*4+0]
            self.TX_buffer[(0*ROBOT_PACKET_SIZE)+2] = self.blueLed[self.currPacketId*4+0]
            self.TX_buffer[(0*ROBOT_PACKET_SIZE)+3] = self.greenLed[self.currPacketId*4+0]
            self.TX_buffer[(0*ROBOT_PACKET_SIZE)+4] = self.flagsTX[self.currPacketId*4+0][0]
            self.TX_buffer[(0*ROBOT_PACKET_SIZE)+5] = self.speed(self.rightSpeed[self.currPacketId*4+0])            
            self.TX_buffer[(0*ROBOT_PACKET_SIZE)+6] = self.speed(self.leftSpeed[self.currPacketId*4+0])
            self.TX_buffer[(0*ROBOT_PACKET_SIZE)+7] = self.smallLeds[self.currPacketId*4+0]
            self.TX_buffer[(0*ROBOT_PACKET_SIZE)+8] = self.flagsTX[self.currPacketId*4+0][1]
            self.TX_buffer[(0*ROBOT_PACKET_SIZE)+9] = self.payloadId;
            self.TX_buffer[(0*ROBOT_PACKET_SIZE)+14] = (self.robotAddress[self.currPacketId*4+0]>>8)&0xFF
            self.TX_buffer[(0*ROBOT_PACKET_SIZE)+15] = self.robotAddress[self.currPacketId*4+0]&0xFF

            # Second robot
            self.TX_buffer[(1*ROBOT_PACKET_SIZE)+1] = self.redLed[self.currPacketId*4+1]
            self.TX_buffer[(1*ROBOT_PACKET_SIZE)+2] = self.blueLed[self.currPacketId*4+1]
            self.TX_buffer[(1*ROBOT_PACKET_SIZE)+3] = self.greenLed[self.currPacketId*4+1]
            self.TX_buffer[(1*ROBOT_PACKET_SIZE)+4] = self.flagsTX[self.currPacketId*4+1][0]
            self.TX_buffer[(1*ROBOT_PACKET_SIZE)+5] = self.speed(self.rightSpeed[self.currPacketId*4+1])
            self.TX_buffer[(1*ROBOT_PACKET_SIZE)+6] = self.speed(self.leftSpeed[self.currPacketId*4+1])
            self.TX_buffer[(1*ROBOT_PACKET_SIZE)+7] = self.smallLeds[self.currPacketId*4+1]
            self.TX_buffer[(1*ROBOT_PACKET_SIZE)+8] = self.flagsTX[self.currPacketId*4+1][1]
            self.TX_buffer[(1*ROBOT_PACKET_SIZE)+9] = self.payloadId
            self.TX_buffer[(1*ROBOT_PACKET_SIZE)+14] = (self.robotAddress[self.currPacketId*4+1]>>8)&0xFF
            self.TX_buffer[(1*ROBOT_PACKET_SIZE)+15] = self.robotAddress[self.currPacketId*4+1]&0xFF

            # Third robot
            self.TX_buffer[(2*ROBOT_PACKET_SIZE)+1] = self.redLed[self.currPacketId*4+2]
            self.TX_buffer[(2*ROBOT_PACKET_SIZE)+2] = self.blueLed[self.currPacketId*4+2]
            self.TX_buffer[(2*ROBOT_PACKET_SIZE)+3] = self.greenLed[self.currPacketId*4+2]
            self.TX_buffer[(2*ROBOT_PACKET_SIZE)+4] = self.flagsTX[self.currPacketId*4+2][0]
            self.TX_buffer[(2*ROBOT_PACKET_SIZE)+5] = self.speed(self.rightSpeed[self.currPacketId*4+2])
            self.TX_buffer[(2*ROBOT_PACKET_SIZE)+6] = self.speed(self.leftSpeed[self.currPacketId*4+2])
            self.TX_buffer[(2*ROBOT_PACKET_SIZE)+7] = self.smallLeds[self.currPacketId*4+2]
            self.TX_buffer[(2*ROBOT_PACKET_SIZE)+8] = self.flagsTX[self.currPacketId*4+2][1]
            self.TX_buffer[(2*ROBOT_PACKET_SIZE)+9] = self.payloadId
            self.TX_buffer[(2*ROBOT_PACKET_SIZE)+14] = (self.robotAddress[self.currPacketId*4+2]>>8)&0xFF
            self.TX_buffer[(2*ROBOT_PACKET_SIZE)+15] = self.robotAddress[self.currPacketId*4+2]&0xFF

            # Fourth robot
            self.TX_buffer[(3*ROBOT_PACKET_SIZE)+1] = self.redLed[self.currPacketId*4+3]
            self.TX_buffer[(3*ROBOT_PACKET_SIZE)+2] = self.blueLed[self.currPacketId*4+3]
            self.TX_buffer[(3*ROBOT_PACKET_SIZE)+3] = self.greenLed[self.currPacketId*4+3]
            self.TX_buffer[(3*ROBOT_PACKET_SIZE)+4] = self.flagsTX[self.currPacketId*4+3][0]
            self.TX_buffer[(3*ROBOT_PACKET_SIZE)+5] = self.speed(self.rightSpeed[self.currPacketId*4+3])
            self.TX_buffer[(3*ROBOT_PACKET_SIZE)+6] = self.speed(self.leftSpeed[self.currPacketId*4+3])
            self.TX_buffer[(3*ROBOT_PACKET_SIZE)+7] = self.smallLeds[self.currPacketId*4+3]
            self.TX_buffer[(3*ROBOT_PACKET_SIZE)+8] = self.flagsTX[self.currPacketId*4+3][1]
            self.TX_buffer[(3*ROBOT_PACKET_SIZE)+9] = self.payloadId
            self.TX_buffer[(3*ROBOT_PACKET_SIZE)+14] = (self.robotAddress[self.currPacketId*4+3]>>8)&0xFF
            self.TX_buffer[(3*ROBOT_PACKET_SIZE)+15] = self.robotAddress[self.currPacketId*4+3]&0xFF

            #print("TX = " + str(self.TX_buffer) + ", len = " + str(len(self.TX_buffer)))
            # Send the data to the radio module (base-station)
            self.handle.write(endpoint=1, data=self.TX_buffer, timeout=1000)

            self.RX_buffer[0] = 0;
            self.RX_buffer[16] = 0;
            self.RX_buffer[32] = 0;
            self.RX_buffer[48] = 0;
            # Receive the ack payload for 4 robots at a time (16 bytes for each one)
            self.RX_buffer = self.handle.read(0x81, 64, timeout=1000)
            #print("RX = " + str(self.RX_buffer) + ", len = " + str(len(self.RX_buffer)))

            # The base-station returns these "error" codes:
            # 0 => transmission succeed (no ack received though)
            # 1 => ack received (should not be returned because if the ack is received, then the payload is read)
            # 2 => transfer failed
            # Interpret the data received based on the packet id (first byte):
            # id=3 | prox0         | prox1         | prox2         | prox3         | prox5         | prox6         | prox7         | flags
            # id=4 | prox4         | gound0        | ground1       | ground2       | ground3       | accX          | accY          | tv remote
            # id=5 | proxAmbient0  | proxAmbient1  | proxAmbient2  | proxAmbient3  | proxAmbient5  | proxAmbient6  | proxAmbient7  | selector
            # id=6 | proxAmbient4  | goundAmbient0 | goundAmbient1 | goundAmbient2 | goundAmbient3 | accZ          | battery       | free byte

            # First robot
            if(self.RX_buffer[0] <= 2): # if something goes wrong skip the data
                #print("transfer failed to robot " + str(self.currPacketId*4+0) + "(addr=" + str(self.robotAddress[self.currPacketId*4+0]));
                pass
            else:
                if(self.RX_buffer[0] == 3):
                    self.proxValue[self.currPacketId*4+0][0] = (self.RX_buffer[2]<<8)|(self.RX_buffer[1])
                    self.proxValue[self.currPacketId*4+0][1] = (self.RX_buffer[4]<<8)|(self.RX_buffer[3])
                    self.proxValue[self.currPacketId*4+0][2] = (self.RX_buffer[6]<<8)|(self.RX_buffer[5])
                    self.proxValue[self.currPacketId*4+0][3] = (self.RX_buffer[8]<<8)|(self.RX_buffer[7])
                    self.proxValue[self.currPacketId*4+0][5] = (self.RX_buffer[10]<<8)|(self.RX_buffer[9])
                    self.proxValue[self.currPacketId*4+0][6] = (self.RX_buffer[12]<<8)|(self.RX_buffer[11])
                    self.proxValue[self.currPacketId*4+0][7] = (self.RX_buffer[14]<<8)|(self.RX_buffer[13])
                    self.flagsRX[self.currPacketId*4+0] = self.RX_buffer[15]
                    #print("prox[" + str(self.currPacketId*4+0) + "][0] = " + str(self.proxValue[self.currPacketId*4+0][0]) + "(" + str(self.RX_buffer[2]) + "," + str(self.RX_buffer[1]) + ")")
                elif(self.RX_buffer[0] == 4):
                    self.proxValue[self.currPacketId*4+0][4] = (self.RX_buffer[2]<<8)|(self.RX_buffer[1])
                    self.groundValue[self.currPacketId*4+0][0] = (self.RX_buffer[4]<<8)|(self.RX_buffer[3])
                    self.groundValue[self.currPacketId*4+0][1] = (self.RX_buffer[6]<<8)|(self.RX_buffer[5])
                    self.groundValue[self.currPacketId*4+0][2] = (self.RX_buffer[8]<<8)|(self.RX_buffer[7])
                    self.groundValue[self.currPacketId*4+0][3] = (self.RX_buffer[10]<<8)|(self.RX_buffer[9])
                    self.accX[self.currPacketId*4+0] = unpack("<h", pack("<BB", self.RX_buffer[11], self.RX_buffer[12]))[0]
                    self.accY[self.currPacketId*4+0] = unpack("<h", pack("<BB", self.RX_buffer[13], self.RX_buffer[14]))[0]
                    self.tvRemote[self.currPacketId*4+0] = self.RX_buffer[15]
                elif(self.RX_buffer[0] == 5):
                    self.proxAmbientValue[self.currPacketId*4+0][0] = (self.RX_buffer[2]<<8)|(self.RX_buffer[1])
                    self.proxAmbientValue[self.currPacketId*4+0][1] = (self.RX_buffer[4]<<8)|(self.RX_buffer[3])
                    self.proxAmbientValue[self.currPacketId*4+0][2] = (self.RX_buffer[6]<<8)|(self.RX_buffer[5])
                    self.proxAmbientValue[self.currPacketId*4+0][3] = (self.RX_buffer[8]<<8)|(self.RX_buffer[7])
                    self.proxAmbientValue[self.currPacketId*4+0][5] = (self.RX_buffer[10]<<8)|(self.RX_buffer[9])
                    self.proxAmbientValue[self.currPacketId*4+0][6] = (self.RX_buffer[12]<<8)|(self.RX_buffer[11])
                    self.proxAmbientValue[self.currPacketId*4+0][7] = (self.RX_buffer[14]<<8)|(self.RX_buffer[13])
                    self.selector[self.currPacketId*4+0] = self.RX_buffer[15]
                elif(self.RX_buffer[0] == 6):
                    self.proxAmbientValue[self.currPacketId*4+0][4] = (self.RX_buffer[2]<<8)|(self.RX_buffer[1])
                    self.groundAmbientValue[self.currPacketId*4+0][0] = (self.RX_buffer[4]<<8)|(self.RX_buffer[3])
                    self.groundAmbientValue[self.currPacketId*4+0][1] = (self.RX_buffer[6]<<8)|(self.RX_buffer[5])
                    self.groundAmbientValue[self.currPacketId*4+0][2] = (self.RX_buffer[8]<<8)|(self.RX_buffer[7])
                    self.groundAmbientValue[self.currPacketId*4+0][3] = (self.RX_buffer[10]<<8)|(self.RX_buffer[9])
                    self.accZ[self.currPacketId*4+0] = unpack("<h", pack("<BB", self.RX_buffer[11], self.RX_buffer[12]))[0]
                    self.batteryAdc[self.currPacketId*4+0] = (self.RX_buffer[14]<<8)|(self.RX_buffer[13])
                    # self.RX_buffer[15] is free
                elif(self.RX_buffer[0] == 7):
                    self.leftMotSteps[self.currPacketId*4+0] = unpack("<l", pack("<BBBB", self.RX_buffer[1], self.RX_buffer[2], self.RX_buffer[3], self.RX_buffer[4]))[0]
                    self.rightMotSteps[self.currPacketId*4+0] =  unpack("<l", pack("<BBBB", self.RX_buffer[5], self.RX_buffer[6], self.RX_buffer[7], self.RX_buffer[8]))[0]
                    self.robTheta[self.currPacketId*4+0] = (unpack("<h", pack("<BB", self.RX_buffer[9], self.RX_buffer[10]))[0])/10
                    self.robXPos[self.currPacketId*4+0] = unpack("<h", pack("<BB", self.RX_buffer[11], self.RX_buffer[12]))[0]
                    self.robYPos[self.currPacketId*4+0] = unpack("<h", pack("<BB", self.RX_buffer[13], self.RX_buffer[14]))[0]

            # Second robot
            if(self.RX_buffer[16] <= 2): # if something goes wrong skip the data
                #print("transfer failed to robot " + str(self.currPacketId*4+1) + "(addr=" + str(self.robotAddress[self.currPacketId*4+1]));
                pass
            else:
                if(self.RX_buffer[16] == 3):
                    self.proxValue[self.currPacketId*4+1][0] = (self.RX_buffer[18]<<8)|(self.RX_buffer[17])
                    self.proxValue[self.currPacketId*4+1][1] = (self.RX_buffer[20]<<8)|(self.RX_buffer[19])
                    self.proxValue[self.currPacketId*4+1][2] = (self.RX_buffer[22]<<8)|(self.RX_buffer[21])
                    self.proxValue[self.currPacketId*4+1][3] = (self.RX_buffer[24]<<8)|(self.RX_buffer[23])
                    self.proxValue[self.currPacketId*4+1][5] = (self.RX_buffer[26]<<8)|(self.RX_buffer[25])
                    self.proxValue[self.currPacketId*4+1][6] = (self.RX_buffer[28]<<8)|(self.RX_buffer[27])
                    self.proxValue[self.currPacketId*4+1][7] = (self.RX_buffer[30]<<8)|(self.RX_buffer[29])
                    self.flagsRX[self.currPacketId*4+1] = self.RX_buffer[31]
                    #print("prox[" + str(self.currPacketId*4+1) + "][0] = " + str(self.proxValue[self.currPacketId*4+1][0]) + "(" + str(self.RX_buffer[18]) + "," + str(self.RX_buffer[17]) + ")")
                elif(self.RX_buffer[16] == 4):
                    self.proxValue[self.currPacketId*4+1][4] = (self.RX_buffer[18]<<8)|(self.RX_buffer[17])
                    self.groundValue[self.currPacketId*4+1][0] = (self.RX_buffer[20]<<8)|(self.RX_buffer[19])
                    self.groundValue[self.currPacketId*4+1][1] = (self.RX_buffer[22]<<8)|(self.RX_buffer[21])
                    self.groundValue[self.currPacketId*4+1][2] = (self.RX_buffer[24]<<8)|(self.RX_buffer[23])
                    self.groundValue[self.currPacketId*4+1][3] = (self.RX_buffer[26]<<8)|(self.RX_buffer[25])
                    self.accX[self.currPacketId*4+1] = unpack("<h", pack("<BB", self.RX_buffer[27], self.RX_buffer[28]))[0]
                    self.accY[self.currPacketId*4+1] = unpack("<h", pack("<BB", self.RX_buffer[29], self.RX_buffer[30]))[0]
                    self.tvRemote[self.currPacketId*4+1] = self.RX_buffer[31]
                elif(self.RX_buffer[16] == 5):
                    self.proxAmbientValue[self.currPacketId*4+1][0] = (self.RX_buffer[18]<<8)|(self.RX_buffer[17])
                    self.proxAmbientValue[self.currPacketId*4+1][1] = (self.RX_buffer[20]<<8)|(self.RX_buffer[19])
                    self.proxAmbientValue[self.currPacketId*4+1][2] = (self.RX_buffer[22]<<8)|(self.RX_buffer[21])
                    self.proxAmbientValue[self.currPacketId*4+1][3] = (self.RX_buffer[24]<<8)|(self.RX_buffer[23])
                    self.proxAmbientValue[self.currPacketId*4+1][5] = (self.RX_buffer[26]<<8)|(self.RX_buffer[25])
                    self.proxAmbientValue[self.currPacketId*4+1][6] = (self.RX_buffer[28]<<8)|(self.RX_buffer[27])
                    self.proxAmbientValue[self.currPacketId*4+1][7] = (self.RX_buffer[30]<<8)|(self.RX_buffer[29])
                    self.selector[self.currPacketId*4+1] = self.RX_buffer[31]
                elif(self.RX_buffer[16] == 6):
                    self.proxAmbientValue[self.currPacketId*4+1][4] = (self.RX_buffer[18]<<8)|(self.RX_buffer[17])
                    self.groundAmbientValue[self.currPacketId*4+1][0] = (self.RX_buffer[20]<<8)|(self.RX_buffer[19])
                    self.groundAmbientValue[self.currPacketId*4+1][1] = (self.RX_buffer[22]<<8)|(self.RX_buffer[21])
                    self.groundAmbientValue[self.currPacketId*4+1][2] = (self.RX_buffer[24]<<8)|(self.RX_buffer[23])
                    self.groundAmbientValue[self.currPacketId*4+1][3] = (self.RX_buffer[26]<<8)|(self.RX_buffer[25])
                    self.accZ[self.currPacketId*4+1] = unpack("<h", pack("<BB", self.RX_buffer[27], self.RX_buffer[28]))[0]
                    self.batteryAdc[self.currPacketId*4+1] = (self.RX_buffer[30]<<8)|(self.RX_buffer[29])
                    # self.RX_buffer[31] is free
                elif(self.RX_buffer[16] == 7):
                    self.leftMotSteps[self.currPacketId*4+1] = unpack("<l", pack("<BBBB", self.RX_buffer[17], self.RX_buffer[18], self.RX_buffer[19], self.RX_buffer[20]))[0]
                    self.rightMotSteps[self.currPacketId*4+1] = unpack("<l", pack("<BBBB", self.RX_buffer[21], self.RX_buffer[22], self.RX_buffer[23], self.RX_buffer[24]))[0]
                    self.robTheta[self.currPacketId*4+1] = (unpack("<h", pack("<BB", self.RX_buffer[25], self.RX_buffer[26]))[0])/10
                    self.robXPos[self.currPacketId*4+1] = unpack("<h", pack("<BB", self.RX_buffer[27], self.RX_buffer[28]))[0]
                    self.robYPos[self.currPacketId*4+1] = unpack("<h", pack("<BB", self.RX_buffer[29], self.RX_buffer[30]))[0]

            # Third robot
            if(self.RX_buffer[32] <= 2): # if something goes wrong skip the data
                #print("transfer failed to robot " + str(self.currPacketId*4+2) + "(addr=" + str(self.robotAddress[self.currPacketId*4+2]));
                pass
            else:
                if(self.RX_buffer[32] == 3):
                    self.proxValue[self.currPacketId*4+2][0] = (self.RX_buffer[34]<<8)|(self.RX_buffer[33])
                    self.proxValue[self.currPacketId*4+2][1] = (self.RX_buffer[36]<<8)|(self.RX_buffer[35])
                    self.proxValue[self.currPacketId*4+2][2] = (self.RX_buffer[38]<<8)|(self.RX_buffer[37])
                    self.proxValue[self.currPacketId*4+2][3] = (self.RX_buffer[40]<<8)|(self.RX_buffer[39])
                    self.proxValue[self.currPacketId*4+2][5] = (self.RX_buffer[42]<<8)|(self.RX_buffer[41])
                    self.proxValue[self.currPacketId*4+2][6] = (self.RX_buffer[44]<<8)|(self.RX_buffer[43])
                    self.proxValue[self.currPacketId*4+2][7] = (self.RX_buffer[46]<<8)|(self.RX_buffer[45])
                    self.flagsRX[self.currPacketId*4+2] = self.RX_buffer[47]
                    #print("prox[" + str(self.currPacketId*4+2) + "][0] = " + str(self.proxValue[self.currPacketId*4+2][0]) + "(" + str(self.RX_buffer[34]) + "," + str(self.RX_buffer[33]) + ")")
                elif(self.RX_buffer[32] == 4):
                    self.proxValue[self.currPacketId*4+2][4] = (self.RX_buffer[34]<<8)|(self.RX_buffer[33])
                    self.groundValue[self.currPacketId*4+2][0] = (self.RX_buffer[36]<<8)|(self.RX_buffer[35])
                    self.groundValue[self.currPacketId*4+2][1] = (self.RX_buffer[38]<<8)|(self.RX_buffer[37])
                    self.groundValue[self.currPacketId*4+2][2] = (self.RX_buffer[40]<<8)|(self.RX_buffer[39])
                    self.groundValue[self.currPacketId*4+2][3] = (self.RX_buffer[42]<<8)|(self.RX_buffer[41])
                    self.accX[self.currPacketId*4+2] =  unpack("<h", pack("<BB", self.RX_buffer[43], self.RX_buffer[44]))[0]
                    self.accY[self.currPacketId*4+2] = unpack("<h", pack("<BB", self.RX_buffer[45], self.RX_buffer[46]))[0]
                    self.tvRemote[self.currPacketId*4+2] = self.RX_buffer[47]
                elif(self.RX_buffer[32] == 5):
                    self.proxAmbientValue[self.currPacketId*4+2][0] = (self.RX_buffer[34]<<8)|(self.RX_buffer[33])
                    self.proxAmbientValue[self.currPacketId*4+2][1] = (self.RX_buffer[36]<<8)|(self.RX_buffer[35])
                    self.proxAmbientValue[self.currPacketId*4+2][2] = (self.RX_buffer[38]<<8)|(self.RX_buffer[37])
                    self.proxAmbientValue[self.currPacketId*4+2][3] = (self.RX_buffer[40]<<8)|(self.RX_buffer[39])
                    self.proxAmbientValue[self.currPacketId*4+2][5] = (self.RX_buffer[42]<<8)|(self.RX_buffer[41])
                    self.proxAmbientValue[self.currPacketId*4+2][6] = (self.RX_buffer[44]<<8)|(self.RX_buffer[43])
                    self.proxAmbientValue[self.currPacketId*4+2][7] = (self.RX_buffer[46]<<8)|(self.RX_buffer[45])
                    self.selector[self.currPacketId*4+2] = self.RX_buffer[47]
                elif(self.RX_buffer[32] == 6):
                    self.proxAmbientValue[self.currPacketId*4+2][4] = (self.RX_buffer[34]<<8)|(self.RX_buffer[33])
                    self.groundAmbientValue[self.currPacketId*4+2][0] = (self.RX_buffer[36]<<8)|(self.RX_buffer[35])
                    self.groundAmbientValue[self.currPacketId*4+2][1] = (self.RX_buffer[38]<<8)|(self.RX_buffer[37])
                    self.groundAmbientValue[self.currPacketId*4+2][2] = (self.RX_buffer[40]<<8)|(self.RX_buffer[39])
                    self.groundAmbientValue[self.currPacketId*4+2][3] = (self.RX_buffer[42]<<8)|(self.RX_buffer[41])
                    self.accZ[self.currPacketId*4+2] = unpack("<h", pack("<BB", self.RX_buffer[43], self.RX_buffer[44]))[0]
                    self.batteryAdc[self.currPacketId*4+2] = (self.RX_buffer[46]<<8)|(self.RX_buffer[45])
                    # self.RX_buffer[47] is free
                elif(self.RX_buffer[32] == 7):
                    self.leftMotSteps[self.currPacketId*4+2] = unpack("<l", pack("<BBBB", self.RX_buffer[33], self.RX_buffer[34], self.RX_buffer[35], self.RX_buffer[36]))[0]
                    self.rightMotSteps[self.currPacketId*4+2] = unpack("<l", pack("<BBBB", self.RX_buffer[37], self.RX_buffer[38], self.RX_buffer[39], self.RX_buffer[40]))[0]
                    self.robTheta[self.currPacketId*4+2] = (unpack("<h", pack("<BB", self.RX_buffer[41], self.RX_buffer[42]))[0])/10
                    self.robXPos[self.currPacketId*4+2] = unpack("<h", pack("<BB", self.RX_buffer[43], self.RX_buffer[44]))[0]
                    self.robYPos[self.currPacketId*4+2] = unpack("<h", pack("<BB", self.RX_buffer[45], self.RX_buffer[46]))[0]

            # Fourth robot
            if(self.RX_buffer[48] <= 2): # if something goes wrong skip the data
                #print("transfer failed to robot " + str(self.currPacketId*4+3) + "(addr=" + str(self.robotAddress[self.currPacketId*4+3]));
                pass
            else:
                if(self.RX_buffer[48] == 3):
                    self.proxValue[self.currPacketId*4+3][0] = (self.RX_buffer[50]<<8)|(self.RX_buffer[49])
                    self.proxValue[self.currPacketId*4+3][1] = (self.RX_buffer[52]<<8)|(self.RX_buffer[51])
                    self.proxValue[self.currPacketId*4+3][2] = (self.RX_buffer[54]<<8)|(self.RX_buffer[53])
                    self.proxValue[self.currPacketId*4+3][3] = (self.RX_buffer[56]<<8)|(self.RX_buffer[55])
                    self.proxValue[self.currPacketId*4+3][5] = (self.RX_buffer[58]<<8)|(self.RX_buffer[57])
                    self.proxValue[self.currPacketId*4+3][6] = (self.RX_buffer[60]<<8)|(self.RX_buffer[59])
                    self.proxValue[self.currPacketId*4+3][7] = (self.RX_buffer[62]<<8)|(self.RX_buffer[61])
                    self.flagsRX[self.currPacketId*4+3] = self.RX_buffer[63]
                    #print("prox[" + str(self.currPacketId*4+3) + "][0] = " + str(self.proxValue[self.currPacketId*4+3][0]) + "(" + str(self.RX_buffer[50]) + "," + str(self.RX_buffer[49]) + ")")
                elif(self.RX_buffer[48] == 4):
                    self.proxValue[self.currPacketId*4+3][4] = (self.RX_buffer[50]<<8)|(self.RX_buffer[49])
                    self.groundValue[self.currPacketId*4+3][0] = (self.RX_buffer[52]<<8)|(self.RX_buffer[51])
                    self.groundValue[self.currPacketId*4+3][1] = (self.RX_buffer[54]<<8)|(self.RX_buffer[53])
                    self.groundValue[self.currPacketId*4+3][2] = (self.RX_buffer[56]<<8)|(self.RX_buffer[55])
                    self.groundValue[self.currPacketId*4+3][3] = (self.RX_buffer[58]<<8)|(self.RX_buffer[57])
                    self.accX[self.currPacketId*4+3] = unpack("<h", pack("<BB", self.RX_buffer[59], self.RX_buffer[60]))[0]
                    self.accY[self.currPacketId*4+3] = unpack("<h", pack("<BB", self.RX_buffer[61], self.RX_buffer[62]))[0]
                    self.tvRemote[self.currPacketId*4+3] = self.RX_buffer[63]
                elif(self.RX_buffer[48] == 5):
                    self.proxAmbientValue[self.currPacketId*4+3][0] = (self.RX_buffer[50]<<8)|(self.RX_buffer[49])
                    self.proxAmbientValue[self.currPacketId*4+3][1] = (self.RX_buffer[52]<<8)|(self.RX_buffer[51])
                    self.proxAmbientValue[self.currPacketId*4+3][2] = (self.RX_buffer[54]<<8)|(self.RX_buffer[53])
                    self.proxAmbientValue[self.currPacketId*4+3][3] = (self.RX_buffer[56]<<8)|(self.RX_buffer[55])
                    self.proxAmbientValue[self.currPacketId*4+3][5] = (self.RX_buffer[58]<<8)|(self.RX_buffer[57])
                    self.proxAmbientValue[self.currPacketId*4+3][6] = (self.RX_buffer[60]<<8)|(self.RX_buffer[59])
                    self.proxAmbientValue[self.currPacketId*4+3][7] = (self.RX_buffer[62]<<8)|(self.RX_buffer[61])
                    self.selector[self.currPacketId*4+3] = self.RX_buffer[63]
                elif(self.RX_buffer[48] == 6):
                    self.proxAmbientValue[self.currPacketId*4+3][4] = (self.RX_buffer[50]<<8)|(self.RX_buffer[49])
                    self.groundAmbientValue[self.currPacketId*4+3][0] = (self.RX_buffer[52]<<8)|(self.RX_buffer[51])
                    self.groundAmbientValue[self.currPacketId*4+3][1] = (self.RX_buffer[54]<<8)|(self.RX_buffer[53])
                    self.groundAmbientValue[self.currPacketId*4+3][2] = (self.RX_buffer[56]<<8)|(self.RX_buffer[55])
                    self.groundAmbientValue[self.currPacketId*4+3][3] = (self.RX_buffer[58]<<8)|(self.RX_buffer[57])
                    self.accZ[self.currPacketId*4+3] = unpack("<h", pack("<BB", self.RX_buffer[59], self.RX_buffer[60]))[0]
                    self.batteryAdc[self.currPacketId*4+3] = (self.RX_buffer[62]<<8)|(self.RX_buffer[61])
                    # self.RX_buffer[63] is free
                elif(self.RX_buffer[48] == 7):
                    self.leftMotSteps[self.currPacketId*4+3] = unpack("<l", pack("<BBBB", self.RX_buffer[49], self.RX_buffer[50], self.RX_buffer[51], self.RX_buffer[52]))[0]
                    self.rightMotSteps[self.currPacketId*4+3] = unpack("<l", pack("<BBBB", self.RX_buffer[53], self.RX_buffer[54], self.RX_buffer[55], self.RX_buffer[56]))[0]
                    self.robTheta[self.currPacketId*4+3] = (unpack("<h", pack("<BB", self.RX_buffer[57], self.RX_buffer[58]))[0])/10
                    self.robXPos[self.currPacketId*4+3] = unpack("<h", pack("<BB", self.RX_buffer[59], self.RX_buffer[60]))[0]
                    self.robYPos[self.currPacketId*4+3] = unpack("<h", pack("<BB", self.RX_buffer[61], self.RX_buffer[62]))[0]
      
        except usb.USBError:
            traceback.print_exc(file=sys.stdout)
            pass
    
    def getIdFromAddress(self, address):
        for i  in range(0, self.currNumRobots):
            if(address == self.robotAddress[i]):
                return i
        return -1

    def setLeftSpeed(self, robotAddr, speed):
        """Set the left speed of the robot.

        Parameters
        ----------
        robotAddr: the address of the robot for which to change the packet.
        speed: The speed, range is between -128 to 127.

        Returns
        -------
        None
        """
        id = self.getIdFromAddress(robotAddr)
        if(id >= 0):
            if speed > 127:
                speed = 127
            if speed < -128:
                speed = -128
            self.leftSpeed[id] = speed.to_bytes(1, 'big', signed=True)[0]
            print("left = " + str(self.leftSpeed[id]))
            print("tx left speed = " + str(self.speed(self.leftSpeed[id])))

    def setRightSpeed(self, robotAddr, speed):
        """Set the left speed of the robot.

        Parameters
        ----------
        robotAddr: the address of the robot for which to change the packet.
        speed: The speed, range is between -128 to 127.

        Returns
        -------
        None
        """
        id = self.getIdFromAddress(robotAddr)
        if(id >= 0):
            if speed > 127:
                speed = 127
            if speed < -128:
                speed = -128
            self.rightSpeed[id] = speed.to_bytes(1, 'big', signed=True)[0]

    def setRed(self, robotAddr, value):
        """Set the red intensity of the RGB led on the robot.

        Parameters
        ----------
        robotAddr: the address of the robot for which to change the packet.
        value: The intensity, range is between 0 (led off) to 100 (max power).

        Returns
        -------
        None
        """
        id = self.getIdFromAddress(robotAddr)
        if(id >= 0):
            if value < 0:
                value = 0
            if value > 100:
                value = 100
            self.redLed[id] = value

    def setGreen(self, robotAddr, value):
        """Set the green intensity of the RGB led on the robot.

        Parameters
        ----------
        robotAddr: the address of the robot for which to change the packet.
        value: The intensity, range is between 0 (led off) to 100 (max power).

        Returns
        -------
        None
        """
        id = self.getIdFromAddress(robotAddr)
        if(id >= 0):
            if value < 0:
                value = 0
            if value > 100:
                value = 100
            self.greenLed[id] = value

    def setBlue(self, robotAddr, value):
        """Set the blue intensity of the RGB led on the robot.

        Parameters
        ----------
        robotAddr: the address of the robot for which to change the packet.
        value: The intensity, range is between 0 (led off) to 100 (max power).

        Returns
        -------
        None
        """    
        id = self.getIdFromAddress(robotAddr)
        if(id >= 0):
            if value < 0:
                value = 0
            if value > 100:
                value = 100
            self.blueLed[id] = value

    def turnOnFrontIRs(self, robotAddr):
        """Turn on both the front IRs transmitter on the robot.

        Parameters
        ----------
        robotAddr: the address of the robot for which to change the packet.

        Returns
        -------
        None
        """   
        id = self.getIdFromAddress(robotAddr)
        if(id >= 0):
            self.flagsTX[id][0] |= (1 << 1)

    def turnOffFrontIRs(self, robotAddr):
        """Turn off both the front IRs transmitter on the robot.

        Parameters
        ----------
        robotAddr: the address of the robot for which to change the packet.

        Returns
        -------
        None
        """  
        id = self.getIdFromAddress(robotAddr)
        if(id >= 0):
            self.flagsTX[id][0] &= ~(1 << 1)

    def turnOnBackIR(self, robotAddr):
        """Turn on the back IR transmitter on the robot.

        Parameters
        ----------
        robotAddr: the address of the robot for which to change the packet.

        Returns
        -------
        None
        """ 
        id = self.getIdFromAddress(robotAddr)
        if(id >= 0):
            self.flagsTX[id][0] |= (1 << 0)

    def turnOffBackIR(self, robotAddr):
        """Turn off the back IR transmitter on the robot.

        Parameters
        ----------
        robotAddr: the address of the robot for which to change the packet.

        Returns
        -------
        None
        """ 
        id = self.getIdFromAddress(robotAddr)
        if(id >= 0):
            self.flagsTX[id][0] &= ~(1 << 0)

    def getAllProximity(self, robotAddr):
        """Request all the proximity sensors values from one robot at once.

        Parameters
        ----------
        robotAddr: the address of the robot from which receive data.

        Returns
        -------
        List of proximity values (size is 8)
        """

        id = self.getIdFromAddress(robotAddr)
        #print("id = " + str(id))
        if(id >= 0):
            return self.proxValue[id]
        else:
            return None

    def getAllGround(self, robotAddr):
        """Request all the ground sensors values from one robot at once.

        Parameters
        ----------
        robotAddr: the address of the robot from which receive data.

        Returns
        -------
        List of ground values (size is 4)
        """
        id = self.getIdFromAddress(robotAddr)
        if(id >= 0):
            return self.groundValue[id]
        else:
            return None

    def getBatteryPercent(self, robotAddr):
        """Request the charge percentage of the battery.

        Parameters
        ----------
        robotAddr: the address of the robot from which receive data.

        Returns
        -------
        Current charge percentage, the values range is between 0 and 100.
        """
        id = self.getIdFromAddress(robotAddr)
        if(id >= 0):
            if(self.batteryAdc[id] >= 934): # 934 is the measured adc value when the battery is charged
                self.batteryPercent[id] = 100
            elif(self.batteryAdc[id] <= 780): # 780 is the measrued adc value when the battery is discharged
                self.batteryPercent[id] = 0
            else:
                self.batteryPercent[id] = ((self.batteryAdc[id]-780.0)/(934.0-780.0))*100.0

            return self.batteryPercent[id]

    def getAccX(self, robotAddr):
        """Request the raw value of the accelerometer x axis.

        Parameters
        ----------
        robotAddr: the address of the robot from which receive data.

        Returns
        -------
        Current x value, the range is between -64 to 64.
        """       
        id = self.getIdFromAddress(robotAddr)
        if(id >= 0):
            return self.accX[id]
    
    def getAccY(self, robotAddr):
        """Request the raw value of the accelerometer y axis.

        Parameters
        ----------
        robotAddr: the address of the robot from which receive data.

        Returns
        -------
        Current y value, the range is between -64 to 64.
        """
        id = self.getIdFromAddress(robotAddr)
        if(id >= 0):
            return self.accY[id]
              
    def getAccZ(self, robotAddr):
        """Request the raw value of the accelerometer z axis.

        Parameters
        ----------
        robotAddr: the address of the robot from which receive data.

        Returns
        -------
        Current z value, the range is between -64 to 64.
        """
        id = self.getIdFromAddress(robotAddr)
        if(id >= 0):
            return self.accZ[id]
              
    def getSelector(self, robotAddr):
        """Request the selector position.

        Parameters
        ----------
        robotAddr: the address of the robot from which receive data.

        Returns
        -------
        Current selector position, range is between 0 and 15.
        """
        id = self.getIdFromAddress(robotAddr)
        if(id >= 0):
            return self.selector[id]
              
    def setSmallLed(self, robotAddr, ledId, state):
        """Set a new state (on or off) for the small green leds around the robot.

        Parameters
        ----------
        robotAddr: the address of the robot from which receive data.
        ledId: the led to change, from 0 to 7 (0 is in front, then increases clockwise).
        state: 0 to turn off, 1 to turn on.

        Returns
        -------
        None
        """
        id = self.getIdFromAddress(robotAddr)
        if(id >= 0):
            if(state==0):
                self.smallLeds[id] &= ~(1<<ledId)
            else:
                self.smallLeds[id] |= (1<<ledId)
                  
    def getOdomTheta(self, robotAddr):
        """Request the current orientation of the robot computed from the measured speed of the motors.

        Parameters
        ----------
        robotAddr: the address of the robot from which receive data.

        Returns
        -------
        Current orientation of the robot expressed in 1/10 of degree (3600 degrees for a full turn).
        """
        id = self.getIdFromAddress(robotAddr)
        if(id >= 0):
           return self.robTheta[id]
                  
    def getOdomXpos(self, robotAddr):
        """Request the current position (x component) of the robot computed from the measured speed of the motors.

        Parameters
        ----------
        robotAddr: the address of the robot from which receive data.

        Returns
        -------
        Current x position of the robot expressed in millimiters.
        """
        id = self.getIdFromAddress(robotAddr)
        if(id >= 0):
           return self.robXPos[id]
           
    def getOdomYpos(self, robotAddr):
        """Request the current position (y component) of the robot computed from the measured speed of the motors.

        Parameters
        ----------
        robotAddr: the address of the robot from which receive data.

        Returns
        -------
        Current y position of the robot expressed in millimiters.
        """
        id = self.getIdFromAddress(robotAddr)
        if(id >= 0):
           return self.robYPos[id]
                  
    def getLeftMotSteps(self, robotAddr):
        """Request the current raw value of the left motor steps (sum of the measured speed).

        Parameters
        ----------
        robotAddr: the address of the robot from which receive data.

        Returns
        -------
        Current motor steps.
        """
        id = self.getIdFromAddress(robotAddr)
        if(id >= 0):
           return self.leftMotSteps[id]
           
    def getRightMotSteps(self, robotAddr):
        """Request the current raw value of the right motor steps (sum of the measured speed).

        Parameters
        ----------
        robotAddr: the address of the robot from which receive data.

        Returns
        -------
        Current motor steps.
        """    
        id = self.getIdFromAddress(robotAddr)
        if(id >= 0):
           return self.rightMotSteps[id]
           
    def run(self):
        if not self.dev:
            print("Cannot find base station!")
            return
        
        while 1:
            self.transfer_data()

            if(self.payloadId == 255):
                self.payloadId = 0
            else:
                self.payloadId = self.payloadId + 1

            self.currPacketId = self.currPacketId + 1
            if(self.currPacketId*4 >= self.currNumRobots):
                self.currPacketId = 0

            time.sleep(0.004) # 4 ms => transfer @ 250 Hz
    



