import time
import EV3BT
import serial

# Connect
EV3 = serial.Serial('/dev/cu.WOL-SerialPort')

def redirect(section, lr):
  s = EV3BT.encodeMessage(EV3BT.MessageType.Text, section, lr)
  EV3.write(s)

# print(EV3BT.printMessage(s))
for i in range(5):
  print("sent..")
  redirect('ab', 'left')
  time.sleep(0.5)
  redirect('cd', 'right')
  time.sleep(1.5)
  redirect('ab', 'right')
  time.sleep(0.5)
  redirect('cd', 'left')
  time.sleep(1.5)

EV3.close()