import cv2
import math
import time
import pytesseract
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import EV3BT
import serial
# from VideoCapture import Device


cap = cv2.VideoCapture(1)   # /dev/video0
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # turn the autofocus off
cap.set(3,720)
cap.set(4,480)

crop_top = 160
crop_bot = 120
crop_lef = 160
crop_rig = 160

target_height = 400

def cropped(frame):
  (h, w, _) = frame.shape
  
  return frame[crop_top:h - crop_bot, crop_lef:w - crop_rig]


def treshed(frame):
  # ret, thresh = cv2.threshold(frame,127,255,cv2.THRESH_BINARY)
  thresh = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
  return thresh


def resize(frame):
  (w, h, _) = frame.shape
  f = float(target_height) / h
  return cv2.resize(frame, None, fx=f, fy=f, interpolation = cv2.INTER_CUBIC)


def kmeans(frame, K = 3):
  # Apply KMeans 
  Z = frame.reshape((-1,3))

  # convert to np.float32
  Z = np.float32(Z)

  # define criteria, number of clusters(K) and apply kmeans()
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 6, 1.0)
  ret,label,center=cv2.kmeans(Z,K,None,criteria,6,cv2.KMEANS_RANDOM_CENTERS)

  # Now convert back into uint8, and make original image
  new_centers = np.uint8(center)
  res = new_centers[label.flatten()]
  res2 = res.reshape((frame.shape))
  return (res2, center)

def colorDist(A, B, l=3):
  sumError = 0
  for k in range(l):
    sumError = sumError + pow(abs(A[k] - B[k]), 2)
  sumError = math.sqrt(sumError)
  return sumError
  
  # dh = min(abs(B[0]-A[0]), 360-abs(B[0]-A[0])) / 180.0
  # ds = abs(B[1]-A[1])
  # dv = abs(B[2]-A[2]) / 255.0
  # return math.sqrt(dh*dh+ds*ds+dv*dv)

def isGray(color):
  avg = (color[0] + color[1] + color[2]) / 3
  sumError = 0
  for v in color:
    sumError = sumError + pow(abs(v - avg), 2)
  sumError = math.sqrt(sumError)

  if (sumError < 20):
    return True

  return False

colorMap = [
  # RGB
    # [27, 55, 35, "green"],
    # [151, 189, 126, "green"],
    # [110, 190, 216, "yellow"],
    # [131, 190, 204, "yellow"],
    # [41, 140, 179, "yellow"],
    # [90, 170, 200, "yellow"],
    [170, 250, 250, "yellow"],
    [90, 180, 220, "yellow"],

    [163, 118, 133, "purple"],
    # [39, 53, 143, "red"],
    [39, 30, 170, "red"],
    [60, 76, 200, "red"],

    [138, 65, 22, "blue"],
    [170, 230, 210, "green"],
    [130, 150, 50, "green"],
    # [85, 150, 230, "orange"],

  # HSV
  # [27, 55, 35, "green"],
  # [151, 189, 126, "green"],
  # [110, 190, 216, "yellow"],
  # [240, 100, 23, "yellow"],
  # [210, 50, 40, "green"],
  # [240, 100, 100, "purple"],
  # [130, 190, 170, "red"],
  # [39, 53, 143, "red"],
  # [210, 210, 120, "blue"],
  # [85, 150, 230, "orange"],
]


def labelOfColor(color):
  if isGray(color):
    return None

  bestError = 99999
  bestColor = None

  for check in colorMap:
    error = colorDist(color, check)
    if (bestError > error):
      bestError = error
      bestColor = check

  if bestError > 70:
    return None

  # print("Best color: " + bestColor[3] + " error: "+str(bestError))
  # print(bestColor)
  return bestColor[3]


def bestColors(centers):
  colors = []
  for center in centers:
    color = labelOfColor(center)

    if color is not None:
      colors.append(color)

  return colors


# Connect
# EV3 = serial.Serial('/dev/cu.WOL-SerialPort')
# s = EV3BT.encodeMessage(EV3BT.MessageType.Text, 'abc', 'Eat responsibly')
# print(EV3BT.printMessage(s))
# EV3.write(s)
# time.sleep(1)
# EV3.close()

import time
import EV3BT
import serial
millis = lambda: int(round(time.time() * 1000))


# Connect
EV3 = serial.Serial('/dev/cu.WOL-SerialPort')

def redirectEsteira(section, lr):
  s = EV3BT.encodeMessage(EV3BT.MessageType.Text, section, lr)
  EV3.write(s)

redirections = []
def checkAndSendRedirections():

  now = millis()
  for redirect in redirections:
    if redirect['at_time'] <= now:
      print('Sending redirection: ' + redirect['color'] + ' section: ' + redirect['section'] + ' lr: '+redirect['lr'])
      redirectEsteira(redirect['section'], redirect['lr'])
      redirections.remove(redirect)

def scheduleColorRedirection(color):
  data = {}
  lr = None
  delta = 0
  section = None
  if color == 'red':
    lr = 'left'
    section = 'ab'

  if color == 'green':
    lr = 'right'
    section = 'ab'

  if color == 'yellow':
    lr = 'left'
    section = 'cd'
    delta = 900

  if color == 'blue':
    lr = 'right'
    section = 'cd'
    delta = 900

  if lr is None:
    return

  data['lr'] = lr
  data['color'] =  color
  data['at_time'] =  millis() + delta
  data['section'] =  section
  redirections.append(data)

lastColor = None
lastColorCounter = 0

while True:
  ret, frame = cap.read()

  # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  crop = cropped(frame)
  resized = resize(crop)
  # thresh = treshed(crop)

  final = resized

  cv2.imshow("frame", frame)
  cv2.imshow("croped", crop)
  cv2.imshow("resized", resized)

  if not ret:
    break


  # Detect dominat colors in image
  (image, centers) = kmeans(final, 3)

  # Filter and get color of part
  colors = bestColors(centers)

  # Get best color
  color = None
  if len(colors) > 0:
    color = colors[0]

  if color:
    if (lastColor == color):
      lastColorCounter += 1
    else:
      lastColorCounter = 0
      lastColor = color

    cv2.imshow('kmeans', image)
    if lastColorCounter == 2:
      print("\n\nColor detected: " + color)
      scheduleColorRedirection(color)
  else:
    lastColor = None
    lastColorCounter = 0
    # print("No color!")

  checkAndSendRedirections()

  # Check for user keys input
  key = cv2.waitKey(1) & 0xFF
  if key == ord('q'):
      break



  # img = Image.fromarray(final)
  # img.show()

  # Whitelisted characters
  # az = "abcdefghijklmnopqrstuvwxyz"
  # AZ = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  # whitelist = az+AZ

  # # out = pytesseract.image_to_string(img, config="-psm 8")
  # # out = pytesseract.image_to_string(img, config="-c tessedit_char_whitelist=" + whitelist + " -psm 8")
  # out = pytesseract.image_to_string(img, config="-c tessedit_char_whitelist=\"" + whitelist + "\" --user-words=words.txt --psm=8")
  # print("Detected: " + out)



# came = Device()
# came.setResolution(320, 240)

# img = cam.getImage()
# # img = Image.open('nome_da_imagem.jpg')

# out = pytesseract.image_to_string(img)

# print(out)