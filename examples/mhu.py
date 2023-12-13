import cv2 
import numpy as np

figuras = cv2.imread("../resources/images/circulo.png")
grices = cv2.cvtColor(figuras, cv2.COLOR_BGR2GRAY)
ret,th = cv2.threshold(grices,200,255,cv2.THRESH_BINARY_INV)
contornos,jerarquia = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

contorno1 = contornos[0]
M = cv2.moments(contorno1)
# print(contornos)
# print(contorno1)
print(M)
cX = int (M["m01"]/M["m00"])
cY = int (M["m10"]/M["m00"])
print(cX);print(cY)
cv2.circle(figuras,(cX,cY),5,(130,130,130),-1)

cv2.imshow("figuras", figuras)
cv2.imshow("th",th)

print("termino")
cv2.waitKey(0)