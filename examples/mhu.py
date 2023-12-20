import cv2 
import numpy as np

#Leo la imagen de diversas figuras
figuras = cv2.imread("../resources/images/formas1.png")
figuras = cv2.imread("../resources/images/formasP.png")
# figuras = cv2.imread("../resources/pictures/tornillo01.jpeg")
# figuras = cv2.imread("../resources/dataset/internet/arandela06.jpg")
#Creo una imagen apartir de una transformación a grices de la imagen original
grices = cv2.cvtColor(figuras, cv2.COLOR_BGR2GRAY)
#
imgCanny = cv2.Canny(figuras, 10, 50)
image_filtered = cv2.GaussianBlur(figuras, (5, 5), 0)

# Encuentra los contornos en la imagen filtrada por Canny
contours, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgCanny, contours, -1, (0,255,0),3)

# Calcula los momentos de Hu para el primer contorno encontrado
for c in range(len(contours)):
    moments = cv2.moments(contours[c])
    hu_moments = cv2.HuMoments(moments)
    print("Hu Moments de la imagen filtrada por Canny:")
    for i in range(0, 7):
        print(f"Hu Moment {i}: {hu_moments[i]}")



_,th = cv2.threshold(grices,100,255,cv2.THRESH_BINARY_INV)
#Consigo los contornos ordenandolos en cantidad y por jerarquia
contornos,_ = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#dibujo los contornos
cv2.drawContours(figuras, contornos, -1, (0,255,0),3)

# contG,_ = cv2.findContours(image_filtered,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# contC,_ = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


cv2.imshow("imagen filtro gausiano", image_filtered)
cv2.imshow("imagen filtro Canny", imgCanny)

#por cada elemento dentro de  la cantidad de contornos
for conts in range (len(contornos)):     
    #trabajo con el contorno i-ésimo
    cnt = contornos[conts]   
    #consigo su momento de hu
    M = cv2.moments(cnt)
    #muestro su valor
    print(M)
    #Calculo sus coordenadas del centroide en X y en Y
    cX = int (M["m10"]/M["m00"]);cY = int (M["m01"]/M["m00"])
    #imprimo sus valores
    print(cX);print(cY)
    #Dibujo un punto en donde corresponde la coordenada de su centroide
    cv2.circle(figuras,(cX,cY),5,(130,130,130),-1)
    #elipse
    if len(contornos[conts]) > 5:            
        elip = cv2.fitEllipse(contornos[conts])
        print("El valor de elipse es: ", elip)
        # Calcular los ejes mayor y menor de la elipse ajustada
        major_axis = max(elip[1])
        minor_axis = min(elip[1])
        # Calcular la excentricidad
        excentricity = major_axis / minor_axis

        # Imprimir el valor de excentricidad
        print("Excentricidad:", excentricity)
        
    
    #Calculo la longitud de curva del contorno
    per = cv2.arcLength(cnt, True)
    #imprimo la longitud de crva del contorno
    print(per)
    #Aproximo las curvas de contorno por medio de rectas
    approx = cv2.approxPolyDP(cnt, 0.009*per, True)
    #imprimo 
    print(approx);print(len(approx))
    
    #Muestro las ventanas de las imagenes
    cv2.imshow("figuras", figuras);cv2.imshow("th",th)
    #Espero presionar una tecla para pasar a la siguiente iteración
    cv2.waitKey(0)

#limprio las ventanas existentes
cv2.destroyAllWindows()