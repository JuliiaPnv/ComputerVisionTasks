import cv2
import numpy as np

#Вывести исходное изображение
source = cv2.imread('cat.jpg')
cv2.imshow('Source', source)

#Перевести изображение в grayscale
grayImage = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
cv2.imwrite('result.jpg ', grayImage)
cv2.imshow('Gray', grayImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Перевести изображение в hsv
hsvImage = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
cv2.imwrite('result.jpg ', hsvImage)
cv2.imshow('HSV', hsvImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Отразить изображение по правой границе
flipCode = 1
result = cv2.flip(source, flipCode)
cv2.imwrite('result.jpg ', result)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Отразить изображение по нижней границе
flipCode = 0
result = cv2.flip(source, flipCode)
cv2.imwrite('result.jpg ', result)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Сместить изображение на 10 пикселей вправо
height, width = source.shape[:2]
size = (width, height)
quarterHeight, quarterWidth = 0, 10
translationMatrix = np.float32([[1, 0, quarterWidth], [0, 1, quarterHeight]])
result = cv2.warpAffine(source, translationMatrix, size)
cv2.imwrite('result.jpg ', result)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Сделать размытие изображения
kernel = (5, 5)
averaging = cv2.blur(source, kernel)
gausBlur = cv2.GaussianBlur(source, kernel, 0)
cv2.imwrite('averaging.jpg', averaging)
cv2.imwrite('gausBlur.jpg', gausBlur)
cv2.imshow('Averaging', averaging)
cv2.imshow('Gaussian Blurring', gausBlur)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Изменить яркость изображения
def increaseBrightness(img, value):
    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsvImage)
    limit = 255 - value
    v[v > limit] = 255
    v[v <= limit] += value
    resultHsv = cv2.merge((h, s, v))
    result = cv2.cvtColor(resultHsv, cv2.COLOR_HSV2BGR)
    return result

result60 = increaseBrightness(source, 60)
result90 = increaseBrightness(source, 90)
cv2.imwrite('result60.jpg', result60)
cv2.imwrite('result90.jpg', result90)
cv2.imshow('Result 60', result60)
cv2.imshow('Result 90', result90)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Изменить контрастность изображения
def increaseContrast(img, clipLim):
    labImage = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lChannel, a, b = cv2.split(labImage)
    gridSize = (8, 8)
    clahe = cv2.createCLAHE(clipLimit=clipLim, tileGridSize=gridSize)
    cl = clahe.apply(lChannel)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

result2 = increaseContrast(source, 2)
result8 = increaseContrast(source, 8)
cv2.imwrite('result2.jpg', result2)
cv2.imwrite('result8.jpg', result8)
cv2.imshow('Result 2', result2)
cv2.imshow('Result 8', result8)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Применить операцию эрозии к изображению
kernelSize = (5, 5)
kernel = np.ones(kernelSize, np.uint8)
erosion = cv2.erode(source, kernel, iterations=1)
cv2.imwrite('result.jpg', erosion)
cv2.imshow('Erosion', erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Применить операцию диляции к изображению
kernelSize = (5, 5)
kernel = np.ones(kernelSize, np.uint8)
dilation = cv2.dilate(source, kernel, iterations=1)
cv2.imwrite('result.jpg', dilation)
cv2.imshow('Dilation', dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()

