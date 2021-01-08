from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


# First_derivative for sharping
def First_derivative(fr) -> list:
    cont = list()
    for i in range(0, len(fr) - 1, 1):
        cont.append(fr[i + 1] - fr[i])
    return cont


# second_derivative for sharping
def second_derivative(fr) -> list:
    cont = list()
    for i in range(1, len(fr) - 1, 1):
        cont.append((fr[i + 1] + fr[i - 1]) - (2 * fr[i]))
    return cont


# highboost_filtering for sharping
def highboost_filtering(min, max, source):
    blurr = mean_linear_smoothong(min, max, source)
    row, col = source.shape
    mask = np.array(np.zeros((row, col)), "uint8")
    finnal = np.array(np.zeros((row, col)), "uint8")
    for i in range(0, row, 1):
        for j in range(0, col, 1):
            mask[i][j] = int(source[i][j]) - int(blurr[i][j])

    for i in range(0, row, 1):
        for j in range(0, col, 1):
            finnal[i][j] = int(source[i][j]) + int(3 * mask[i][j])
    cv.imshow("highboost_filtering", finnal)
    cv.waitKey()
    cv.destroyAllWindows()


# mean_linear_smoothong for blurr
def mean_linear_smoothong(min, max, source):
    opencvImage = source.copy()
    row, col = source.shape
    for i in range(min, row - min):

        for j in range(min, col - min):
            sum = 0

            for k in range(-min, max):
                for l in range(-min, max):
                    sum = sum + source[i + k][j + l]

            opencvImage[i][j] = int(sum / 9)
    return opencvImage


# Laplace for sharping
def Laplace(min, max, r, c, source):
    opencvImage = source.copy()
    filterlow = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    filterhigh = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    filtersoble = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    for i in range(min, r - min, 1):
        print((i / r) * 100)
        for j in range(min, c - min, 1):
            sum = 0

            for k in range(-min, max):
                for l in range(-min, max):
                    sum = sum + (source[i + k][j + l] * filterlow[k + 1][l + 1])
                # print((source[i + k][j + l][0] * filterlow[k + 1][l + 1]))

            opencvImage[i][j] = sum
            opencvImage[i][j] = sum
            opencvImage[i][j] = sum
    cv.imshow("Laplace", opencvImage)
    cv.waitKey()
    cv.destroyAllWindows()


# read image
image = Image.open("Untitled.png")
image = image.convert("L")
image2 = cv.imread("Untitled.png", cv.IMREAD_GRAYSCALE)

# histogram and caculate First_derivative and second_derivative
freq = image.histogram()
firstD = First_derivative(freq)
SD = second_derivative(freq)

# row and col in image
row, col = image2.shape
# execute two method sharping
#Laplace(1, 2, row, col, image2)
#highboost_filtering(1, 2, image2)

fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle('Vertically stacked subplots')
ax1.plot(freq)
ax2.plot(firstD)
ax3.plot(SD)
plt.show()
Image._show(image)
