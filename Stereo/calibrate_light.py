import cv2
import numpy as np

def calibrate_light_test(imageDirectory, maskDirectory, lightDirectory, numLights, imageName, showImages):
    calLitOutputLoc = 'CalibratedLightOutputs/calibratedLight.txt'

    if imageDirectory[-1] != '/':
        fileName = imageDirectory + '/'
    else:
        fileName = imageDirectory

    
    fileName = fileName + imageName
    maskFileName = maskDirectory + imageName + 'mask.tiff'
    
    circle = cv2.imread(maskFileName, cv2.IMREAD_GRAYSCALE)

    maxval = np.max(circle)
    circleRow, circleCol = np.where(circle == maxval)
    maxRow = max(circleRow)
    minRow = min(circleRow)
    maxCol = max(circleCol)
    minCol = min(circleCol)
    xc = float((maxCol + minCol) / 2)
    yc = float((maxRow + minRow) / 2)
    center = [xc, yc]
    radius = float((maxRow - minRow) / 2)

    R = [0, 0, 1.0]
    L = []

    rows, cols = circle.shape
    card = ['n', 'e', 's', 'w']


    for i in range(numLights):
        id = card[i]
        imgFileName = fileName + id + '.tiff'
        image = cv2.imread(imgFileName, cv2.IMREAD_GRAYSCALE)

        if showImages:
            cv2.imshow("image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        for n in range(rows):
            for m in range(cols):
                if circle[n, m] != 255:
                    image[n, m] = 0

        for n in range(rows):
            for m in range(cols):
                if image[n, m] != 255:
                    image[n, m] = 0

        for n in range(rows):
            for m in range(cols):
                pixelR = np.sqrt((n - yc)**2 + (m - xc)**2)
                if pixelR >= (radius - 5):
                    image[n, m] = 0

        maxval = np.max(image)
        pointRow, pointCol = np.where(image == maxval)
        nSize = len(pointRow)
        px = np.sum(pointCol) / float(nSize)
        py = np.sum(pointRow) / float(nSize)
        Nx = px - xc
        Ny = -(py - yc)
        Nz = np.sqrt(radius**2 - Nx**2 - Ny**2)
        normal = np.array([Nx, Ny, Nz])
        normal = normal / radius
        NR = normal.dot(R)
        L.append(2 * NR * normal - R)

    with open(lightDirectory + imageName + 'light.txt', 'w') as fid:
        #fid.write(str(numLights) + '\n')
        for row in L:
            fid.write('%10.5f %10.5f %10.5f\n' % (row[0], row[1], row[2]))