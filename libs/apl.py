import cv2
import numpy
from matplotlib.path import Path
import matplotlib.pyplot as plt

def ColorCorrection(imgBench, img):
    ## step 1. find the NA part
    channel_1 = img[:, :, 0]
    channel_2 = img[:, :, 1]
    channel_3 = img[:, :, 2]
    img1 = numpy.abs(channel_1 - channel_2) + numpy.abs(channel_2 - channel_3) + numpy.abs(channel_1 - channel_3)
    img2 = (channel_1 > 250).astype(int) * (channel_2 > 250).astype(int) * (channel_3 > 250).astype(int)
    img2 = 1 - img2
    res1 = img1 + img2
    ## data: 1, na: 0
    res1 = (res1 > 0).astype(int)
    ## step 2. find shadow part
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgGray, 87, 255, cv2.THRESH_BINARY)
    blur = cv2.blur(thresh, (1000, 1000))
    ret2, thresh2 = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)
    res2 = thresh2 * res1 // 255 ## light: 1, other: 0
    res3 = (255 - thresh2) * res1 // 255## shadow: 1, other 0
    
    ## step 3. color correction
    imgLight = numpy.zeros(img.shape, dtype = numpy.uint8)
    imgShade = numpy.zeros(img.shape, dtype = numpy.uint8)
    imgNew = numpy.zeros(img.shape, dtype = numpy.uint8)
    
    s1 = res2.sum()
    s2 = res3.sum()
    
    for i in range(3):
        vecBench = imgBench[:, :, i].flatten()
        vecLight = img[res2 > 0, i].flatten()
        vecShade = img[res3 > 0, i].flatten()
        
        
        ## correct the light part
        if s1 > 0:
            hist1, bin_edge1 = numpy.histogram(vecLight, bins=numpy.arange(256), density=True)
            quantile1 = numpy.cumsum(hist1) * 100
            quantile1[quantile1 > 100] = 100
            quantile1[quantile1 < 0] = 0

            q1 = numpy.percentile(vecBench, quantile1)
            q1 = numpy.append(q1, 255)
            for row in range(img.shape[0]):
                original = img[row, :, i]
                imgLight[row, :, i] = q1[original]
        
        ## correct the shadow part
        if s2 >0:
            hist2, bin_edge2 = numpy.histogram(vecShade, bins=numpy.arange(256), density=True)
            quantile2 = numpy.cumsum(hist2) * 100
            quantile2[quantile2 > 100] = 100
            quantile2[quantile2 < 0] = 0

            q2 = numpy.percentile(vecBench, quantile2)
            q2 = numpy.append(q2, 255)
            for row in range(img.shape[0]):
                original = img[row, :, i]
                imgShade[row, :, i] = q2[original]
                
        imgNew[:, :, i] = res2 * imgLight[:, :, i] + res3 * imgShade[:, :, i]
    return imgNew

## poly2mask function is from stackoverflow: https://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask
def poly2mask(nx, ny, poly_verts):
    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    x, y = numpy.meshgrid(numpy.arange(nx), numpy.arange(ny))
    x, y = x.flatten(), y.flatten()
    points = numpy.vstack((x,y)).T
    path = Path(poly_verts)
    grid = path.contains_points(points)
    grid = grid.reshape((ny,nx))
    return grid

        

