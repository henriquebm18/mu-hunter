import time
import cv2
import mss
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

def show_by_its_eyes(img_in_np_array, title='q to escape'):
    cv2.imshow(title, img_in_np_array)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        cv2.waitKey(1)

def takes_screenshots(top_left_corner_and_size):
    with mss.mss() as sct:
        monitor = {"top": top_left_corner_and_size[0],
                   "left": top_left_corner_and_size[1],
                   "width": top_left_corner_and_size[2],
                   "height": top_left_corner_and_size[3]
                   }
        img = numpy.array(sct.grab(monitor))
        return img

def captch_ex(file_name):
    img = cv2.imread(file_name)

    img_final = cv2.imread(file_name)
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
    image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
    ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY)  # for black text , cv.THRESH_BINARY_INV
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    dilated = cv2.dilate(new_img, kernel, iterations=9)
    # dilate , more the iteration more the dilation

    # for cv2.x.x

    _, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # findContours returns 3 variables for getting contours

    # for cv3.x.x comment above line and uncomment line below

    #image, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)


    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # Don't plot small false positives that aren't text
        if w < 35 and h < 35:
            continue

        # draw rectangle around contour on original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

        '''
        #you can crop image and send to OCR  , false detected will return no text :)
        cropped = img_final[y :y +  h , x : x + w]

        s = file_name + '/crop_' + str(index) + '.jpg'
        cv2.imwrite(s , cropped)
        index = index + 1

        '''
    # write original image with added contours to disk
    cv2.imshow('captcha_result', img)
    cv2.waitKey()




if __name__ == "__main__":
    print('aaaaa')
    ## top left corner and size
    # c_n_size = [23, 850, 500, 500]
    # psc = takes_screenshots(c_n_size)
    # import numpy as np
    # import sys
    #%matplotlib inline
    im = cv2.imread('prova.jpg')
    # im = cv2.imread('IMG_FILENAME',0)
    h,w = im.shape[:2]
    print(im.shape)
    plt.imshow(im,cmap='gray')
    plt.show()

    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    plt.imshow(hsv)
    plt.show()
    lower = np.array([0, 0, 218])
    upper = np.array([157, 54, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
    dilate = cv2.dilate(mask, kernel, iterations=5)
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        ar = w / float(h)
        if ar < 5:
            cv2.drawContours(dilate, [c], -1, (0,0,0), -1)
    result = 255 - cv2.bitwise_and(dilate, mask)
    # plt.imshow(result,cmap='gray')
    # plt.show()
    data = pytesseract.image_to_string(result, lang='eng',config='--psm 6')
    print(data)

    # show_by_its_eyes(psc)
    # # captch_ex(psc)
    # img2gray = cv2.cvtColor(psc, cv2.COLOR_BGR2GRAY)
    # show_by_its_eyes(img2gray)
    #
    # ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
    # # image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
    # # ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY)