import time
import cv2
import mss
import pytesseract
import numpy as np
from mss import mss
from PIL import Image
import matplotlib.pyplot as plt

class Vision:
    def __init__(self):
        self.static_templates = {
            'tela': 'prova.jpg',
            'zen': 'zen.png'
        }
        self.templates = { k: cv2.imread(v, 0) for (k, v) in self.static_templates.items() }
        self.monitor = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
        self.screen = mss()
        self.frame = None

    def take_screenshot(self):
        sct_img = self.screen.grab(self.monitor)
        img = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
        img = np.array(img)
        img = self.convert_rgb_to_bgr(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img_gray

    def get_image(self, path):
        return cv2.imread(path, 0)

    def bgr_to_rgb(self, img):
        b,g,r = cv2.split(img)
        return cv2.merge([r,g,b])

    def convert_rgb_to_bgr(self, img):
        return img[:, :, ::-1]

    def match_template(self, img_grayscale, template, threshold=0.9):
        """
        Matches template image in a target grayscaled image
        """
        res = cv2.matchTemplate(img_grayscale, template, cv2.TM_CCOEFF_NORMED)
        matches = np.where(res >= threshold)
        return matches

    def find_template(self, name, image=None, threshold=0.9):
        if image is None:
            if self.frame is None:
                self.refresh_frame()

            image = self.frame

        return self.match_template(
            image,
            self.templates[name],
            threshold
        )

    def scaled_find_template(self, name, image=None, threshold=0.9, scales=[1.0, 0.9, 1.1]):
        if image is None:
            if self.frame is None:
                self.refresh_frame()

            image = self.frame

        initial_template = self.templates[name]
        for scale in scales:
            scaled_template = cv2.resize(initial_template, (0,0), fx=scale, fy=scale)
            matches = self.match_template(
                image,
                scaled_template,
                threshold
            )
            if np.shape(matches)[1] >= 1:
                return matches
        return matches

    def refresh_frame(self):
        self.frame = self.take_screenshot()

imBGR = cv2.imread('prova.jpg')
im = cv2.cvtColor(imBGR, cv2.COLOR_BGR2RGB)
imHSV = cv2.cvtColor(imBGR, cv2.COLOR_BGR2HSV)
imgray = cv2.cvtColor(imBGR, cv2.COLOR_BGR2GRAY)

def locator(img, ob):
    result = cv2.matchTemplate(ob, img, cv2.TM_SQDIFF_NORMED)
    mn,_,mnLoc,_ = cv2.minMaxLoc(result)
    print(mn, mnLoc)
    return mnLoc


def highlighter(img, mnLoc, ob):
    MPx,MPy = mnLoc
    trows,tcols = small_image.shape[:2]
    return cv2.rectangle(img, (MPx,MPy),(MPx+tcols,MPy+trows),(255,0,0),2)

def aaa(im):
    plt.imshow(im)
    plt.show()



def show_by_its_eyes(img_in_np_array, title='q to escape'):
    cv2.imshow(title, img_in_np_array)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        cv2.waitKey(1)

def takes_screenshots(top_left_corner_and_size):
    with mss() as sct:
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
