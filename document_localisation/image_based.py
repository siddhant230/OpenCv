import cv2
import imutils
import numpy as np
from skimage.filters import threshold_otsu, threshold_local
from skimage import io
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-s", "--src", type=str, help="path to the source image")
args = parser.parse_args()
img_path = args.src
import cv2
import imutils
import numpy as np
from skimage.filters import threshold_otsu, threshold_local
from skimage import io


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def threshold(warp):
    warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warp, offset=10, method='gaussian', block_size=11)
    val = (warp > T).astype('uint8') * 255

    return val


def apply_transform(gray):
    filter = cv2.GaussianBlur(gray, (3, 3), 0)
    ret, thresh = cv2.threshold(filter, 170, 255, 0)
    #cv2.imwrite('test_images/thresh.png', thresh)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # cv2.imwrite('test_images/check.png', gray)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    scr = []
    if len(cnts) > 0:
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.1 * peri, True)

            if len(approx) == 4:
                scr = approx
                break

        if len(scr) > 0:
            cv2.drawContours(img, [scr], -1, (0, 255, 0), 2)

            warp = four_point_transform(orig, scr.reshape(4, 2))
            return warp
    return None


def threshold(warp):
    warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warp, offset=10, method='gaussian', block_size=11)
    val = (warp > T).astype('uint8') * 255

    return val


def sharpen(img):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    opt = cv2.filter2D(img, -1, kernel)

    return opt


def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


if __name__ == "__main__":
    img = cv2.imread(img_path)
    org = img.copy()
    orig = img.copy()
    img = adjust_gamma(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    opt = apply_transform(gray)

    if opt.shape[0] > opt.shape[1]:
        opt = cv2.rotate(opt, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if opt is not None:

        otsu = threshold(opt)
        sharp = sharpen(opt)

        save_name = img_path.split('.')[0]
        cv2.imwrite(save_name + '_output.png', opt)
        cv2.imwrite(save_name + '_otsu.png', otsu)
        cv2.imwrite(save_name + '_sharp.png', sharp)

    else:
        print('Nothing detected...')
