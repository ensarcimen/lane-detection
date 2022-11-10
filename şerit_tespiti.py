import cv2
import numpy as np

def canny(img):
    if img is None:
        cap.release()
        cv2.destroyAllWindows()
        exit()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 170)
    return canny

def mask_of_img(img):
    height = img.shape[0]
    polygons = np.array([[(300,height),(1100,height),(700,480)]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,polygons,255)
    masked_img = cv2.bitwise_and(img,mask)
    return masked_img

def houghLines(cropped_canny):
    return cv2.HoughLinesP(cropped_canny, 2, np.pi / 180, 100,np.array([]), minLineLength=40, maxLineGap=5)

def addWeighted(frame, line_image):
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)

def display_lines(img, lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)

    return line_image


def make_points(img, line):
    slope, intercept = line
    y1 = int(img.shape[0])
    y2 = int(y1 * 4 / 5)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(img,lines):
    left_fit = []
    right_fit = []
    try:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            parameters = np.polyfit((x1,x2),(y1,y2),1, rcond= 1.6)
            slope = parameters[0]
            intercept = parameters[1]
            if slope <0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line = make_points(img, left_fit_average)
        right_line = make_points(img, right_fit_average)
        averaged_lines = [left_line, right_line]
        return averaged_lines
    except:
      return None


cap = cv2.VideoCapture("yolvideosu.mp4")
while (cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_canny = mask_of_img(canny_image)
    lines = houghLines(cropped_canny)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = addWeighted(frame, line_image)
    cv2.imshow("result", combo_image)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()