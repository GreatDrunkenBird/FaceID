import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture("Contagem_exercicio.mp4")
# cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

hue_list = [0, 109, 74, 98]


def search_contours(mask):
    mask = mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    i = 0

    for contour in contours:
        if i == 0:
            i = 1
            continue

        M = cv2.moments(contour)
        if M['m00'] != 0.0:
            x = int(M['m10'] / M['m00'])
            y = int(M['m01'] / M['m00'])
        else:
            x, y = 0, 0

        if len(contour) == 3:
            cv2.drawContours(video, [contour], -1, (0, 255, 0), 2)
            cv2.putText(video, 'Triangle', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        elif len(contour) == 4:
            cv2.drawContours(video, [contour], -1, (0, 255, 0), 2)
            cv2.putText(video, 'Quadrilateral', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        elif len(contour) > 10:
            cv2.drawContours(video, [contour], -1, (0, 255, 0), 2)
            cv2.putText(video, 'circle', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return mask


while cap.isOpened():
    ret, video = cap.read()
    if ret:
        hsv = cv2.cvtColor(video, cv2.COLOR_BGR2HSV)
        for hue in hue_list:
            lower_hue = hue - 90
            upper_hue = hue + 90

            lower_hsv = np.array([lower_hue, 50, 20])
            upper_hsv = np.array([upper_hue, 255, 255])

            mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

            b_a = search_contours(mask)

            cv2.imshow('Video', video)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
