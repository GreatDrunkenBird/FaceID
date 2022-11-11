import face_recognition as fr
import cv2
import numpy as np
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

path = "./train/"

known_inds = []
known_names = []
known_name_encodings = []

images = os.listdir(path)
for img in images:
    image = fr.load_image_file(path + img)
    image_path = path + img
    encoding = fr.face_encodings(image)[0]

    known_name_encodings.append(encoding)
    known_inds.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())


for ind in known_inds:
    nome = ind.split('_')[0]
    nivel = ind.split('_')[-1]
    known_names.append(nome)

Tk().withdraw()
filename = askopenfilename()
image = cv2.imread(filename)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

face_locations = fr.face_locations(image)
face_encodings = fr.face_encodings(image, face_locations)

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = fr.compare_faces(known_name_encodings, face_encoding)
    name = ""

    face_distances = fr.face_distance(known_name_encodings, face_encoding)
    best_match = np.argmin(face_distances)

    if matches[best_match]:
        name = known_names[best_match]

    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.rectangle(image, (left, bottom - 15), (right, bottom), (0, 255, 0), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (0, 0, 0), 1)
    print(f"Nome: {nome}, Nível de acesso: {nivel}")


cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
