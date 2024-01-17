import face_recognition as fr
import cv2
import numpy as np
import os
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image

path = "./train/"
image_cv = []
known_inds = []
known_names = []
known_name_encodings = []

images = os.listdir(path)
for img in images:
    train_image = fr.load_image_file(path + img)
    image_path = path + img
    encoding = fr.face_encodings(train_image)[0]

    known_name_encodings.append(encoding)
    known_inds.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())


for ind in known_inds:
    nome = ind.split('_')[0]
    known_names.append(nome)


def select_img():
    global image_cv
    Tk().withdraw()
    filename = askopenfilename()
    image_cv = cv2.imread(filename)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    display_img()
    return


def display_img():
    global image_cv
    blue, green, red = cv2.split(image_cv)
    image = cv2.merge((red, green, blue))
    im = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=im)
    render = imgtk
    image = Label(root, image=render, height=600, width=800)
    image.image = render
    image.place(x=0, y=0)
    return


def face_rec():
    global image_cv
    face_locations = fr.face_locations(image_cv)
    face_encodings = fr.face_encodings(image_cv, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_name_encodings, face_encoding)
        name = ""

        face_distances = fr.face_distance(known_name_encodings, face_encoding)
        best_match = np.argmin(face_distances)

        if matches[best_match]:
            name = known_names[best_match]

        cv2.rectangle(image_cv, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(image_cv, (left, bottom - 15), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        identified = cv2.putText(image_cv, name, (left + 6, bottom - 6), font, 1.0, (0, 0, 0), 1)
        image_cv = identified
        display_img()
    return


# User interface
root = Tk()
root.geometry("1000x600")
root.title("Face Recognition")

# load = ImageTk.PhotoImage(Image.open("elon.jpg"))


select_button = Button(root, text="Select Image", command=select_img)
select_button.place(x=847, y=80)

indf_button = Button(root, text="Identify Face", command=face_rec)
indf_button.place(x=847, y=120)

# Exit and close the app
root.mainloop()
