from facenet_pytorch import MTCNN
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
import os
import piexif


def locate_and_blur (image, boxes):
    for box in boxes:
        # Print the location of each face in this image
        top, right, bottom, left = int(box[1]), int(box[2]), int(box[3]), int(box[0])

        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]

        #blur image and save to original
        image_blur = cv2.GaussianBlur(face_image, (99, 99), 30)
        image[top:bottom, left:right] = image_blur

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



PROCESSED_DIR = './processed/'
DIR_PATH = './photos/'
PROCESSED_FILE = ''
file_count = 0
error_log = open("error_log.txt", "a")


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print('Running on device: {}'.format(device))


mtcnn = MTCNN(keep_all=True, device=device)


for filename in os.listdir(DIR_PATH):
    try:
        print ("processing " + filename)
        photo_full_path = DIR_PATH + filename

        # load exif data, without piexif lib
        im = Image.open(photo_full_path)
        img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # exif = im.info['exif']

        # load exif data
        exif_dict = im.info.get("exif")
        if exif_dict:
            exif = piexif.load(im.info["exif"])
        else:
            exif = None

        boxes, _ = mtcnn.detect(img)

        # TODO: fix parsing bug
        if boxes is not None:
            locate_and_blur(img, boxes)

        cv2.imshow('img', img)
        cv2.waitKey(0)

        pil_image = Image.fromarray(img)

        # PROCESSED_FILE = "../processed/" + str(file_count)
        PROCESSED_FILE = PROCESSED_DIR + str(file_count) + "_blured_"  + filename
        print("saving file " + PROCESSED_FILE)

        if exif is not None:
            pil_image.save(PROCESSED_FILE, exif=exif)
        else:
            pil_image.save(PROCESSED_FILE)

        file_count += 1

    except Exception as e:
        print (e)
        # append error to file
        error_log.write(str(e) + " on " + filename + "\n")

error_log.close()
