
import cv2

import urllib.request

class PictureHandler:

    def __init__(self):

        print()
        print("PictureHandler : Building new instance !", end="\n\n")

        self.url = "https://firebasestorage.googleapis.com/v0/b/persona-tech.appspot.com/o/Photos%2F41579eb9-" \
                   "2a76-4e86-b4fa-ef64a21ba9ca.jpg?alt=media&token=692bb276-2a0c-4ec8-8e5f-7a5dbf729ba6"

        urllib.request.urlretrieve(self.url, "../predict/predict1.jpg")

        img = cv2.imread("../predict/predict1.jpg")

        cv2.imshow("Image d", img)
        cv2.waitKey(3000)




