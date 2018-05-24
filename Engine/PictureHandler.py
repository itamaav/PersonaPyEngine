import ntpath
import urllib.request
import cv2

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class PictureHandler:

    """
        Initialize The PictureHandler Class

        parameters:

            * url - original url
            * imgInstance - hold the image for all of the analysis process
            * image_org_name - extract the name file from he path and the parameters if exist

    """

    def __init__(self , url_to_image):

        print()
        print("PictureHandler : Building new instance !", end="\n\n")

        self.url = url_to_image

        try:
            # Extract file name from path
            if self.url.find("?") != -1:
                self.image_org_name = self.url[:self.url.find("?")]
                self.image_org_name = ntpath.basename(self.image_org_name)
            else:
                self.image_org_name = ntpath.basename(self.url)

            # Save the image downloaded by the name we get
            urllib.request.urlretrieve(self.url, "/home/personaitaben/PersonaPyEngine/downloads/" + self.image_org_name)

            # Open the image from the directory after save
            self.imgInstance = cv2.imread("/home/personaitaben/PersonaPyEngine/downloads/" + self.image_org_name)

            self.img_saved_path = "/home/personaitaben/PersonaPyEngine/downloads/" + self.image_org_name

            self.image_default_folder = "/home/personaitaben/PersonaPyEngine/predict/" + self.image_org_name[:self.image_org_name.find(".")]

        except Exception as e:
            print("Error : wrong url or wrong input type !")

    def create_folder(self, directory = ""):

        if directory != "":
            if directory.find(".") != -1:
                self.image_default_folder = "/home/personaitaben/PersonaPyEngine/predict/" + directory[:directory.find(".")]
            else:
                self.image_default_folder = "/home/personaitaben/PersonaPyEngine/predict/" + directory

        try:
            if not os.path.exists(self.image_default_folder):

                os.makedirs(self.image_default_folder)

                print("Create new directory : ", self.image_default_folder , "!")
                return 1
            else:
                print("The directory already exist !")
                return 0
        except OSError:
            print('Error: Creating directory. ' + self.image_default_folder)
