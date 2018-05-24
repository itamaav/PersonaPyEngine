import ntpath
import cv2
import numpy as np
import imutils
from Engine.State import State
from Engine.Status import Status
from Engine.ImageStorage import ImageData
from Engine.ImageStorage import ImageDataSet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append('/home/personaitaben/PersonaPyEngine/')
class PerosnaEngine:
    """"===================================================================================================
        ===================================================================================================

                PersonaEngine Class - Handle all analysis process of an image Processing

                Arguments:

                    image_path : path to the image file

                    type : two options : 1. Debug ( analysis process )
                                         2. Full ( analysis process + store analyzed products )

        ===================================================================================================
        ==================================================================================================="""

    def __init__(self, type):

        print()
        print("PersonaEngine : Building new instance !", end="\n\n")

        self.type = type

        self.image_data_set = ImageDataSet()




    def start(self,image_path):

        self.image_path = image_path
        self.image_filename = ntpath.basename(image_path)

        self.image_data_set.image_filename = self.image_filename

        print(" * Full Path -->  ", self.image_path)
        print(" * File Name -->  ", self.image_filename)

        if self.image_path:

            self.imageOriginal = cv2.imread(self.image_path)
            self.image = self.imageOriginal.copy()

            if self.image is not None:

                self.height, self.width, self.channels = self.image.shape

                print(" * image read successfully")
                print(" * Height = ", self.height, "Width = ", self.width, "Channels = ", self.channels)

                self.currentStatus = Status()
                self.image_objects_list_original = []
                self.names_of_drawn_objects = []
                self.image_objects_list_drawn = []
                self.names_of_original_objects = []

        else:

            print("Error ---> Image not found ! ", end="\n\n")

        stateType = State()
        currentstate = stateType.resizeState

        while (currentstate != stateType.End):

            if (currentstate == stateType.resizeState):

                print("\n", "Current State : No.", currentstate, " resizeState")

                if (self.resizeStateExec() == Status.SUCCESS):
                    currentstate = stateType.thresholdState
                else:
                    currentstate = stateType.End

            elif (currentstate == stateType.thresholdState):

                print("\n","Current State : No.", currentstate," thresholdState")

                if (self.thresholdStateExec() == Status.SUCCESS):
                    currentstate = stateType.DilatationState
                else:
                    currentstate = stateType.End

            elif (currentstate == stateType.DilatationState):

                print("\n","Current State : No.", currentstate," DilatationState")

                if (self.DilatationStateExec() == Status.SUCCESS) :
                    currentstate = stateType.ContoursState
                else :
                    currentstate = stateType.End

            elif (currentstate == stateType.ContoursState):

                print("\n","Current State : No.", currentstate," ContoursState")

                if (self.ContoursStateExec() == Status.SUCCESS):
                    currentstate = stateType.BuildBlankImageState
                else:
                    currentstate = stateType.End

            elif (currentstate == stateType.BuildBlankImageState):

                print("\n","Current State : No.", currentstate," BuildBlankImageState")

                if (self.BuildBlankImageStateExec() == Status.SUCCESS):
                    currentstate = stateType.FindRectOfElement
                else:
                    currentstate = stateType.End

            elif (currentstate == stateType.FindRectOfElement):

                print("\n","Current State : No.", currentstate," FindRectOfElement")

                if (self.FindRectOfElementExec() == Status.SUCCESS):
                    currentstate = stateType.PaddingElement
                else:
                    currentstate = stateType.End

            elif (currentstate == stateType.PaddingElement):

                print("\n","Current State : No.", currentstate," PaddingElement")

                if (self.PaddingElementExec() == Status.SUCCESS):
                    currentstate = stateType.StoreImage
                else:
                    currentstate = stateType.End

            elif (currentstate == stateType.StoreImage):

                print("\n","Current State : No.", currentstate," StoreImage")

                currentstate = stateType.End

                if (self.StoreImageExec() == Status.SUCCESS):
                    currentstate = stateType.End
                else:
                    currentstate = stateType.End

            else:

                print("\n","Current State : ", "Unknown Error !")

                break



        print("Engine end ----> The image has been analyzed successfully !" , end="\n\n")
        # print("\n","Current State : ", currentstate )
        # print("\n","Current Status : " , self.currentStatus.currentStat )

    def resizeStateExec(self):


        print("resizeStateExec say: ", "Height = ", self.height, "Width = ", self.width)


        if self.height > 500 and self.width > 500:

            print("resizeStateExec - start resize process")

            self.imageOriginal = cv2.resize(self.imageOriginal, (int(500), int(500)), interpolation=cv2.INTER_AREA)

            self.image = cv2.resize(self.image, (int(500), int(500)), interpolation=cv2.INTER_AREA)

            self.height, self.width, self.channels = self.image.shape

            if self.height > 0 and self.width > 0 and self.channels > 0:

                self.image_data_set.set_width_height(self.width, self.height)

                return self.currentStatus.SUCCESS

            else:

                return self.currentStatus.FAILED

        else:

            self.image_data_set.set_width_height(self.width, self.height)

            print("resizeStateExec - No need for resize process , move to next state.")



            return self.currentStatus.SUCCESS

    def thresholdStateExec(self):

        print("thresholdStateExec say: ", "Height = ", self.height, "Width = ", self.width)

        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.GaussianBlur(self.image, (5, 5), 0)
        ret, self.image = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        if self.image is not None:
            return self.currentStatus.SUCCESS
        else:
            return self.currentStatus.FAILED

    def DilatationStateExec(self):

        print("DilatationStateExec say: ", "Height = ", self.height, "Width = ", self.width)

        kernel = np.ones((4, 4), np.uint8)
        self.image = cv2.dilate(self.image, kernel, iterations=3)

        if self.image is not None:
            return self.currentStatus.SUCCESS
        else:
            return self.currentStatus.FAILED

    def ContoursStateExec(self):

        safe_counter = 0

        print("ContoursStateExec say: ", "Height = ", self.height, "Width = ", self.width)

        cnts = cv2.findContours(self.image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        # self.image = cv2.bitwise_not(self.image)  WHY ???????

        for c in cnts:

            safe_counter += 1

            M = cv2.moments(c)

            if M["m00"]:

                cX = int((M["m10"] / M["m00"]))
                cY = int((M["m01"] / M["m00"]))

                print("Object found ( ", safe_counter, " ) ---> cX= ", cX, " | cY= ", cY)

                c = c.astype("float")
                c = c.astype("int")

                blank_image = np.zeros((self.height, self.width, self.channels), np.uint8)

                cv2.drawContours(blank_image, [c], 0, (255, 255, 255), 2)

                blank_image = cv2.bitwise_not(blank_image)

                self.image_objects_list_drawn.append(blank_image)

            if safe_counter > 20:

                print("ContoursStateExec say: ERROR - too many objects found on the image")

                return self.currentStatus.FAILED

        return self.currentStatus.SUCCESS

    def BuildBlankImageStateExec(self):

        print("BuildBlankImageStateExec say: ", "Height = ", self.height, "Width = ", self.width)

        return self.currentStatus.SUCCESS

    def FindRectOfElementExec(self):

        print("FindRectOfElementExec say: ", "Height = ", self.height, "Width = ", self.width)

        size_of_image_objects_list_drawn = len(self.image_objects_list_drawn)

        safe_counter = 0

        if size_of_image_objects_list_drawn:

            # for obj_drawn in self.image_objects_list_drawn:

            for i in range(len(self.image_objects_list_drawn)):

                self.image_objects_list_drawn[i] = cv2.cvtColor(self.image_objects_list_drawn[i], cv2.COLOR_BGR2GRAY)
                self.image_objects_list_drawn[i] = cv2.GaussianBlur(self.image_objects_list_drawn[i], (5, 5), 0)
                ret, self.image_objects_list_drawn[i] = cv2.threshold(self.image_objects_list_drawn[i], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                cnts_inner = cv2.findContours(self.image_objects_list_drawn[i].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts_inner = cnts_inner[0] if imutils.is_cv2() else cnts_inner[1]

                for c_inner in cnts_inner:

                    M_inner = cv2.moments(c_inner)

                    if M_inner["m00"]:

                        safe_counter += 1

                        cx = int(M_inner['m10'] / M_inner['m00'])
                        cy = int(M_inner['m01'] / M_inner['m00'])

                        x, y, w, h = cv2.boundingRect(c_inner)

                        print("Object found ( ", safe_counter, " ) ---> cX= ", cx, " | cY= ", cy)
                        print("Object Top-left coordinate: ( x=", x, "y=", y, " )")
                        print("Object Size : ( width= ", w, " height=", h, " )" , end="\n\n")

                        self.image_objects_list_drawn[i] = self.image_objects_list_drawn[i][y - 5:y + h + 5, x - 5:x + w + 5]
                        self.image_objects_list_drawn[i] = cv2.bitwise_not(self.image_objects_list_drawn[i])

                        obj_org = self.imageOriginal[y - 5:y + h + 5, x - 5:x + w + 5]
                        self.image_objects_list_original.append(obj_org)
                        image_data_obj = ImageData()
                        image_data_obj.calculate_precentage(w+5, h+5, self.imageOriginal.shape[1], self.imageOriginal.shape[0])
                        self.image_data_set.append_to_image_data_list(image_data_obj)

                    if safe_counter > 20:

                        print("FindRectOfElementExec say: ERROR - too many objects found on the image")

                        return self.currentStatus.FAILED

        return self.currentStatus.SUCCESS

    def PaddingElementExec(self):

        print("PaddingElementExec say: ", "Height = ", self.height, "Width = ", self.width)

        size_of_image_objects_list_drawn = len(self.image_objects_list_drawn)
        size_of_image_objects_list_org = len(self.image_objects_list_original)

        WHITE = [255, 255, 255]
        safe_counter = 0

        if size_of_image_objects_list_drawn:

            for i in range(len(self.image_objects_list_drawn)):

                print(i+1, "Before:", self.image_objects_list_drawn[i].shape[:2])

                h, w = self.image_objects_list_drawn[i].shape[:2]

                if w > h:

                    diff = w - h
                    print("Diff w > h = ", diff)
                    padding_height_inner = diff
                    add_both = w * 0.2
                    padding_height_inner += add_both
                    padding_width, padding_height = int(add_both / 2), int(padding_height_inner / 2)

                else:

                    diff = h - w
                    print("Diff w < h = ", diff)
                    padding_width_inner = diff
                    add_both = h * 0.2
                    padding_width_inner += add_both
                    padding_width, padding_height = int(padding_width_inner / 2), int(add_both / 2)

                self.image_objects_list_drawn[i] = cv2.copyMakeBorder(self.image_objects_list_drawn[i], padding_height,
                                            padding_height,padding_width,padding_width, cv2.BORDER_CONSTANT, value=WHITE)


                print(i+1, "After: ", self.image_objects_list_drawn[i].shape[:2], end="\n\n")
                # cv2.imshow("Image d", self.image_objects_list_drawn[i])
                # cv2.waitKey(3000)

        if size_of_image_objects_list_org:

            for i in range(len(self.image_objects_list_original)):

                print(i+1, "Before:", self.image_objects_list_original[i].shape[:2])

                h, w = self.image_objects_list_original[i].shape[:2]

                if w > h:

                    diff = w - h
                    print("Diff w > h = ", diff)
                    padding_height_inner = diff
                    add_both = w * 0.2
                    padding_height_inner += add_both
                    padding_width, padding_height = int(add_both / 2), int(padding_height_inner / 2)

                else:

                    diff = h - w
                    print("Diff w < h = ", diff)
                    padding_width_inner = diff
                    add_both = h * 0.2
                    padding_width_inner += add_both
                    padding_width, padding_height = int(padding_width_inner / 2), int(add_both / 2)

                self.image_objects_list_original[i] = cv2.copyMakeBorder(self.image_objects_list_original[i], padding_height,
                                            padding_height,padding_width,padding_width, cv2.BORDER_CONSTANT, value=WHITE)


                print(i+1, "After: ", self.image_objects_list_original[i].shape[:2] , end="\n\n")
                # cv2.imshow("Image r", self.image_objects_list_original[i])
                # cv2.waitKey(3000)

        return self.currentStatus.SUCCESS

    def StoreImageExec(self):

        print("StoreImageExec say: ", "Height = ", self.height, "Width = ", self.width)

        size_of_image_objects_list_drawn = len(self.image_objects_list_drawn)
        size_of_image_objects_list_org = len(self.image_objects_list_original)

        if self.type == "Full":

            index = 0

            if size_of_image_objects_list_drawn:

                for img_drawn in self.image_objects_list_drawn:

                    index += 1
                    str_to_save_d = "/home/personaitaben/PersonaPyEngine/predict/" + self.image_filename[:self.image_filename.find(".")] + \
                                    "/" + self.image_filename[:self.image_filename.find(".")] + "_" + \
                                    str(index) + "_drawn.png"
                    cv2.imwrite(str_to_save_d, img_drawn)
                    print("saving ---> ", str_to_save_d)
                    self.names_of_drawn_objects.append(str_to_save_d)

            index = 0

            if size_of_image_objects_list_org:

                for img_org in self.image_objects_list_original:

                    index += 1
                    str_to_save_o = "/home/personaitaben/PersonaPyEngine/predict/" + self.image_filename[:self.image_filename.find(".")] + \
                                    "/" + self.image_filename[:self.image_filename.find(".")] + "_" + \
                                    str(index) + "_original.png"
                    cv2.imwrite(str_to_save_o, img_org)
                    self.names_of_original_objects.append(str_to_save_o)
                    print("saving ---> ", str_to_save_o)

            index = 0

            if self.image_data_set:

                for img_data in self.image_data_set.image_data_list:

                    index += 1
                    img_data.image_filename = self.image_filename[:self.image_filename.find(".")] + \
                                             "_" + str(index) + ".png"
                    img_data.image_info = "check - "+str(index)
                    print("saving log ---> ", img_data.image_filename)

        return self.currentStatus.SUCCESS








