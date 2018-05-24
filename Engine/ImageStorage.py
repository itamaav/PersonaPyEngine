import json
import sys
sys.path.append('../')

class ImageData:

    def __init__(self):

        print()
        print("ImageDataSet: Building new instance !", end="\n\n")

        self.image_filename = "None"
        self.image_label_after_classification = "None"
        self.image_info = "None"

        self.image_width = 0
        self.image_height = 0

        self.image_percentage_size_from_origin = 0

        self.colors_analyzed_instance = []

    def calculate_precentage(self, width_obj, height_obj, width_org, height_org ):

        self.image_width = width_obj
        self.image_height = height_obj

        self.image_percentage_size_from_origin = float((width_obj*height_obj)/(width_org*height_org) * 100)
        self.image_percentage_size_from_origin = round(self.image_percentage_size_from_origin, 2)

        print("Step 1 - calculate percentages say : ", " w of obj = ", width_obj, "H of obj =  ", height_obj )
        print("Total object size = ", width_obj*height_obj)

        print("Step 2 - calculate percentages say : ", " w of org = ", width_org, "H of org =  ", height_org)
        print("Total original size = ", width_org*height_org)

        print("Results -------> ", self.image_percentage_size_from_origin, "\n\n")



    def createAsJson(self):

        jsonData = {}

        jsonData["segment_name"] = self.image_filename
        jsonData["label"] = self.image_label_after_classification
        jsonData["info"] = self.image_info
        jsonData["width"] = self.image_width
        jsonData["height"] = self.image_height
        jsonData["percentage"] = self.image_percentage_size_from_origin

        return json.dumps(jsonData)

    def createAsDict(self):

        jsonData = {}

        jsonData["segment_name"] = self.image_filename
        jsonData["label"] = self.image_label_after_classification
        jsonData["info"] = self.image_info
        jsonData["width"] = self.image_width
        jsonData["height"] = self.image_height
        jsonData["percentage"] = self.image_percentage_size_from_origin
        jsonData["colors"] = self.colors_analyzed_instance

        return jsonData


class ImageDataSet:

    def __init__(self):

        print()
        print("ImageDataSet : Building new instance !", end="\n\n")

        self.image_filename = ""
        self.image_data_list = []
        self.image_org_width = 0
        self.image_org_height = 0

    def set_width_height(self, w, h):

        print("ImageDataSet ---> set values :  width = ", w, " height = ", h, end="\n\n")

        self.image_org_width = w
        self.image_org_height = h

    def append_to_image_data_list(self, image_data):

        self.image_data_list.append(image_data)



    def createAsJson(self):

        jsonData = {}

        jsonData["image_name"] = self.image_filename
        jsonData["width"] = self.image_org_width
        jsonData["height"] = self.image_org_height
        jsonData["segments"] = [segment.createAsDict() for segment in self.image_data_list]

        return json.dumps(jsonData)
