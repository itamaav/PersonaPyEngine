import numpy as np
import cv2
import webcolors as webcolors
from sklearn.cluster import KMeans
import sys
sys.path.append('/home/personaitaben/PersonaPyEngine/')


class ColorsProcessor:

    def __init__(self, image_filename_set = ""):

        print()
        print("ColorsProcessor : Building new instance !", end="\n\n")

        self.image_filename = ""

        if image_filename_set != "":

            self.image_filename = image_filename_set


    @staticmethod
    def centroid_histogram(clt):
        # grab the number of different clusters and create a histogram
        # based on the number of pixels assigned to each cluster
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)

        # normalize the histogram, such that it sums to one
        hist = hist.astype("float")
        hist /= hist.sum()

        # return the histogram
        return hist

    @staticmethod
    def plot_colors(hist, centroids):
        # initialize the bar chart representing the relative frequency
        # of each of the colors
        bar = np.zeros((50, 300, 3), dtype="uint8")
        startX = 0
        color_list=[]
        # loop over the percentage of each cluster and the color of
        # each cluster
        for (percent, color) in zip(hist, centroids):
            # plot the relative percentage of each cluster
            color_list.append(round(percent*100,2))
            endX = startX + (percent * 300)
            cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                          color.astype("uint8").tolist(), -1)
            startX = endX

        # return the bar chart
        return bar , color_list

    @staticmethod
    def closest_colour(requested_colour):
        min_colours = {}
        for key, name in webcolors.css3_hex_to_names.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_colour[0]) ** 2
            gd = (g_c - requested_colour[1]) ** 2
            bd = (b_c - requested_colour[2]) ** 2
            min_colours[(rd + gd + bd)] = name
        return min_colours[min(min_colours.keys())]

    @staticmethod
    def get_colour_name(requested_colour):
        try:
            closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
        except ValueError:
            closest_name = ColorsProcessor.closest_colour(requested_colour)
            actual_name = None
        return actual_name, closest_name

    @staticmethod
    def clear_color(tempColor):

        color_conv = ['red', 'turquoise', 'green', 'brown', 'yellow', 'blue', 'orange', 'purple', 'grey', 'white', 'black', 'pink', 'gold']

        for color in color_conv:
            if color in tempColor:
                return color

        return tempColor

    def analyse_image(self):

        image = cv2.imread(self.image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape((image.shape[0] * image.shape[1], 3))
        clt = KMeans(n_clusters=5)
        clt.fit(image)

        hist = ColorsProcessor.centroid_histogram(clt)
        bar, list_of_col = ColorsProcessor.plot_colors(hist, clt.cluster_centers_)

        print("len of list :" ,len(list_of_col))

        colors_percent = {}

        for i in range(len(list_of_col)):

            real_color, closest_color = ColorsProcessor.get_colour_name(clt.cluster_centers_[i])
            if real_color is not None:
                colors_percent[ColorsProcessor.clear_color(real_color)] = list_of_col[i]
            else:
                colors_percent[ColorsProcessor.clear_color(closest_color)] = list_of_col[i]

        return colors_percent
