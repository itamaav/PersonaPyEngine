
import datetime

import cv2

import ntpath


from Engine.PersonaEngine import PerosnaEngine

from Engine.PersonaConNeuralNetwork import PersonaConNeuralNetwork

from Engine.PictureHandler import PictureHandler



""""===================================================================================================
    ===================================================================================================

                                PERSONA ENGINE - MAIN FUNCTION

                              written by : 1. Itamar Fred Avrahami
                                           2. Ben Shmuel
                                     
                                     

    This is the Main function of Persona Engine.

    Each time the engine will get a trap in order to start the analysis process, this function will invoke.

    The function build an PersonaEngine instance and pass an image to it as an array of numpy array type.

    Each step on the analysis process is present by a state on the engine state machine.

    ===================================================================================================
    =================================================================================================== """


def main():


    print("====================== Persona Engine Started ======================")
    print()
    print("Current Time: ", datetime.datetime.now())

    path1 = "../ExamplesForAnalysis/example_1.jpg"
    path2 = "../ExamplesForAnalysis/example_2.jpeg"
    path3 = "../ExamplesForAnalysis/example_3.JPG"
    path4 = "../ExamplesForAnalysis/example_4.png"
    path5 = "../ExamplesForAnalysis/example_5.png"
    path6 = "../ExamplesForAnalysis/example_6.png"

    engine_of_persona = PerosnaEngine("Full")

    # Example No 1
    # engine_of_persona.start(path1)
    # files_to_classify_example_1 = engine_of_persona.names_of_drawn_objects

    # Example No 2
    # engine_of_persona.start(path6)
    # files_to_classify_example_2 = engine_of_persona.names_of_drawn_objects


    print("====================== Persona Cnn Model Started ======================")
    print()
    print("Current Time: ", datetime.datetime.now())


    persona_cnn = PersonaConNeuralNetwork("Full")

    """
    model_of_persona_cnn = persona_cnn.load_model_persona("../PersonaModels/persona_multi_classes_six_elements_epoch6.h5")

    for file_image in files_to_classify_example_1:

        print("\n", "image to predict ---> ", file_image)
        image_to_predict = persona_cnn.load_image_for_predict(file_image,(64,64))
        print("predict object return ---->" , persona_cnn.predict_persona(image_to_predict))

    for file_image in files_to_classify_example_2:

        print("\n", "image to predict ---> ", file_image)
        image_to_predict = persona_cnn.load_image_for_predict(file_image, (64, 64))
        print("predict object return ---->", persona_cnn.predict_persona(image_to_predict)) """

    """
    persona_cnn_for_training = PersonaConNeuralNetwork("Full")

    model_of_persona_cnn_for_training = persona_cnn_for_training.BuildPersonaCnnModel(6,64,64)

    training_data_gen, test_data_gen = persona_cnn_for_training.create_image_generators()

    training_set,test_set = persona_cnn_for_training.create_image_datasets("../dataset_persona/training", "../dataset_persona/test")

    persona_cnn_for_training.fit_model(2)       """

    print("====================== Persona Picture Handler Started ======================")
    print()
    print("Current Time: ", datetime.datetime.now())

    # pictur_handler = PictureHandler()







if __name__ == "__main__":
    main()