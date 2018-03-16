from .VNAAD_extraction_SVM import Detectidioms
from .VNAAD_extraction_SVM_best import trainingIdiomDetector
from .VNAAD_extraction_SVM_input import predictIdiom
import os


def handle_uploaded_file(f):
    # with open('/Users/kananekatichatviwat/Downloads/IdiomDetectorServer-2/IdiomDetectorServer/idiom_detector/uploadFiles/dataset.txt', 'wb+') as destination:
    with open(os.path.abspath(os.path.dirname(__file__)) + "/uploadFiles/data.txt", 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    print(trainingIdiomDetector(os.path.abspath(os.path.dirname(__file__)) + "/uploadFiles/data.txt"))
    return trainingIdiomDetector(os.path.abspath(os.path.dirname(__file__)) + "/uploadFiles/data.txt")

def handle_input_sentence(sentence):
    print(predictIdiom(sentence))
    return predictIdiom(sentence)