import os
import numpy as np
import lib.tensor as tensor

# Answers in this dictionary doesn't have white space
def answer_dict():
    ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

    try:
        f = open(ROOT_DIR + "/Data/answer_candidates.txt", "r", encoding="utf-8")
    except:
        print("Can not find /Data/answer_candidates.txt ")
        return

    ansToidx = {}
    idxToans = {}
    for index, _answer in enumerate(f.readlines()):
        answer = tensor.preprocess(_answer.replace("\n", "")).replace(" ","")
        ansToidx[answer] = index
        idxToans[index] = answer

    return ansToidx, idxToans

#This function converts .txt data into 3-d tensors
def toCorpusTensor(file_name):

    f = open(file_name, "r", encoding="utf-8")
    ans2idx, idx2ans = answer_dict()

    sentenceTensorList = []
    answerTensorList = []

    for line in f.readlines():
        # Read a line
        try:
            _title, sentence = line.replace("\n","").split("\t")
        except :
            continue

        # Make sentence tensor
        try:
            sentenceTensor = tensor.toTensor(sentence)
        except Exception as es :
            #print("sentence error : " + str(es) + " " + str(sentence))
            continue

        # Make answer tensor
        try:
            title = tensor.preprocess(_title)
            answerTensor = tensor.toAnswerTensor(ans2idx[title])
        except Exception as ae :
            print(_title)
            print("answer error : " + str(ae) + " " + str(sentence))
            continue

        # Append to the tensors to each list if both tensors have no problem
        answerTensorList.append(answerTensor)
        sentenceTensorList.append(sentenceTensor)


    return np.stack(sentenceTensorList), np.stack(answerTensorList)

def prepareTrainingData():
    ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
    return toCorpusTensor(ROOT_DIR + "/Data/Training/sample.txt")