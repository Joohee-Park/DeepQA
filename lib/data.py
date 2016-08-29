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
def toCorpusTensor(file_list):

    for file_name in file_list :
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

            title = tensor.preprocess(_title)

            hit_flag = False
            for entry in ans2idx:
                if title in entry or entry in title :

                    # Make sentence tensor
                    try:
                        sentenceTensor = tensor.toTensor(sentence)
                    except Exception as es :
                        #print("sentence error : " + str(es) + " " + str(sentence))
                        continue

                    # Make answer tensor
                    try:
                        title = tensor.preprocess(_title)
                        answerTensor = tensor.toAnswerTensor(ans2idx[entry])
                    except Exception as ae:
                        #print("answer error : " + str(ae) + " " + str(sentence))
                        continue

                    hit_flag = True

            # Append to the tensors to each list if both tensors have no problem
            if hit_flag:
                answerTensorList.append(answerTensor)
                sentenceTensorList.append(sentenceTensor)

    length = len(answerTensorList)
    if length == 0 :
        return
    answerTensor = np.zeros((length, answerTensorList[0].shape[0]))
    sentenceTensor = np.zeros((length, sentenceTensorList[0].shape[0], sentenceTensorList[0].shape[1]))
    for i in range(length):
        answerTensor[i,:] = answerTensorList[i][:]
        sentenceTensor[i,:,:] = sentenceTensorList[i][:,:]

    return sentenceTensor, answerTensor

def prepareTrainingData():
    TRAINING_DATA_DIR = os.path.dirname(os.path.dirname(__file__)) + "/Data/Training"
    file_list = [ TRAINING_DATA_DIR + "/" + file for file in os.listdir(TRAINING_DATA_DIR) if file.endswith(".txt") ]
    print("[1] Number of Training Text file is " + str(len(file_list)))
    return toCorpusTensor(file_list)