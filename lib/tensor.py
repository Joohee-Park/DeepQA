import re
import numpy as np
import lib.korean as korean

_embedim = 98
_maxlen = 700
_answerDim = 150
dict = { "ㄱ":0, "ㄲ":1, "ㄴ":2, "ㄷ":3, "ㄸ":4, "ㄹ":5, "ㅁ":6, "ㅂ":7, "ㅃ":8, "ㅅ":9, "ㅆ":10, "ㅇ":11, "ㅈ":12, "ㅉ":13,
             "ㅊ":14, "ㅋ":15, "ㅌ":16, "ㅍ":17, "ㅎ":18, "ㅏ":19, "ㅐ":20, "ㅑ":21, "ㅒ":22, "ㅓ":23, "ㅔ":24,
             "ㅕ":25, "ㅖ":26, "ㅗ":27, "ㅘ":28, "ㅙ":29, "ㅚ":30, "ㅛ":31, "ㅜ":32,"ㅝ":33, "ㅞ":34, "ㅟ":35, "ㅠ":36, "ㅡ":37,
             "ㅢ":38, "ㅣ":39, "ㄳ":40, "ㄵ":41, "ㄶ":42, "ㄺ":43, "ㄻ":44, "ㄼ":45, "ㄽ":46, "ㄾ":47, "ㄿ":48, "ㅀ":49, "ㅄ":50,
             "a":51, "b":52, "c":53, "d":54, "e":55, "f":56, "g":57, "h":58, "i":59, "j":60, "k":61, "l":62, "m":63, "n":64,
             "o":65, "p":66, "q":67, "r":68, "s":69, "t":70, "u":71, "v":72, "w":73, "x":74, "y":75, "z":76, "!":77, "\"":78,
             "?":79, ".":80, ",":81, "-":82, ":":83, "~":84, "%":85, "\'":86, "0":87, "1":88, "2":89, "3":90, "4":91, "5":92,
             "6":93, "7":94, "8":95, "9":96, " ":97 }

# This function converts sentences into tensors
# Tensor can express : Korean Alphabet, English Alphabet (lower case), Numbers and punctuation marks
# Its dimension is 98
def toTensor(sentence):
    # Input : Normal sentence e.g "나는 밥을 먹었다."
    # Output : 300 X 98 tensor
    embedim = _embedim
    maxlen = _maxlen

    stage_1 = preprocess(sentence)
    stage_2 = korean.unfold(stage_1)

    tindex = [97] * maxlen
    for i, letter in enumerate(stage_2) :
        if not letter in dict :
            continue
        else :
            tindex[i] = dict[letter]

    tensor = np.zeros((maxlen,embedim))
    for i in range(len(tindex)):
        tensor[i][tindex[i]] = 1

    return tensor

# This function converts tensor into sentenes
# Input dimension : 300 X 98
# Output : Corresponding sentence
def toSentence(tensor):

    embedim = _embedim
    maxlen = _maxlen
    if tensor.shape != (maxlen, embedim) :
        print("Tensor dimension doesn't match to (" + str(embedim) + ", " + str(maxlen) + "), given :" + str(tensor.shape))
        return

    inv_dict = {v:k for k,v in dict.items()}

    result = ""
    # Check the tensor and lookup in the inverted dictionary
    for i in range(maxlen):
        tindex = 0
        for j in range(embedim):
            if tensor[i][j] != 0 :
                tindex = j
                break
        result += inv_dict[tindex]

    return result

def toAnswerTensor(index):
    resultTensor = np.zeros(_answerDim)
    resultTensor[index] = 1
    return resultTensor

# This function preprocesses the sentence
# 1. It removes the irregular space
# 2. If it does not end with ".", then append it
# 3. It removes the characters between parenthesis
# 4. If it contains out-of-scope characters, it removes it
def preprocess(sentence):

    if len(sentence) < 1:
        return sentence

    # [1] It removes the characters between parenthesis : (),[],{}
    # ? in regex means "greedy"
    sentence = re.sub(r"\(.*?\)","", sentence)
    sentence = re.sub(r"\[.*?\]","", sentence)
    sentence = re.sub(r"\{.*?\}","", sentence)

    # [2] If it contains out-of-scope characters, remove it
    # Korean Syllable : \uAC00-\uD7AF
    # Korean Alphabet : \u1100-\u11FF
    # Alphanumeric and punctuation marks : \u0021-\u007E
    sentence = re.sub('[^\uAC00-\uD7AF\u1100-\u11FF\u0021-\u007E ]+',"",sentence)

    # [3] Some Preprocessing
    # Replace various whitespace into normal space
    sentence = re.sub('[\s]+'," ",sentence)
    # Convert to lower-case
    sentence = sentence.lower()

    # If out-of-string-index error occurs, just ignore it
    try:
        # [3] It removes start-space
        if sentence[0] == ' ':
            sentence = sentence[1:]

    except:
        return sentence

    return sentence

# Split the sentence out of text corpus
# Returns the list of sentence
def splitSentence(text):
    result = []
    # Do not split the sentence for the case of "0.3"
    text = re.sub(r'([0-9])\.([0-9])',r'\1@\2',text)
    # Do not split the sentence for the case of "e.g", "cf.", "St.", "st.", "s.t"
    text = text.replace("e.g", "e@g").replace("cf.", "cf@").replace("St.", "St@").replace("st.", "st@").replace("s.t", "s@t")
    for line in text.split("."):
        line = line.replace("@",".")
        if len(line) > 0:
            result.append(line)
    return result

