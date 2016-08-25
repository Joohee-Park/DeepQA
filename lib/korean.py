# This function returns stream of unfolded Korean alphabet given regular Korean sentences.
# e.g 자연어처리 -> ㅈㅏㅇㅕㄴㅇㅓㅊㅓㄹㅣ

def unfold(sentence):

    first = [ "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ" ]
    middle = [ "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ","ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ" ]
    last = [ "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ" ]

    result = []
    for letter in sentence :

        if 0xAC00 <= ord(letter) and ord(letter) <= 0xD7AF:
            korean_value = ord(letter) - 0xAC00
            last_index = int(korean_value % 28 )
            middle_index = int(((korean_value - last_index) / 28) % 21)
            first_index = int((((korean_value - last_index) / 28) - middle_index) / 21)
            result.append(first[first_index])
            result.append(middle[middle_index])
            result.append(last[last_index])
        else :
            # 한글 모아쓰기의 범위가 아닌 것들은 그냥 그대로 리턴
            # e.g ㅋㅋㅋ, abc
            result.append(letter)

    return "".join(result)
