import numpy as np
from pyvi import ViTokenizer
from pyvi import ViUtils
from preprocess import text_preprocess
from collections import defaultdict
from difflib import SequenceMatcher
import re

from difflib import SequenceMatcher


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


# Food
foods = [
    "đồ ăn",
    "hương vị",
    "ăn",
    "món ăn",
    "khẩu phần",
    "thức ăn",
    "mùi",
    "mùi thơm",
    "đồ nướng",
    "đồ",
    "ngon",
    "tệ",
    "lèo tèo",
    "tẩm ướp",
    "no",
    "tươi",
    "kim chi", "dưa chua", "dưa chuột", "bánh mì bơ", "bánh mì", "khoai", "ngô", "khoai chiên", "ngô chiên",
    "ba chỉ", "bò", "ướp", "nấm", "bạch tuộc", "mực", "chân gà", "dạ dày", "gà", "lòng", "lòng nướng",
    "mực trứng", "trứng", "ngao", "ngao nướng", "mỡ hàng", "mỡ", "tôm", "tôm nướng", "trâu", "sụn", "thịt dải",
    "hàu nướng", "mỡ hành", "ba chỉ", "sườn", "há cảo", "cá hồi", "hcao", "tráng miệng", "hoa quả",
    "lẩu", "thập cẩm", "lẩu thái", "riêu cua", "cá khoai", "hải sản", "rau",
    "đắng", "cay", "ngọt", "mặm", "thức ăn"
]
# Drink
drinks = [
    "uống",
    "vị",
    "thơm", "Coca", "cam", "chanh", "ép", "sữa chua", "thạch",
    "rượu", "bia", "trà đá"

]
# Restaurant 
restaurant = [
    "menu"
    "nhà hàng",
    "vệ sinh",
    "không gian",
    "bẩn",
    "sạch",
    "kgian",
    "quán"

]
# Frice
prices = [
    "giá cả",
    "tiền",
    "túi tiền",
    "đắt",
    "rẻ",
]
# Staff
staff = ["nam", "nữ",
         "nhân viên",
         "phục vụ",
         "nvien",
         "pvu",
         "nhanh",
         "chậm",
         "bảo vệ"

         ]
# Location
locations = [
    "vị trí", "địa điểm"
]
all_regs = {"foods": foods, "drinks": drinks, "restaurant": restaurant, "prices": prices, "staff": staff,
            "locations": locations}

for k, v in all_regs.items():
    reg_temp = []
    for word in v:
        if len(word.split(" ")) == 2:
            temp = ViTokenizer.tokenize(word)
            temp1 = ViUtils.remove_accents(word).decode("utf-8")
            temp2 = ViUtils.remove_accents(temp).decode("utf-8")
            # print(temp, temp1, temp2)
            if temp != word:
                reg_temp.extend([temp, temp2])
            else:
                (t1, t2) = temp.split(" ")
                reg_temp.extend(
                    [t1, t2, ViUtils.remove_accents(t1).decode("utf-8"), ViUtils.remove_accents(t2).decode("utf-8")])
        else:
            reg_temp.extend([word, ViUtils.remove_accents(word).decode("utf-8")])
    all_regs[k] = (reg_temp)


def has_price(word):
    if re.search("[0-9]*[0-9][a-z]", word) is not None:
        return True
    return False


hash_word = {}
for k, v in all_regs.items():
    for i in v:
        hash_word[i] = k


def ner_sent(sent_processed):
    split_sent = sent_processed.split(" ")
    labels = []
    for word in split_sent:
        if has_price(word): labels.append("price")
        if hash_word.get(word) is not None:
            labels.append(hash_word.get(word))
    label, percent = np.unique(labels, return_counts=True)
    percent = percent / len(labels)
    result_dict = {}
    for i in range(len(label)):
        result_dict[label[i]] = percent[i]
    return result_dict
