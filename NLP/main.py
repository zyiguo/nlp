import re
from model import *

chamodel = channelmodel()
chamodel.load()
languageprob = LanguageModel()
languageprob.load()

vocab = set([line.strip() for line in open('vocab.txt')])
def chosen(line):
    items1 = line.split('\t')
    line = re.sub(r'([a-zA-Z_]\w*),', r'\1 ,', items1[2])  # 将带标点符号的单词用空格隔开
    line = re.sub(r'\.$', ' .', line)
    line = line.replace("'", " ' ")
    word_list = line.split()
    word_list = ['<s>'] + word_list + ['</s>']
    num=0
    temp=0
    for index, word in zip(range(1, len(word_list)), word_list[1:-1]):
        word = word.strip(',.')
        word_length=len(word_list)
        if word in vocab and num != word_length:num = num + 1
        elif num == word_length:return 1
        else: return 0
#str1为正确的的，str2为错误的
def prob1(str1, str2):
    temp = 0
    _, tran = edit_distance(str1, str2)
    for index, word in enumerate(tran):
        if word in chamodel.cha_probs:
            temp = temp + float(chamodel.cha_probs[word])
        else:
            temp += (min(chamodel.cha_probs.values()) - 1)
    return temp


def generate_candinates(wrong_word):
    """
    word: 给定的输入（错误的输入）
    返回所有(valid)候选集合
    """
    # 生成编辑距离为1的单词
    # 1.insert 2. delete 3. replace
    # appl: replace: bppl, cppl, aapl, abpl...
    #       insert: bappl, cappl, abppl, acppl....
    #       delete: ppl, apl, app
    #letters = 'abcdefghijklmnopqrstuvwxyz'
    if wrong_word.islower():
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(wrong_word[:i], wrong_word[i:]) for i in range(len(wrong_word) + 1)]
        inserts = [left + letter + right for left, right in splits for letter in letters]
        deletes = [left + right[1:] for left, right in splits]
        replaces = [left + letter + right[1:] for left, right in splits for letter in letters]
    elif wrong_word.isupper():
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        splits = [(wrong_word[:i], wrong_word[i:]) for i in range(len(wrong_word) + 1)]
        inserts = [left + letter + right for left, right in splits for letter in letters]
        deletes = [left + right[1:] for left, right in splits]
        replaces = [left + letter + right[1:] for left, right in splits for letter in letters]
    elif len(wrong_word) == 1:
        inserts = wrong_word
        deletes = wrong_word
        replaces = wrong_word
    else:
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(wrong_word[:i], wrong_word[i:]) for i in range(len(wrong_word) + 1)]
        inserts = [left + letter + right for left, right in splits for letter in letters]
        deletes = [left + right[1:] for left, right in splits]
        replaces = [left + letter + right[1:] for left, right in splits for letter in letters]
    candidates = set(inserts + deletes + replaces)
    result = [candi for candi in candidates if candi in vocab]
    # 过滤掉不存在于词典库里面的单词
    return result


# 生成编辑距离为2的单词
def generate_edit_two(wrong_word):
    def generate_edit_one(wrong_word):
        if wrong_word.islower():
            letters = 'abcdefghijklmnopqrstuvwxyz'
            splits = [(wrong_word[:i], wrong_word[i:]) for i in range(len(wrong_word) + 1)]
            inserts = [left + letter + right for left, right in splits for letter in letters]
            deletes = [left + right[1:] for left, right in splits]
            replaces = [left + letter + right[1:] for left, right in splits for letter in letters]
        elif wrong_word.isupper():
            letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            splits = [(wrong_word[:i], wrong_word[i:]) for i in range(len(wrong_word) + 1)]
            inserts = [left + letter + right for left, right in splits for letter in letters]
            deletes = [left + right[1:] for left, right in splits]
            replaces = [left + letter + right[1:] for left, right in splits for letter in letters]
        elif len(wrong_word) == 1:
            inserts = wrong_word
            deletes = wrong_word
            replaces = wrong_word
        else:
            letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
            splits = [(wrong_word[:i], wrong_word[i:]) for i in range(len(wrong_word) + 1)]
            inserts = [left + letter + right for left, right in splits for letter in letters]
            deletes = [left + right[1:] for left, right in splits]
            replaces = [left + letter + right[1:] for left, right in splits for letter in letters]

        return set(inserts + deletes + replaces)

    candi_one = generate_edit_one(wrong_word)
    candi_list = []
    for candi in candi_one:
        candi_list.extend(generate_edit_one(candi))
    candi_two = set(candi_list)
    result = [candi for candi in candi_two if candi in vocab]
    return result
count = 1
chao = 1  #语言模型权重 基础为一
flag = 0
num=0
with open('testdata.txt') as file, open('result.txt', 'w') as file2:
    for line in file:
        items2=line
        items1 = line.split('\t')
        line = re.sub(r'([a-zA-Z_]\w*),', r'\1 ,', items1[2])  # 将带标点符号的单词用空格隔开
        line = re.sub(r'\.$', ' .', line)
        line = line.replace("'", " ' ")
        word_list = line.split()
        word_list = ['<s>'] + word_list + ['</s>']
        newitem = items1[2]
        if chosen(items2)==0:
            for index, word in zip(range(1, len(word_list)), word_list[1:-1]):
                word = word.strip(',.')
                word_length= len(word)
                if word_length > 3 and word not in vocab:
                    candidates = list(set(generate_candinates(word) + generate_edit_two(word)))
                    probs = []
                    candidates.append(word)
                    min_channel_prob = 0
                    for candi in candidates:
                        prob = 0
                        if candi != word:
                            prob += prob1(candi, word)
                        else:
                            prob += min_channel_prob
                        if min_channel_prob > prob:
                            min_channel_prob = prob

                        pre_word = word_list[index - 1]
                        biagram_pre = ' '.join([pre_word, candi])
                        if  biagram_pre in languageprob.bigram_prob:
                            prob = prob + chao * languageprob.bigram_prob[biagram_pre]
                        else:
                            prob = prob + chao * np.log(1e-7)
                        # 再计算log p(next_word|word)

                        if index + 1 < len(word_list):
                            next_word = word_list[index + 1]
                            biagram_next = ' '.join([candi, next_word])
                            if biagram_next in languageprob.bigram_prob:
                                prob += chao * languageprob.bigram_prob[biagram_next]
                            else:
                                prob += chao * np.log(1e-7)
                        probs.append(prob)
                    if probs:
                        max_idx = probs.index(max(probs))
                        newitem = re.sub(r'\b' + word + r'\b', candidates[max_idx], newitem)
                        flag = 1
            file2.write(items1[0] + '\t' + newitem)
        else :
            for index, word in zip(range(1, len(word_list)), word_list[1:-1]):
                word = word.strip(',.')
                if len(word) > 3 :
                    candidates = list(set(generate_candinates(word)))
                    probs = []
                    candidates.append(word)
                    min_channel_prob = 0
                    for candi in candidates:
                        prob = 0
                        if candi != word:
                            prob += prob1(candi, word)
                        else:
                            prob += min_channel_prob
                        if min_channel_prob > prob:
                            min_channel_prob = prob

                        pre_word = word_list[index - 1]
                        biagram_pre = ' '.join([pre_word, candi])
                        if biagram_pre in languageprob.bigram_prob:
                            prob = prob + chao * languageprob.bigram_prob[biagram_pre]
                        else:
                            prob = prob + chao * np.log(1e-7)
                            # 再计算log p(next_word|word)
                        if index + 1 < len(word_list):
                            next_word = word_list[index + 1]
                            biagram_next = ' '.join([candi, next_word])
                            if biagram_next in languageprob.bigram_prob:
                                prob += chao * languageprob.bigram_prob[biagram_next]
                            else:
                                prob += chao * np.log(1e-7)
                        probs.append(prob)
                    if probs:
                        max_idx = probs.index(max(probs))
                        newitem = re.sub(r'\b' + word + r'\b', candidates[max_idx], newitem)
            file2.write(items1[0] + '\t' + newitem)
