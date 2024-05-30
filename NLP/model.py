import numpy as np
import json
from tqdm import tqdm
from nltk.corpus import reuters
class LanguageModel:
    def __init__(self):
        self.bigram_prob = {}

    def train(self):
        term_count = {}
        biagram_term_count = {}
        categories = reuters.categories()
        corpus = reuters.sents(categories=categories)
        # 构建语言模型: bigram
        for doc in tqdm(corpus):
            doc = ['<s>'] + doc + ['</s>']
            for i in range(len(doc)):
                term = doc[i]
                biagram_term = doc[i:i + 2]
                biagram_term = ' '.join(biagram_term)

                if term in term_count:
                    term_count[term] += 1
                else:
                    term_count[term] = 1

                if biagram_term in biagram_term_count:
                    biagram_term_count[biagram_term] += 1
                else:
                    biagram_term_count[biagram_term] = 1

        for bigram in tqdm(biagram_term_count.keys()):
            self.bigram_prob[bigram] = np.log(biagram_term_count[bigram] / term_count[bigram.split()[0]])

        with open('languagemodel.json', 'w', encoding='utf8') as f:
            json.dump(self.bigram_prob, f)

    def load(self):
        with open('languagemodel.json', 'r', encoding='utf8') as f:
            self.bigram_prob = json.load(f)


def edit_distance(str1, str2):
    m = len(str1)
    n = len(str2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    # Backtrack to find the transformation
    transformations = []
    i, j = m, n
    while i > 0 or j > 0:  # delete
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            transformations.append(f"{str1[i - 2]}|{str1[i - 2]}{str1[i - 1]}")
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:  # insert
            if j - 1 == 0:
                transformations.append(f"{str2[j - 1]}|#")
            else:
                transformations.append(f"{str2[j - 1]}{str2[j - 2]}|{str2[j - 1]}")
            j -= 1
        else:  # replace
            if dp[i][j] != dp[i - 1][j - 1]:
                transformations.append(f"{str2[j - 1]}|{str1[i - 1]}")
            i -= 1
            j -= 1

    transformations.reverse()
    return dp[m][n], transformations


#_,tran=edit_distance('aple', 'apple')
#print(tran)
#str1为正确的的，str2为错误的

class channelmodel:
    def __init__(self):
        self.cha_probs = {}

    def train(self):
        with open('vocab.txt','r',encoding='utf8') as f,open('spell-singal-error.txt', 'r',encoding='utf8') as f2:
            for item in tqdm(f2):
                item = item.strip()
                count = 1
                for word in f:
                    letter = item.split('\t')[0].split('|')[1].strip()
                    count += word.count(letter)
                f.seek(0)
                self.cha_probs[item.split('\t')[0]] = np.log(int(item.split('\t')[1].strip()) / count)

        with open('channelprob.json', 'w', encoding='utf8') as f:
            json.dump(self.cha_probs, f)

    def load(self):
        with open('channelprob.json', 'r', encoding='utf8') as f:
            self.cha_probs = json.load(f)
            print('channelprob load successfully!')




