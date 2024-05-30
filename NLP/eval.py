import nltk
anspath='./ans.txt'
resultpath='./result.txt'
testpath='./testdata.txt'
ansfile=open(anspath,'r')
resultfile=open(resultpath,'r')
testfile=open(testpath,'r')
count=1
with open('data_ans_result.txt', 'w') as file:
    for i in range(1000):
        ansline=ansfile.readline().split('\t')
        ansset=set(nltk.word_tokenize(ansline[1]))
        resultline=resultfile.readline().split('\t')
        resultset=set(nltk.word_tokenize(resultline[1]))
        testline = testfile.readline().split('\t')
        if ansset==resultset:
            count+=1
        else:
            file.write(testline[0] + '\t' + testline[2])
            file.write(ansline[0] + '\t' + ansline[1])
            file.write(resultline[0] + '\t' + resultline[1] + '\n')
    print("Accuracy is : %.2f%%" % (count * 1.00 / 10))
