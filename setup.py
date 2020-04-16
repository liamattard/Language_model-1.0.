import language_model.preprocessing as preprocessing
from sympy.utilities.iterables import variations
import language_model.models as vm
import language_model.textGen as tg
import language_model.probabilityCalc as pc
import endpoint.api as api
from itertools import permutations 
from collections import Counter
import numpy as np


def main():
    print("Started API")


    x,y = preprocessing.train_split('text_files/outputs/preprocessing.txt')
    # np.set_printoptions(precision=25)    
    # y = vm.vanillaUnigram(x)
    # print(pc.calculateProbabilityFromUnigram(y,["<s>"],"</s>"))
    
    # b = vm.vanillaBigram(x)
    # print(pc.calculateProbabilityFromBigram(b,["<s>"],"meta"))
    

    # c = vm.vanillaTrigram(x)
    # print(pc.calculateProbabilityFromTrigram(c,["<s>","nhar"],"il-"))

    # print("Text generated form Vanilla Unigran: ")
    # a = tg.generateTextFromUnigram(y,["<s>"],lastWord="</s>")
    # print(a)

    # b = vm.vanillaBigram(x)
    # print("Text generated form Vanilla Bigram: ")
    # a = tg.generateTextFromBigram(b,["<s>"],lastWord="</s>")
    # print(a)

    # c = vm.vanillaTrigram(x)
    # print("Text generated form Vanilla Trigram: ")
    # a = tg.generateTextFromTrigram(c,["<s>"],lastWord="</s>")
    # print(a)
    
    # bigramCounts = Counter([(word,x[i+1]) for i,word in enumerate(x[:-1])])
    # y = Counter(x)
    # print("Text generated form Laplace Bigram: ")
    # d = tg.generateTextFromLaplaceBigram(x,bigramCounts,["<s>"],lastWord="</s>",count = 5)
    # print(d)
    # print(pc.calculateProbabilityFromLaplaceBigram(y,bigramCounts,["<s>"],"</s>"))

    # trigramCounts = Counter([(word,x[i+1],x[i+2])for i,word in enumerate(x[:-2])])
    # print("Text generated form Laplace Trigram: ")
    # a = tg.generateTextFromLaplaceTrigram(x,bigramCounts,trigramCounts,["<s>"],lastWord="</s>",count = 10)
    # print(a)


    # y = vm.unkModel(x)
    
    # y = vm.unkBigramModel(x)

    # y = vm.unkTrigramModel(x)
    
    # print(y)
    
    # a = vm.vanillaUnigram(x)
    # b = vm.vanillaBigram(x)
    # c = vm.vanillaTrigram(x)
    # d = pc.calculateProbabilityInterpolation(a,b,c,["<s>","nhar"],"il-")
    # print(d)
    # e = tg.generateTextInterpolation(a,b,c,["<s>"],"</s>")
    # print(e)
    
    # trainCount = Counter(x)
    # unigramCount = vm.laplaceUnigram(x)
    # bigramCounts = Counter([(word,x[i+1]) for i,word in enumerate(x[:-1])])
    # trigramCounts = Counter([(word,x[i+1],x[i+2])for i,word in enumerate(x[:-2])])
    
    # f = tg.generateTextLaplaceInterpolation(trainCount,unigramCount,bigramCounts,b,trigramCounts,["<s>"],lastWord="</s>", count=20)
    # print(f)

if __name__ == "__main__":
    main()
