import random
import numpy as np
import language_model.probabilityCalc as pc

from collections import Counter


def generateTextFromUnigram(unigram, sentence, lastWord="", count=None):
    if(count != 0 and sentence[-1] != lastWord):

        weights = np.array(list(unigram.values()))
        normalized_weights = weights / np.sum(weights)
        resample_counts = np.random.multinomial(1, normalized_weights)
        chosenKey = list(resample_counts).index(1)
        chosenVal = list(unigram.keys())[chosenKey]
        sentence.append(chosenVal)
        if count != None:
            generateTextFromUnigram(unigram, sentence, lastWord, count-1)
        else:
            generateTextFromUnigram(unigram, sentence, lastWord)
    return sentence


def generateTextFromBigram(bigram, sentence, lastWord=" ", count=None):
    if (count != 0 and (sentence[-1] != lastWord)):
        chosenBigrams = {}

        for word in bigram:
            if word[0] == sentence[-1]:
                chosenBigrams[word] = bigram[word]
        if(chosenBigrams == []):
            return sentence

        weights = np.array(list(chosenBigrams.values()))
        normalized_weights = weights / np.sum(weights)
        resample_counts = np.random.multinomial(1, normalized_weights)
        chosenKey = list(resample_counts).index(1)
        chosenVal = list(chosenBigrams.keys())[chosenKey]
        sentence.append(chosenVal[1])

        if count != None:
            generateTextFromBigram(bigram, sentence, lastWord, count-1)
        else:
            generateTextFromBigram(bigram, sentence, lastWord)
    return sentence


def generateTextFromTrigram(bigram, trigram, sentence, lastWord="", count=None):
    if len(sentence) == 1:
        sentence = generateTextFromBigram(bigram, sentence, lastWord, count=1)

    if (count != 0 and (sentence[-1] != lastWord)):
        chosenTrigrams = {}

        for word in trigram:

            if word[0] == sentence[-2] and word[1] == sentence[-1]:
                chosenTrigrams[word] = trigram[word]
        if(chosenTrigrams == []):
            return sentence

        weights = np.array(list(chosenTrigrams.values()))
        normalized_weights = weights / np.sum(weights)
        resample_counts = np.random.multinomial(1, normalized_weights)
        chosenKey = list(resample_counts).index(1)
        chosenVal = list(chosenTrigrams.keys())[chosenKey]
        sentence.append(chosenVal[2])

        if count != None:
            generateTextFromTrigram(
                bigram, trigram, sentence, lastWord, count-1)
        else:
            generateTextFromTrigram(bigram, trigram, sentence, lastWord)
    return sentence


def generateTextFromLaplaceBigram(train, bigramCounts, sentence, lastWord="", count=None):
    if (count != 0 and (sentence[-1] != lastWord)):

        word_by_count = Counter(train)
        chosenBigrams = {}

        for word in word_by_count:
            x = (sentence[-1], word)
            if x in bigramCounts.keys():
                chosenBigrams[sentence[-1], word] = (bigramCounts[sentence[-1], word] + 1)/(
                    word_by_count[sentence[-1]] + len(word_by_count))

            else:
                chosenBigrams[sentence[-1],
                              word] = (1)/(word_by_count[sentence[-1]] + len(word_by_count))
        weights = np.array(list(chosenBigrams.values()))
        normalized_weights = weights / np.sum(weights)
        # for item in range(len(normalized_weights)):
        # output.write(str(list(chosenBigrams.kelist(ys())[item][]))
        # output.write(" ")
        # output.write("=")
        # output.write(str(normalized_weights[item]))
        # output.write("\n")

        resample_counts = np.random.multinomial(1, normalized_weights)
        chosenKey = list(resample_counts).index(1)
        chosenVal = list(chosenBigrams.keys())[chosenKey]
        sentence.append(chosenVal[1])
        if count != None:
            generateTextFromLaplaceBigram(
                train, bigramCounts, sentence, lastWord, count-1)
        else:
            generateTextFromLaplaceBigram(
                train, bigramCounts, sentence, lastWord)
    return sentence


def generateTextFromLaplaceTrigram(train, bigramCounts, trigramCounts, sentence, lastWord="", count=None):
    if len(sentence) == 1:
        sentence = generateTextFromLaplaceBigram(
            train, bigramCounts, sentence, count=1)

    if (count != 0 and (sentence[-1] != lastWord)):

        word_by_count = Counter(train)
        chosenTrigram = {}

        for word in word_by_count:
            if (sentence[-2], sentence[-1], word) in trigramCounts.keys():
                chosenTrigram[sentence[-2], sentence[-1], word] = (trigramCounts[sentence[-2], sentence[-1], word]+1)/(
                    bigramCounts[sentence[-2], sentence[-1]] + len(word_by_count))

            else:
                chosenTrigram[sentence[-2], sentence[-1], word] = (1)/(
                    bigramCounts[sentence[-2], sentence[-1]] + len(word_by_count))
        # print(sum(list(chosenTrigram.values())))
        weights = np.array(list(chosenTrigram.values()))
        normalized_weights = weights / np.sum(weights)
        resample_counts = np.random.multinomial(1, normalized_weights)
        # resample_counts = random.choices(list(chosenTrigram.keys()),weights)
        chosenKey = list(resample_counts).index(1)
        chosenVal = list(chosenTrigram.keys())[chosenKey]

        sentence.append(chosenVal[2])

        if count != None:
            generateTextFromLaplaceTrigram(
                train, bigramCounts, trigramCounts, sentence, lastWord, count-1)
        else:
            generateTextFromLaplaceTrigram(
                train, bigramCounts, trigramCounts, sentence, lastWord)
    return sentence


def generateTextInterpolation(unigram, bigram, trigram, sentence, lastWord="", count=None):
    if len(sentence) == 1:
        sentence = generateTextFromBigram(bigram, sentence, lastWord, count=1)

    if (count != 0 and (sentence[-1] != lastWord)):
        wordsDict = {}

        for word in unigram:
            wordsDict[word] = pc.calculateProbabilityInterpolation(
                unigram, bigram, trigram, sentence, word)
        weights = np.array(list(wordsDict.values()))
        normalized_weights = weights / np.sum(weights)
        resample_counts = np.random.multinomial(1, normalized_weights)
        chosenKey = list(resample_counts).index(1)
        chosenVal = list(wordsDict.keys())[chosenKey]
        sentence.append(chosenVal)
        if count != None:
            generateTextInterpolation(
                unigram, bigram, trigram, sentence, lastWord, count-1)
        else:
            generateTextInterpolation(
                unigram, bigram, trigram, sentence, lastWord)
    return sentence


def generateTextLaplaceInterpolation(trainCount, unigramCount, bigram, bigramCounts, trigramCounts, sentence, lastWord="", count=None):
    if len(sentence) == 1:
        sentence = generateTextFromBigram(bigram, sentence, lastWord, count=1)

    if (count != 0 and (sentence[-1] != lastWord)):
        wordsDict = {}

        for word in trainCount:
            wordsDict[word] = pc.calculateProbabilityLaplaceInterpolation(
                trainCount, unigramCount, bigramCounts, trigramCounts, sentence, lastWord)

        weights = np.array(list(wordsDict.values()))
        normalized_weights = weights / np.sum(weights)
        resample_counts = np.random.multinomial(1, normalized_weights)
        chosenKey = list(resample_counts).index(1)
        chosenVal = list(wordsDict.keys())[chosenKey]
        sentence.append(chosenVal)
        if count != None:
            generateTextLaplaceInterpolation(
                trainCount, unigramCount, bigram, bigramCounts, trigramCounts, sentence, lastWord, count-1)
        else:
            generateTextLaplaceInterpolation(
                trainCount, unigramCount, bigram, bigramCounts, trigramCounts, sentence, lastWord)
    return sentence
