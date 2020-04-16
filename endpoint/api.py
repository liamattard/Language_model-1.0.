from flask import Flask, request, render_template
from collections import Counter

import language_model.textGen as tg
import language_model.probabilityCalc as pc
import language_model.preprocessing as preprocessing
import language_model.models as models
import endpoint.tools
import numpy as np

app = Flask(__name__)

@app.route('/')
def initIndex():
    return render_template('%s.html' % "init")

@app.route('/preprocessing')
def MenuPage():
    file = request.args.get('myfile')

    cwf = open("text_files/outputs/cwf.txt", "w")
    cwf.write(file)

    preprocessing.lexiconFromCorpus(
        'text_files/corpora/'+file, 'text_files/outputs/preprocessing.txt')
    return render_template('%s.html' % "menu", value=file)

@app.route('/readFile')
def FileRead():

    return render_template('%s.html' % "preprocessing")

@app.route('/testGenMenu')
def testGenMenu():
    cwf = open("text_files/outputs/cwf.txt", "r")
    cwf = cwf.read()
    print(cwf)
    return render_template('%s.html' % "testGenMenu", value=cwf, valueOne = "Generate Text", valueTwo = "testGen")

@app.route('/probCalcMenu')
def probCalcMenu():
    cwf = open("text_files/outputs/cwf.txt", "r")
    cwf = cwf.read()
    print(cwf)
    return render_template('%s.html' % "testGenMenu", value=cwf, valueOne = "Calculate Probability", valueTwo = "probCalc")

@app.route('/testGenSubMenu')
def testGenSubMenu():
    flavour = request.args.get('flavour')
    firstOp = request.args.get('valueOne')
    secondOp = request.args.get('valueTwo')
    thirdOp = request.args.get('valueThree')

    cwf = open("text_files/outputs/cwf.txt", "r")
    cwf = cwf.read()
    print(cwf)
    return render_template('%s.html' % "testGenSubMenu", value=cwf, flavour=flavour, valueOne=firstOp, valueTwo=secondOp, valueThree=thirdOp,valueFour = "testGen", valueFive = "Generate text", valueSix = "Generate")

@app.route('/probCalcIntSubMenu')
def intprobCalcSubMenu():
    
    flavour = request.args.get('flavour')
    firstOp = request.args.get('valueOne')
    secondOp = request.args.get('valueTwo')
    thirdOp = request.args.get('valueThree')

    cwf = open("text_files/outputs/cwf.txt", "r")
    cwf = cwf.read()
    print(cwf)
    return render_template('%s.html' % "testGenSubMenu", value=cwf, flavour=flavour, valueOne=firstOp, valueTwo=secondOp, valueThree=thirdOp,valueFour = "probCalc", valueFive = "Calculate Probabaility", valueSix = "IntCalculate")

@app.route('/testGenIntSubMenu')
def inttestGenSubMenu():
    
    flavour = request.args.get('flavour')
    firstOp = request.args.get('valueOne')
    secondOp = request.args.get('valueTwo')
    thirdOp = request.args.get('valueThree')

    cwf = open("text_files/outputs/cwf.txt", "r")
    cwf = cwf.read()
    print(cwf)
    return render_template('%s.html' % "testGenSubMenu", value=cwf, flavour=flavour, valueOne=firstOp, valueTwo=secondOp, valueThree=thirdOp,valueFour = "testGen", valueFive = "Generate Text", valueSix = "IntGenerate")


@app.route('/probCalcSubMenu')
def probCalcSubMenu():
    flavour = request.args.get('flavour')
    firstOp = request.args.get('valueOne')
    secondOp = request.args.get('valueTwo')
    thirdOp = request.args.get('valueThree')

    cwf = open("text_files/outputs/cwf.txt", "r")
    cwf = cwf.read()
    print(cwf)
    return render_template('%s.html' % "testGenSubMenu", value=cwf, flavour=flavour, valueOne=firstOp, valueTwo=secondOp, valueThree=thirdOp, valueFour = "probCalc", valueFive = "Calculate Probabaility", valueSix = "Calculate")

@app.route('/vanillaUnigramGenerate')
def vanillaUnigramGen():

    train, y = preprocessing.train_split(
        'text_files/outputs/preprocessing.txt')
    del y

    vanillaUnigram = models.vanillaUnigram(train)

    first,last,count = endpoint.tools.stringParser(request.args.get('firstword'),request.args.get('lastword'),request.args.get('count'))

    sentence = tg.generateTextFromUnigram(vanillaUnigram, first, lastWord=last, count=count)

    
    fullSentence = ""
    for word in sentence:
        fullSentence += " " + word

    return render_template('%s.html' % "textGen",value = fullSentence,type="vanillaUnigram",flavour="vanilla",back="testGen")

@app.route('/vanillaBigramGenerate')
def vanillaBigramGenerator():
    train, y = preprocessing.train_split(
        'text_files/outputs/preprocessing.txt')
    del y
    vanillaBigram = models.vanillaBigram(train)

    first,last,count = endpoint.tools.stringParser(request.args.get('firstword'),request.args.get('lastword'),request.args.get('count'))

    sentence = tg.generateTextFromBigram(vanillaBigram,first,lastWord=last,count=count)

    fullSentence = ""
    for word in sentence:
        fullSentence += " " + word
    return render_template('%s.html' % "textGen",value = fullSentence,type="vanillaBigram",flavour="vanilla",back="testGen")

@app.route('/vanillaTrigramGenerate')
def vanillaTrigramGenerator():
    train, y = preprocessing.train_split(
        'text_files/outputs/preprocessing.txt')
    del y
    
    vanillaBigram = models.vanillaBigram(train)
    vanillaTrigram = models.vanillaTrigram(train)

    first,last,count = endpoint.tools.stringParser(request.args.get('firstword'),request.args.get('lastword'),request.args.get('count'))

    sentence = tg.generateTextFromTrigram(vanillaBigram, vanillaTrigram,first,lastWord=last,count=count)

    fullSentence = ""
    for word in sentence:
        fullSentence += " " + word
    return render_template('%s.html' % "textGen",value = fullSentence,type="vanillaTrigram",flavour="vanilla",back="testGen")

@app.route('/laplaceUnigramGenerate')
def laplaceUnigramGenerator():
    train, y = preprocessing.train_split(
        'text_files/outputs/preprocessing.txt')
    del y
    laplaceUnigram = models.laplaceUnigram(train)

    first,last,count = endpoint.tools.stringParser(request.args.get('firstword'),request.args.get('lastword'),request.args.get('count'))

    sentence = tg.generateTextFromUnigram(laplaceUnigram,first,lastWord=last,count=count)
   
    fullSentence = ""
    for word in sentence:
        fullSentence += " " + word
    return render_template('%s.html' % "textGen",value = fullSentence, type= "laplaceUnigram",flavour="laplace",back="testGen")

@app.route('/laplaceBigramGenerate')
def laplaceBigramGenerator():
    train, y = preprocessing.train_split(
        'text_files/outputs/preprocessing.txt')
    del y
    bigramCounts = Counter([(word,train[i+1]) for i,word in enumerate(train[:-1])])

    first,last,count = endpoint.tools.stringParser(request.args.get('firstword'),request.args.get('lastword'),request.args.get('count'))
    
    sentence = tg.generateTextFromLaplaceBigram(train,bigramCounts,first,lastWord=last,count=count)

    fullSentence = ""
    for word in sentence:
        fullSentence += " " + word
    return render_template('%s.html' % "textGen",value = fullSentence, type="laplaceBigram",flavour="laplace",back="testGen")

@app.route('/laplaceTrigramGenerate')
def laplaceTrigramGenerator():
    train, y = preprocessing.train_split(
        'text_files/outputs/preprocessing.txt')
    del y
    bigramCounts = Counter([(word,train[i+1]) for i,word in enumerate(train[:-1])])
    trigramCounts = Counter([(word,train[i+1],train[i+2])for i,word in enumerate(train[:-2])])

    first,last,count = endpoint.tools.stringParser(request.args.get('firstword'),request.args.get('lastword'),request.args.get('count'))
    
    sentence = tg.generateTextFromLaplaceTrigram(train,bigramCounts,trigramCounts,first,lastWord=last,count=count)

    fullSentence = ""
    for word in sentence:
        fullSentence += " " + word
    return render_template('%s.html' % "textGen",value = fullSentence,type="laplaceTrigram",flavour="laplace",back="testGen")

@app.route('/unkUnigramGenerate')
def unkUnigramGen():
    train, y = preprocessing.train_split(
        'text_files/outputs/preprocessing.txt')
    del y
    unkUnigram = models.unkUnigram(train)
    
    first,last,count = endpoint.tools.stringParser(request.args.get('firstword'),request.args.get('lastword'),request.args.get('count'))
    
    sentence = tg.generateTextFromUnigram(unkUnigram, first, last, count=count)
                                          
    fullSentence = ""
    for word in sentence:
        fullSentence += " " + word
    return render_template('%s.html' % "textGen",value = fullSentence,type="unkUnigram",flavour="unk",back="testGen")

@app.route('/unkBigramGenerate')
def unkBigramGenerator():
    train, y = preprocessing.train_split(
        'text_files/outputs/preprocessing.txt')
    del y
    unkBigram = models.unkBigram(train)

        
    first,last,count = endpoint.tools.stringParser(request.args.get('firstword'),request.args.get('lastword'),request.args.get('count'))
    sentence = tg.generateTextFromBigram(unkBigram,first,lastWord=last,count=count)

    fullSentence = ""
    for word in sentence:
        fullSentence += " " + word
    return render_template('%s.html' % "textGen",value = fullSentence,type="unkBigram",flavour="unk",back="testGen")

@app.route('/unkTrigramGenerate')
def unkTrigramGenerator():
    train, y = preprocessing.train_split(
        'text_files/outputs/preprocessing.txt')
    del y
    unkBigram = models.unkBigram(train)
    unkTrigram = models.unkTrigram(train)

    first,last,count = endpoint.tools.stringParser(request.args.get('firstword'),request.args.get('lastword'),request.args.get('count'))
    
    sentence = tg.generateTextFromTrigram(unkBigram,unkTrigram,first,lastWord=last,count=count)

    fullSentence = ""
    for word in sentence:
        fullSentence += " " + word
    return render_template('%s.html' % "textGen",value = fullSentence,type="unkTrigram",flavour="unk",back="testGen")

@app.route('/vanillaUnigramCalculate')
def vanillaUnigramCalc():

    train, y = preprocessing.train_split(
        'text_files/outputs/preprocessing.txt')
    del y

    vanillaUnigram = models.vanillaUnigram(train)

    first,last = endpoint.tools.stringParserNoCount(request.args.get('firstword'),request.args.get('lastword'))
    if first == "" and last == "":
        probability = 0
    else:
        probability = pc.calculateProbabilityFromUnigram(vanillaUnigram,first,last)
    

    return render_template('%s.html' % "probabilityCalc",value = str(probability),type="vanillaUnigram",flavour="vanilla",back="probCalc")

@app.route('/vanillaBigramCalculate')
def vanillaBigramCalc():

    train, y = preprocessing.train_split(
        'text_files/outputs/preprocessing.txt')
    del y

    vanillaBigram = models.vanillaBigram(train)
    print(request.args.get('count'))
    first,last = endpoint.tools.stringParserNoCount(request.args.get('firstword'),request.args.get('lastword'))
    if first == "" and last == "":
        probability = 0
    else:
        probability = pc.calculateProbabilityFromBigram(vanillaBigram,first,last)
    
    return render_template('%s.html' % "probabilityCalc",value = str(probability),type="vanillaBigram",flavour="vanilla",back="probCalc")

@app.route('/vanillaTrigramCalculate')
def vanillaTrigramCalc():

    train, y = preprocessing.train_split(
        'text_files/outputs/preprocessing.txt')
    del y

    vanillaTrigram = models.vanillaTrigram(train)

    first,last = endpoint.tools.stringParserNoCount(request.args.get('firstword'),request.args.get('lastword'))
    if first == "" and last == "":
        probability = 0
    else:
        probability = pc.calculateProbabilityFromTrigram(vanillaTrigram,first,last)
    
    return render_template('%s.html' % "probabilityCalc",value = str(probability),type="vanillaTrigram",flavour="vanilla",back="probCalc")

@app.route('/laplaceUnigramCalculate')
def laplaceUnigramCalc():

    train, y = preprocessing.train_split(
        'text_files/outputs/preprocessing.txt')
    del y

    laplace = models.laplaceUnigram(train)

    first,last = endpoint.tools.stringParserNoCount(request.args.get('firstword'),request.args.get('lastword'))
    if first == "" and last == "":
        probability = 0
    else:
        probability = pc.calculateProbabilityFromUnigram(laplace,first,last)
    

    return render_template('%s.html' % "probabilityCalc",value = str(probability),type="laplaceUnigram",flavour="laplace",back="probCalc")

@app.route('/laplaceBigramCalculate')
def laplaceBigramCalc():

    train, y = preprocessing.train_split(
        'text_files/outputs/preprocessing.txt')
    del y

    trainCount = Counter(train)
    bigramCounts = Counter([(word,train[i+1]) for i,word in enumerate(train[:-1])])

    first,last = endpoint.tools.stringParserNoCount(request.args.get('firstword'),request.args.get('lastword'))
    if first == "" and last == "":
        probability = 0
    else:
        probability = pc.calculateProbabilityFromLaplaceBigram(trainCount,bigramCounts,first,last)
    

    return render_template('%s.html' % "probabilityCalc",value = str(probability),type="laplaceBigram",flavour="laplace",back="probCalc")

@app.route('/laplaceTrigramCalculate')
def laplaceTrigramCalc():

    train, y = preprocessing.train_split(
        'text_files/outputs/preprocessing.txt')
    del y

    trainCount = Counter(train)
    bigramCounts = Counter([(word,train[i+1]) for i,word in enumerate(train[:-1])])
    trigramCounts = Counter([(word,train[i+1],train[i+2])for i,word in enumerate(train[:-2])])
    
    first,last = endpoint.tools.stringParserNoCount(request.args.get('firstword'),request.args.get('lastword'))
    if first == "" and last == "":
        probability = 0
    else:
        probability = pc.calculateProbabilityFromLaplaceTrigram(trainCount,bigramCounts,trigramCounts,first,last)
    

    return render_template('%s.html' % "probabilityCalc",value = str(probability),type="laplaceTrigram",flavour="laplace",back="probCalc")

@app.route('/unkUnigramCalculate')
def unkUnigramCalc():

    train, y = preprocessing.train_split(
        'text_files/outputs/preprocessing.txt')
    del y

    unkUnigram = models.unkUnigram(train)

    first,last = endpoint.tools.stringParserNoCount(request.args.get('firstword'),request.args.get('lastword'))
    if first == "" and last == "":
        probability = 0
    else:
        probability = pc.calculateProbabilityFromUnigram(unkUnigram,first,last)
    

    return render_template('%s.html' % "probabilityCalc",value = str(probability),type="unkUnigram",flavour="unk",back="probCalc")

@app.route('/unkBigramCalculate')
def unkBigramCalc():

    train, y = preprocessing.train_split(
        'text_files/outputs/preprocessing.txt')
    del y

    unkBigram = models.unkBigram(train)
    print(request.args.get('count'))
    first,last = endpoint.tools.stringParserNoCount(request.args.get('firstword'),request.args.get('lastword'))
    if first == "" and last == "":
        probability = 0
    else:
        probability = pc.calculateProbabilityFromBigram(unkBigram,first,last)
    
    return render_template('%s.html' % "probabilityCalc",value = str(probability),type="unkBigram",flavour="unk",back="probCalc")

@app.route('/unkTrigramCalculate')
def unkTrigramCalc():

    train, y = preprocessing.train_split(
        'text_files/outputs/preprocessing.txt')
    del y

    unkTrigram = models.unkTrigram(train)

    first,last = endpoint.tools.stringParserNoCount(request.args.get('firstword'),request.args.get('lastword'))
    if first == "" and last == "":
        probability = 0
    else:
        probability = pc.calculateProbabilityFromTrigram(unkTrigram,first,last)
    
    return render_template('%s.html' % "probabilityCalc",value = str(probability),type="unkTrigram",flavour="unk",back="probCalc")

@app.route('/vanillaIntCalculate')
def interpolationVanillaCalc():

    train, y = preprocessing.train_split(
        'text_files/outputs/preprocessing.txt')
    del y

    vanillaUnigram = models.vanillaUnigram(train)
    vanillaBigram = models.vanillaBigram(train)
    vanillaTrigram = models.vanillaTrigram(train)


    first,last = endpoint.tools.stringParserNoCount(request.args.get('firstword'),request.args.get('lastword'))
    if first == "" and last == "":
        probability = 0
    else:
        probability = pc.calculateProbabilityInterpolation(vanillaUnigram,vanillaBigram,vanillaTrigram,first,last)
    
    return render_template('%s.html' % "probabilityCalcInt",value = str(probability),type="vanillaInt",flavour="vanilla", back="probCalcInt")



@app.route('/laplaceIntCalculate')
def interpolationLaplaceCalc():

    train, y = preprocessing.train_split(
        'text_files/outputs/preprocessing.txt')
    del y

    trainCount = Counter(train)
    unigramCount = models.laplaceUnigram(train)
    bigramCounts = Counter([(word,train[i+1]) for i,word in enumerate(train[:-1])])
    trigramCounts = Counter([(word,train[i+1],train[i+2])for i,word in enumerate(train[:-2])])
    
    first,last = endpoint.tools.stringParserNoCount(request.args.get('firstword'),request.args.get('lastword'))
    if first == "" and last == "":
        probability = 0
    else:
        probability = pc.calculateProbabilityLaplaceInterpolation(trainCount,unigramCount,bigramCounts,trigramCounts,first,last)
    
    return render_template('%s.html' % "probabilityCalcInt",value = str(probability),type="laplaceInt",flavour="laplace",back= "probCalcInt")

@app.route('/unkIntCalculate')
def interpolationUnkCalc():

    train, y = preprocessing.train_split(
        'text_files/outputs/preprocessing.txt')
    del y

    unkUnigram = models.unkUnigram(train)
    unkBigram = models.unkBigram(train)
    unkTrigram = models.unkTrigram(train)


    first,last = endpoint.tools.stringParserNoCount(request.args.get('firstword'),request.args.get('lastword'))
    if first == "" and last == "":
        probability = 0
    else:
        probability = pc.calculateProbabilityInterpolation(unkUnigram,unkBigram,unkTrigram,first,last)
    
    return render_template('%s.html' % "probabilityCalcInt",value = str(probability),type="unkInt",flavour="unk",back="probCalcInt")


@app.route('/vanillaIntGenerate')
def interpolationVanillaGen():

    train, y = preprocessing.train_split(
        'text_files/outputs/preprocessing.txt')
    del y


    vanillaUnigram = models.vanillaUnigram(train)
    vanillaBigram = models.vanillaBigram(train)
    vanillaTrigram = models.vanillaTrigram(train)


    first,last,count = endpoint.tools.stringParser(request.args.get('firstword'),request.args.get('lastword'),request.args.get('count'))

    sentence = tg.generateTextInterpolation(vanillaUnigram,vanillaBigram,vanillaTrigram, first, lastWord=last, count=count)

    
    fullSentence = ""
    for word in sentence:
        fullSentence += " " + word

    return render_template('%s.html' % "textGenInt",value = fullSentence,type="vanillaInt",flavour="vanilla",back="testGenInt")


@app.route('/laplaceIntGenerate')
def interpolationlaplaceGen():

    train, y = preprocessing.train_split(
        'text_files/outputs/preprocessing.txt')
    del y


    vanillaBigram = models.vanillaBigram(train)

    trainCount = Counter(train)
    unigramCount = models.laplaceUnigram(train)
    bigramCounts = Counter([(word,train[i+1]) for i,word in enumerate(train[:-1])])
    trigramCounts = Counter([(word,train[i+1],train[i+2])for i,word in enumerate(train[:-2])])
    

    first,last,count = endpoint.tools.stringParser(request.args.get('firstword'),request.args.get('lastword'),request.args.get('count'))

    sentence = tg.generateTextLaplaceInterpolation(trainCount,unigramCount,bigramCounts,vanillaBigram,trigramCounts, first, lastWord=last, count=count)

    
    fullSentence = ""
    for word in sentence:
        fullSentence += " " + word

    return render_template('%s.html' % "textGenInt",value = fullSentence,type="laplaceInt",flavour="laplace",back="testGenInt")

@app.route('/unkIntGenerate')
def interpolationUnkGen():

    train, y = preprocessing.train_split(
        'text_files/outputs/preprocessing.txt')
    del y


    unkUnigram = models.unkUnigram(train)
    unkBigram = models.unkBigram(train)
    unkTrigram = models.unkTrigram(train)


    first,last,count = endpoint.tools.stringParser(request.args.get('firstword'),request.args.get('lastword'),request.args.get('count'))

    sentence = tg.generateTextInterpolation(unkUnigram,unkBigram,unkTrigram, first, lastWord=last, count=count)

    
    fullSentence = ""
    for word in sentence:
        fullSentence += " " + word

    return render_template('%s.html' % "textGenInt",value = fullSentence,type="unkInt",flavour="unk",back="testGenInt")


app.run(host='127.0.0.1', port=8081)