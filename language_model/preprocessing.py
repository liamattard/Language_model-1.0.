import numpy as np

from sklearn.model_selection import train_test_split


def lexiconFromCorpus(corpus, output):
    '''
    Carries out preprocessing on corpus by adding sentence tags '<s>'
    and '</s>' on end statement punctuation (ie: '?', '!', '.') and
    removing non end statement punctuation (ie: ',', ':', ';') and 
    outputs the result to output

    Parameters
    ----------
    corpus:string Raw corpus text file
    output:string Output Processed text file
    '''

    corpus = open(corpus, "r")

    output = open(output, "w")

    html = open("endpoint/templates/preprocessing.html", "w")

    output.write(("<s> "))

    html.write("""<!DOCTYPE html>
        <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{value}}</title>
    </head>
    <body>
    <div class='table'>
        <div class='cell'>
               <form >""")
    html.write(("< s > "))

    SentenceBegin = False

    for word in corpus:
        if word == '</s>\n':
            output.write(("</s> "))
            html.write(("< / s> "))
            SentenceBegin = True
        if (word[0] != '<' and word[-2] != '>'):
            if SentenceBegin == True:
                output.write(("<s> "))
                html.write(("< s > "))
                SentenceBegin = False
            x = word.split('\t')
            if x[0] not in [',', ':', ';', ".", '?', '!', "\"", "(", ")"]:
                output.write((x[0] + " ").lower())
                html.write((x[0] + " ").lower())
    html.write("""
           </form >
            </div>
    </div>
    </body>
    </html""")
    print("Processed output complete")


def train_split(file):
    '''
    Split the corpus into the different sets – training, test – the split should be random.

    Parameters
    ----------
    file:string Text file to load and split

    Returns
    ----------
    train:arr Array of strings with training data
    test:arr Aray of strings with testing data


   '''
    file = open(file, "r")
    for line in file:
        words = np.array(line.split())

    train, test = train_test_split(words, shuffle=False)
    return train, test
