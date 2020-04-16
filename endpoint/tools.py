import urllib.parse

def stringParser(first,last,count):
    
    if first == "":
        first = ["<s>"]
    else:
        first = urllib.parse.unquote(first)
        first = first.split()

    if last == "":
        last = "</s>"
    else:
        last = urllib.parse.unquote(last)

    if count == "":
        count = 20
    else:
        count = int(count)

    return first, last, count

def stringParserNoCount(first,last):
    
    if first != "":
        first = urllib.parse.unquote(first)
        first = first.split()

    if last != "":

        last = urllib.parse.unquote(last)


    return first, last
