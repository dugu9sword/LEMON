import xml.etree.ElementTree as ET

tree = ET.ElementTree(file='../dataset/msra/msra_bakeoff3_test.xml')
# tree = ET.ElementTree(file='../dataset/msra/test.xml')

tags = {"ORGANIZATION": "ORG", "PERSON": "PER", "LOCATION": "LOC"}
# print(tree.getroot())
# out = None
out = open("../dataset/msra/test.ner.bmes.gold", "w")
seg_out = open("../dataset/msra/test.seg.bmes.gold", "w")
# seg_out = None
# out= None
lengths = {}


def output(chars, tag):
    if chars is None or len(chars) == 0:
        return
    if tag not in tags:
        for char in chars:
            print(char, "O", file=out)
    else:
        if len(chars) not in lengths:
            lengths[len(chars)] = 1
        else:
            lengths[len(chars)] += 1
        if len(chars) == 1:
            print(chars, "S-{}".format(tags[tag]), file=out)
        else:
            print(chars[0], "B-{}".format(tags[tag]), file=out)
            if len(chars) > 2:
                for i in range(1, len(chars) - 1):
                    print(chars[i], "M-{}".format(tags[tag]), file=out)
            print(chars[-1], "E-{}".format(tags[tag]), file=out)

def output_seg(chars):
    if chars is None or len(chars) == 0:
        return
    if len(chars) not in lengths:
        lengths[len(chars)] = 1
    else:
        lengths[len(chars)] += 1
    if len(chars) == 1:
        print(chars, "S", file=seg_out)
    else:
        print(chars[0], "B", file=seg_out)
        if len(chars) > 2:
            for i in range(1, len(chars) - 1):
                print(chars[i], "M", file=seg_out)
        print(chars[-1], "E", file=seg_out)


line_num = 0
for sentence in tree.getroot():
    print(line_num)
    line_num += 1
    for words in sentence:
        seg_chars = "".join(words.itertext())
        output_seg(seg_chars)
        if len(words) > 0:
            if words.text:
                output(words.text, None)
            for word in words:
                chars = word.text
                tag = word.attrib['TYPE']
                output(chars, tag)
                if word.tail:
                    output(word.tail, None)
        elif len(words) == 0:
            tag = None
            chars = words.text
            output(chars, tag)
        else:
            raise Exception

    print(file=out)
    print(file=seg_out)

    # exit()
print(lengths)