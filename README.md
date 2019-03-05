# LEMON

**LEMON** stands for **LE**xicon-**M**em**O**ry-augmented-**N**er.

## Requirements
- Python: `3.6` or higher
- PyTorch: `1.0`

## Dataset
The dataset can be configured in the `dataset.py`.

A mixed version of the CoNLL format, with each character and its label(segmentation, POS-tag and NER-tag) for one line. 
Sentences are split with a null line.

```
中 B B-ni B-ORG
共 M M-ni M-ORG
中 M M-ni M-ORG
央 E E-ni E-ORG
致 S S-v O
中 B B-ns B-ORG
国 E E-ns M-ORG
致 B B-ni M-ORG
公 M M-ni M-ORG
党 E E-ni M-ORG
十 B B-j M-ORG
一 M M-j M-ORG
大 E E-j E-ORG
的 S S-u O
贺 B B-n O
词 E E-n O
```

##Pretrained Embeddings

- Character embeddings: [gigaword_chn.all.a2b.uni.ite50.vec](https://pan.baidu.com/s/1pLO6T9D)
- Lexicon embeddings: [ctb.50d.vec](https://pan.baidu.com/s/1pLO6T9D)

##Running
```
python main.py
```