# import torch
# import torch.nn.functional as F
# from buff import focal_loss
# from seq_encoder import FofeSeqEncoder, RNNSeqEncoder
import re

raw_str = """
<ENAMEX TYPE="ORG" E_OFF="1">亚太 经济 暨 合作 会议</ENAMEX> 
"""

def extract_ner(raw_str):
    ret = []
    found = re.findall(r'(?:<ENAMEX TYPE="[^"]*">[^<]+</ENAMEX>|[^\s]+)', raw_str)
    for ele in found:
        entity = re.search(r'<ENAMEX TYPE="([^"]*)">([^<]+)</ENAMEX>', ele)
        if entity:
            entities = re.split(r"\s+", entity.group(2))
            if len(entities) == 1:
                ret.append(('S-{}'.format(entity.group(1)), entities[0]))
            else:
                ret.append(('B-{}'.format(entity.group(1)), entities[0]))
                if len(entities) >= 3:
                    for i in range(1, len(entities) - 1):
                        ret.append(('M-{}'.format(entity.group(1)), entities[i]))
                ret.append(('E-{}'.format(entity.group(1)), entities[-1]))
        else:
            ret.append(("O", ele))
    return ret
print(ret)
# print(found)
# print(found.group(0))
#
# print(found.group(1))
#
# print(found.group(2))

# tags = []
# ptr = 0
# while True:
#     while raw_str[ptr] == ' ':
#         ptr += 1
#     start_ptr = ptr
#     if raw_str[ptr] == '<':
#         ptr += 13
#
#         while raw_str[ptr] != ' ':
#             ptr += 1
