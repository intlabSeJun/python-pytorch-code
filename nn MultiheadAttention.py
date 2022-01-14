import torch
import torch.nn as nn
#https://www.youtube.com/watch?v=_iDanMWVj98
#transformer에서 self attention 과 같다. 하지만 norm등의 추가는 없고 쿼리,키,벨류에 대해서 연산한 것.
"""
query, key, value에 대한 통합된 weight를 만들고 각 wieght를 곱해주고 softmax 취해주고 벨류 곱해주고임. 
"""