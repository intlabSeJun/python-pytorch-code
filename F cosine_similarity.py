import torch
import torch.nn.functional as F

a = torch.randn(1,3,1,3)
b = torch.randn(1,1,4,3)
cos = F.cosine_similarity(a,b,dim=3)
print(cos)
# dim=3이면 a의 마지막 3개 요소를 갖는 벡터가 b의 (4,3) 각각에 대해 cosin 유사도 구하고 이를 a의 3번 반복.