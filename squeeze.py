import torch
"""
squeeze 함수는 차원이 1인 차원을 제거, 따로 차원을 설정하지 않으면 1인 차원을 모두 제거,
차원 설정시 그 차원만 제거.
주의할 점은 batch가 1일 때 batch 차원도 없애버릴 수 있다.
"""

x = torch.rand(3,1,20,128)
x = x.squeeze(1)
print(x.shape)
print(x.unsqueeze().shape)