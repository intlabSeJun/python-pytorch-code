import torch


# contiguous를 통한 shuffle 예시.
""" 
아래와 같이 batch=1, (6,2)인 매트릭스가 있다면,
random으로 섞지 않고 고정된 위치가 바뀌도록 간단하게 할 수 있다.
맨 앞, 맨 뒤 행은 고정되고 중간 행들이 고정적인 형태로 섞이게 된다. 
"""
x = torch.randn(1,6,2)
print(x)
x = x.view(1,2,-1,2) #(1,2,3,2)
print(x)
x = torch.transpose(x,1,2).contiguous()  #(1,3,2,2) # contiguous()를 통해 주소를 새로 할당함으로써 새로운 텐서가 만들어진다.
print(x) # 시점이 바뀌게 됨.
x = x.view(1, -1, 2)
print(x)

#https://f-future.tistory.com/entry/Pytorch-Contiguous
"""
view, transpose, permute 등과 같은 원본 Tesnor 메타데이터만 변경하기 때문에 
non-contiguous Tensor이고 주소를 공유함. 

non-contiguous Tensor는 주소값 재배열 연산이 필요할 때 사용할 수 없음.

contiguous 함수로 새로운 메모리에 하당하여 contiguous Tensor로 변경하면 주소값 재배열이 가능. 
"""