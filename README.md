# Monai 사용법
```python
MetaTensor에 포함된 tensor는 tensor.get_array()로 구한다.
```

output quality check
```python
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,8))
plt.subplot(3,3,1)
plt.imshow(seg.get_array()[0,0,:,:,100])
plt.subplot(3,3,2)
plt.imshow(seg.get_array()[0,1,:,:,100])
plt.subplot(3,3,3)
plt.imshow(seg.get_array()[0,2,:,:,100])
plt.subplot(3,3,4)
plt.imshow(seg.get_array()[0,3,:,:,100])
plt.subplot(3,3,5)
plt.imshow(seg.get_array()[0,4,:,:,100])
plt.subplot(3,3,6)
plt.imshow(seg.get_array()[0,5,:,:,100])
plt.subplot(3,3,7)
plt.imshow(seg.get_array()[0,6,:,:,100])
plt.subplot(3,3,8)
plt.imshow(seg.get_array()[0,7,:,:,100])
plt.subplot(3,3,9)
plt.imshow(seg.get_array()[0,8,:,:,100])
plt.savefig('./_qc')

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,8))
plt.subplot(3,3,1)
plt.imshow(seg.get_array()[0,:,:,100])
plt.savefig('./_qc')

print(np.unique(seg.get_array()[0,0,:,:,:]))
print(np.unique(seg.get_array()[0,1,:,:,:]))
print(np.unique(seg.get_array()[0,2,:,:,:]))
print(np.unique(seg.get_array()[0,3,:,:,:]))
print(np.unique(seg.get_array()[0,4,:,:,:]))
print(np.unique(seg.get_array()[0,5,:,:,:]))
print(np.unique(seg.get_array()[0,6,:,:,:]))
print(np.unique(seg.get_array()[0,7,:,:,:]))
print(np.unique(seg.get_array()[0,8,:,:,:]))

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,8))
plt.subplot(3,3,1)
plt.imshow(label.get_array()[0,0,:,:,100])
plt.subplot(3,3,2)
plt.imshow(label.get_array()[0,1,:,:,100])
plt.subplot(3,3,3)
plt.imshow(label.get_array()[0,2,:,:,100])
plt.subplot(3,3,4)
plt.imshow(label.get_array()[0,3,:,:,100])
plt.subplot(3,3,5)
plt.imshow(label.get_array()[0,4,:,:,100])
plt.subplot(3,3,6)
plt.imshow(label.get_array()[0,5,:,:,100])
plt.subplot(3,3,7)
plt.imshow(label.get_array()[0,6,:,:,100])
plt.subplot(3,3,8)
plt.imshow(label.get_array()[0,7,:,:,100])
plt.subplot(3,3,9)
plt.imshow(label.get_array()[0,8,:,:,100])
plt.savefig('./_qc')

print(np.unique(label.get_array()[0,0,:,:,:]))
print(np.unique(label.get_array()[0,1,:,:,:]))
print(np.unique(label.get_array()[0,2,:,:,:]))
print(np.unique(label.get_array()[0,3,:,:,:]))
print(np.unique(label.get_array()[0,4,:,:,:]))
print(np.unique(label.get_array()[0,5,:,:,:]))
print(np.unique(label.get_array()[0,6,:,:,:]))
print(np.unique(label.get_array()[0,7,:,:,:]))
print(np.unique(label.get_array()[0,8,:,:,:]))
```

label이 안나올때 Unet 마지막 단에 nn.Sigmoid()가 아니라 nn.Softmax(dim=1)을 써보자. Sigmoid를 썼을 경우, Background의 prob이 다른 레이블들보다 높게 되면, output의 segmentation이 모두 Background가 되버려 label missing현상이 생길 수 있다.

--> Unet 마지막 softmax layer 또는 sigmoid layer 제거하고 main()에서
```python
output = model(input)
softmax = nn.Softmax(dim=1)
output = softmax(output)
```

Dice 수치가 올라가지 않거나 Loss가 떨어지지 않는다면 softmax나 sigmoid가 연속적으로 적용된 것이 아닌지 확인해보자

보통 모델 output은 feature map의 Raw 상태 사용하고, Loss를 계산하는 부분에서 Loss내에서 softmax나 sigmoid를 취한다.
