# conv 계산

2D convolution using a kernel size of 3, stride of 1 and padding,

```python
#### conv
nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1) --> output size = input size

#### upsample
nn.ConvTranspose2d(in_feature, out_feature, kernel_size=4, stride=2, padding=1) -->  output size = 2*(input size)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

#### downsample
nn.Sequential(nn.AvgPool2d(2,2), conv3x3(planes, planes))
```

#### FID score
"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectivly."""

<img src="https://github.com/sandokim/monai_tip/blob/main/images/inceptionscore.png" width="60%">

<img src="https://github.com/sandokim/monai_tip/blob/main/images/inceptionargs.png" width="60%">

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

# model.pkl

텍스트 상태의 데이터가 아닌 파이썬 객체 자체를 파일로 저장하는 것을 말한다.

이 때 원하는 객체 자체를 바이너리로 저장해놓는 것이고 필요할 때 불러오기만 하면 되기 때문에 속도가 빠르다는 장점이 있다.

```python
import pickle
my_list = ['a','b','c'] 

## Save pickle
with open("data.pickle","wb") as fw:    
    pickle.dump(my_list, fw) 
    
## Load pickle
with open("data.pickle","rb") as fr:
    data = pickle.load(fr)
   
print(data) 
#['a', 'b', 'c']
```
