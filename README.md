# Monai 사용법
```python
MetaTensor에 포함된 tensor는 tensor.get_array()로 구한다.
```

output quality check
```python
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
```
