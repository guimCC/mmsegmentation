import torch
import torch.nn as nn
import torch.nn.functional as F

from mIoUv2 import IoU


#dimensions_i = [1, 19, 512, 1024]
#dimensions_t = [1, 512, 1024]
dimensions_i = [2, 2, 10, 10]
dimensions_t = [2, 10, 10]

# Create an input of type torch.float32 with all the pixels one class (1)
# This means that from the second dimension, all values of index 1 are 1

pred_tensor = torch.zeros(dimensions_i, dtype=torch.float32) # Empty input tensor

#pred_tensor0 = torch.zeros(dimensions_i, dtype=torch.float32) # Empty input tensor
pred_tensor[:, 0, :, :] = 1 # Set all the pixels of class 1 to 1

#pred_tensor[1, 0, :, dimensions_i[3]//2:] = 1 # Set half the pixels of class 1 to 1
#pred_tensor[1, 1, :, :(dimensions_i[3]+1)//2] = 1 # Set half the pixels of class 1 to 1

# Create a target of type torch.int64 with all the pixels one class (1)
target_tensor = torch.zeros(dimensions_t, dtype=torch.int64) # Empty target tensor
target_tensor[0, :, :] = 1 # Set all the pixels of class 1 to 1

# Compute the mIoU:
#eps = 1e-6
eps = 1e-12

#metric = IoU(num_classes=2)

#metric.reset()
#metric.add(pred_tensor, target_tensor)
m = IoU(num_classes=2)

# m.batch_intersection_union(pred_tensor, target_tensor, 2)

# print("Results:", m.results)
# a, b, c, d = m.compute_total()
# print(a, b, c, d)

# i, m, = m.iou(a, b, c, d)
print(m.compute_reward(pred_tensor, target_tensor))


#print("MIoU loss metric:", metric.value())

#plot_tensor(target_tensor)

#pred = F.softmax(pred_tensor, dim=1) # Convert to probabilities (not necessary?)
#pred = torch.argmax(pred, dim=1) # Convert to class labels

#plot_tensor(pred)