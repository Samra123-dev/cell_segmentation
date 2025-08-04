import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
mask = np.array(Image.open('give your image path'))
plt.imshow(mask, cmap='gray')
plt.colorbar()
plt.show()
print(np.unique(mask))
