import numpy as np
import matplotlib.pyplot as plt

class OcclusionExp():
    def __init__(self, image, obstruction_size, stride, model):
        self.image = image
        self.obstruction_size = obstruction_size
        self.stride = stride
        
        # Define variable for ease of access
        self.image_row = image.shape[0]
        self.image_col = image.shape[1]
        self.image_chnl = image.shape[2]
        
        self.obstruction_row = obstruction_size[0]
        self.obstruction_col = obstruction_size[1]
        
        self.stride_row = stride[0]
        self.stride_col = stride[1]
        
        # Define batch size and instantiate batch
        self.batch_size = int(np.ceil((self.image_row-self.obstruction_row)/self.stride_row)) * int(np.ceil((self.image_col-self.obstruction_col)/self.stride_col))
        self.batch = np.zeros((self.batch_size, self.image_row, self.image_col, self.image_chnl))
        
        # Define model
        self.model = model
        
        # Flag to avoid repeatative computation
        self.is_batch_generated = False
    
    def generate_occlusion(self):    
        batch_index = 0

        for row in range(0, self.image_row-self.obstruction_row, self.stride_row):
            for col in range(0, self.image_col-self.obstruction_col, self.stride_col):
                occluded = self.image.copy()
                occluded[row:row+self.obstruction_row, col:col+self.obstruction_col, :] = 0
                self.batch[batch_index,:,:,:] = occluded[:,:,:]
                batch_index += 1

        return self.batch.astype(np.int)
    
    def predict(self):
        class_num = np.argmax(model.predict(self.image.reshape(1, self.image_row, self.image_col, self.image_chnl)))
        
        # Prepare batch if not done yet
        if not self.is_batch_generated:
            self.batch = self.generate_occlusion()
            self.is_batch_generated = True
        
        # Predict and reshape as an image
        heat_map = 1 - self.model.predict(self.batch)[:, class_num]
        heat_map = heat_map.reshape(int(np.sqrt(self.batch_size)), int(np.sqrt(self.batch_size)))
        
        return heat_map
    
    def plot_hmap(self):
        heat_map = self.predict()
        plt.imshow(heat_map)
        plt.colorbar()
        plt.show()
