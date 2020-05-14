import numpy as np
import matplotlib.pyplot as plt

class visualize_filters():
    def __init__(self, model):
        self.model = model
        self.layer_index = None
    
    # Define layer to index mapping
    def layer_index_mapping(self):
        print('Index', '\t','Layer')
        for index, layer in enumerate(self.model.layers):
            print(index, '\t',layer.get_config()['name'])
            
    def deprocess_image(self, image):
        # normalize tensor: center on 0., ensure std is 0.1
        image -= image.mean()
        image /= (image.std() + 1e-5)
        image *= 0.1

        # clip to [0, 1]
        image += 0.5
        image = np.clip(image, 0, 1)

        # convert to RGB array
        image *= 255
        image = np.clip(image, 0, 255).astype('uint8')
        return image
    
    # Retrieve index corresponding to the input layer name
    def get_layer_index(self, layer_name):
        for index, layer in enumerate(self.model.layers):
            if layer.get_config()['name'] == layer_name:
                return index
    
    # Retrieve weights corresponding to layer index
    def get_layer_weights(self):
        for index, layer in enumerate(self.model.layers):
            if index == self.layer_index:
                return layer.get_weights()[0]
    
    def get_batch_dims(self, weights):
        # Return dimensions of the batch as: img_rows x img_cols x input_channels x output_filters
        return weights.shape[0], weights.shape[1], weights.shape[2], weights.shape[3]
    
    # View the filters
    def view(self, layer_index, filter=None, num_cols=None, figsize=(10,10), filter_limit=None):        
        if type(layer_index) == int:
            self.layer_index = layer_index
        else:
            self.layer_index = self.get_layer_index(layer_index) 
        
        try:
            weights = self.get_layer_weights()
            print('Layer Shape =', weights.shape)
            self.plot(weights, num_cols, figsize, filter_limit)
        except Exception as e:
            print('Try again with a valid filter!')
    
    def get_plotting_dims(self, weights, num_cols):
        if weights.shape[2] == 3:
            if not num_cols:
                num_cols = int(np.sqrt(weights.shape[3]))
                num_rows = int(np.ceil(weights.shape[3]/num_cols))
                return num_rows, num_cols
            else:
                num_rows = int(np.ceil(weights.shape[3]/num_cols))
                return num_rows, num_cols
        
        else:
            if not num_cols:
                return weights.shape[3], weights.shape[2]
            else:
                num_rows = int(np.ceil(weights.shape[3] * weights.shape[2])/num_cols)
                return num_rows, num_cols
            
    def plot(self, weights, num_cols, figsize, filter_limit):
        img_row, img_col, input_channel, output_filters = self.get_batch_dims(weights)
        num_rows, num_cols = self.get_plotting_dims(weights, num_cols)
        
        if filter_limit:
            num_rows, num_cols = min(filter_limit, num_rows), min(filter_limit, num_cols)
            input_channel, output_filters = min(filter_limit, input_channel), min(filter_limit, output_filters)
        
        # plot grid
        if input_channel == 3:
            fig = plt.figure(figsize = figsize)
            for i in range(output_filters):
                image = self.deprocess_image(weights[:,:,:,i].reshape(img_row, img_row, 3).astype(np.float))
                ax = fig.add_subplot(num_rows, num_cols, i+1)
                ax.title.set_text('Filter: ' + str(i+1))
                plt.imshow(image)
            plt.show()
        
        else:
            index = 1
            fig = plt.figure(figsize = figsize)
            for i in range(output_filters):
                for j in range(input_channel):
                    image = self.deprocess_image(weights[:,:,j,i].reshape(img_row, img_row).astype(np.float))
                    ax = fig.add_subplot(num_rows, num_cols, index)
                    ax.title.set_text('Filter: ' + str(index))
                    plt.imshow(image)
                    
                    index += 1
            plt.show()
