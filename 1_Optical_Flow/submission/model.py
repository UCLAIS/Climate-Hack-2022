import numpy as np
import pandas as pd
import cv2
import torch

import numpy as np
import pandas as pd
import cv2
import torch

import numpy as np
import pandas as pd
import cv2
import torch

class Model:
    
    NUM_WARM_UP_IMAGES = 12
    NUM_PREDICTION_TIMESTEPS = 24
    
    
    def __init__(self, batch_input):
        
        self.batch_input = batch_input
        
    def compute_flows(self, **kwargs):
        
        flows = []
        
        for image_i in range(self.NUM_WARM_UP_IMAGES):
            flow = cv2.calcOpticalFlowFarneback(
                prev=self.batch_input[image_i-1], next=self.batch_input[image_i], flow=None, **kwargs)
            flows.append(flow)
        return np.stack(flows).astype(np.float32)
    
    def weighted_average(self, flow):
        return np.average(flow, axis=0, weights=range(1, self.NUM_WARM_UP_IMAGES+1)).astype(np.float32)
        
    def remap_image(self, image, flow):

        height, width = flow.shape[:2]
        remap = -flow.copy()
        remap[..., 0] += np.arange(width)  # x map
        remap[..., 1] += np.arange(height)[:, np.newaxis]  # y map
        remapped_image = cv2.remap(src=image, map1=remap, map2=None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        return cv2.resize(remapped_image, (64, 64))
    
    def generate(self):
        
        targets = []
        
        start_image = self.batch_input[-1]
        
        flows_default = self.compute_flows(pyr_scale=0.5, levels=2, winsize=40, 
        iterations=3, poly_n=5, poly_sigma=0.7, 
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        flow_default = self.weighted_average(flows_default)
        
        for i in range(self.NUM_PREDICTION_TIMESTEPS):
            remapped_image = self.remap_image(start_image, flow_default * i)
            
            targets.append(remapped_image)
            
        return np.array(targets)
