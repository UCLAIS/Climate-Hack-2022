from datetime import datetime, time, timedelta
from functools import lru_cache
from random import randrange
from typing import Iterator, T_co

import numpy as np
import xarray as xr
from numpy import float32

class ClimateHackDataset():
    def __init__(
        self,
        data_path: str,
        crops_per_slice: int = 1,
    ) -> None:
        self.crops_per_slice = crops_per_slice
        self.coordinates = []
        self.features = []
        self.labels = []
        
        self._process_data(np.load(data_path))
        
    def load_data(self):
        return np.array(self.coordinates), np.array(self.features), np.array(self.labels)        

    def _process_data(self, data):
        print("Processing...")
        self.osgb_data = np.stack(
            [
                data["x_osgb"],
                data["y_osgb"],
            ]
        )

        for day in data["data"]:
            for i in range(0, 85 - 1, 1):
                input_slice = day[i : i + 1, :, :]
                target_slice = day[i + 1 : i + 2, :, :]
                crops = 0
                while crops < self.crops_per_slice:
                    crop = self._get_crop(input_slice, target_slice)
                    if crop:
                        (osgb_data, input_data, target_data) = crop
                        self.coordinates += osgb_data,
                        self.features += input_data,
                        self.labels += target_data,

                    crops += 1

    def _get_crop(self, input_slice, target_slice):
        # roughly over the mainland UK
        rand_x = randrange(550, 950 - 128)
        rand_y = randrange(375, 700 - 128)

        # get the input satellite imagery
        osgb_data = self.osgb_data[:, rand_y : rand_y + 128, rand_x : rand_x + 128]
        input_data = input_slice[:, rand_y : rand_y + 128, rand_x : rand_x + 128]
        target_data = target_slice[
            :, rand_y +32 : rand_y + 96, rand_x + 32: rand_x + 96
        ]

        if input_data.shape != (1, 128, 128) or target_data.shape != (1, 64, 64):
            return None

        return osgb_data, input_data, target_data

#     def __iter__(self) -> Iterator[T_co]:
#         for item in self.cached_items:
#             yield item
