from datetime import datetime, time, timedelta
from functools import lru_cache
from random import randrange
from typing import Iterator, T_co

import numpy as np
import xarray as xr
from numpy import float32
from torch.utils.data import IterableDataset


class ClimateHackDataset(IterableDataset):
    def __init__(
        self,
        data_path: str,
        crops_per_slice: int = 1,
    ) -> None:
        super().__init__()

        self.crops_per_slice = crops_per_slice

        self._process_data(np.load(data_path))

    def _process_data(self, data):
        self.osgb_data = np.stack(
            [
                data["x_osgb"],
                data["y_osgb"],
            ]
        )

        self.cached_items = []
        for day in data["data"]:
            for i in range(0, 85 - 24, 4):
                input_slice = day[i : i + 12, :, :]
                target_slice = day[i + 12 : i + 36, :, :]

                crops = 0
                while crops < self.crops_per_slice:
                    crop = self._get_crop(input_slice, target_slice)
                    if crop:
                        self.cached_items.append(crop)

                    crops += 1

    def _get_crop(self, input_slice, target_slice):
        # roughly over the mainland UK
        rand_x = randrange(550, 950 - 128)
        rand_y = randrange(375, 700 - 128)

        # get the input satellite imagery
        osgb_data = self.osgb_data[:, rand_y : rand_y + 128, rand_x : rand_x + 128]
        input_data = input_slice[:, rand_y : rand_y + 128, rand_x : rand_x + 128]
        target_data = target_slice[
            :, rand_y + 32 : rand_y + 96, rand_x + 32 : rand_x + 96
        ]

        if input_data.shape != (12, 128, 128) or target_data.shape != (24, 64, 64):
            return None

        return osgb_data, input_data, target_data

    def __iter__(self) -> Iterator[T_co]:
        for item in self.cached_items:
            yield item
