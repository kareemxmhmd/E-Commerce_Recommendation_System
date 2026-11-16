import keras
from keras import layers
from keras import ops
from typing import Any
import numpy as np
import polars as pl


def build_integer_lookup(
          values: np.ndarray | list[int] | pl.Series,
          mask_token: int | None = None,
          num_oov_indices: int = 1,
) -> layers.IntegerLookup:

     if isinstance(values, pl.Series):
          values = values.drop_nulls().to_numpy()
     clean_vals = np.asarray(values, dtype=np.int64)
     unique_vals = np.unique(clean_vals)
     lookup = layers.IntegerLookup(
          vocabulary=unique_vals,
          mask_token=mask_token,
          num_oov_indices=num_oov_indices,
          output_mode="int",
     )
     return lookup