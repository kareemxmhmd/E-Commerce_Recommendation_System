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


def build_string_lookup(
    values: np.ndarray | list[str] | pl.Series,
    mask_token: str | None = None,
    num_oov_indices: int = 1,
) -> layers.StringLookup:
    
    if isinstance(values, pl.Series):
       values = values.drop_nulls().to_list()

    unique_vals = sorted({str(v) for v in values if v is not None})
    lookup = layers.StringLookup(
        vocabulary=unique_vals,
        mask_token=mask_token,
        num_oov_indices=num_oov_indices,
        output_mode="int",
    )
    return lookup


def build_normalizer(
    values: np.ndarray | list[float] | pl.Series,
) -> layers.Normalization:
    if isinstance(values, pl.Series):
       values = values.fill_null(0.0).to_numpy()
    clean_vals = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    clean_vals = np.nan_to_num(clean_vals, nan=0.0, posinf=0.0, neginf=0.0)

    mean = float(np.mean(clean_vals))
    var = float(np.var(clean_vals))
    if var < 1e-8:
        var = 1.0

    return layers.Normalization(
        axis=-1,
        mean=np.array([mean], dtype=np.float32),
        variance=np.array([var], dtype=np.float32),
    )


def _ensure_2d_float(tensor: keras.KerasTensor) -> keras.KerasTensor:
    
    x = ops.cast(tensor, "float32")
    if len(x.shape) == 1:
        x = ops.expand_dims(x, axis=-1)
    return x


@keras.saving.register_keras_serializable(package="recommender")
class UserPreprocessor(keras.layers.Layer):
    
    def __init__(
            self,
            user_df: pl.DataFrame | None = None,
            user_vocab: list[int] | None = None,
            city_vocab: list[str] | None = None,
            device_vocab: list[str] | None = None,
            language_vocab: list[str] | None = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if user_df is not None:
            user_vocab = (
                user_df["user_id"].drop_nulls().unique().sort().to_list()   
            )
            city_vocab = (
                user_df["city"].drop_nulls().cast(pl.Utf8).unique().sort().to_list()
                if "city" in user_df.columns
                else None
            )
            device_vocab = (
                user_df["device"].drop_nulls().cast(pl.Utf8).unique().sort().to_list()
                if "device" in user_df.columns
                else None
            )
            language_vocab = (
                user_df["language"]
                .drop_nulls()
                .cast(pl.Utf8)
                .unique()
                .sort()
                .to_list()
                if "language" in user_df.columns
                else None
            )

        self._user_vocab = [int(x) for x in user_vocab] if user_vocab else None
        self._city_vocab = [str(x) for x in city_vocab] if city_vocab else None
        self._device_vocab = ([str(x) for x in device_vocab] if device_vocab else None)
        self._language_vocab = ([str(x) for x in language_vocab] if language_vocab else None)

        self.user_lookup = (
            build_integer_lookup(self._user_vocab)
            if self._user_vocab
            else layers.IntegerLookup(output_mode="int")
        )
        self.city_lookup = (build_string_lookup(self._city_vocab) if self._city_vocab else None)
        self.device_lookup = (build_string_lookup(self._device_vocab) if self._device_vocab else None)

        self.language_lookup = (
            build_string_lookup(self._language_vocab)
            if self._language_vocab
            else None
        )
        self.built = True

    @property
    def vocab_sizes():
        pass
    
