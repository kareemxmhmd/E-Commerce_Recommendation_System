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
    def vocab_sizes(self) -> dict[str, int]:
        
        sizes = {"user_id": int(self.user_lookup.vocabulary_size())}
        if self.city_lookup is not None:
           sizes["city"] = int(self.city_lookup.vocabulary_size())
        if self.device_lookup is not None:
            sizes["device"] = int(self.device_lookup.vocabulary_size())
        if self.language_lookup is not None:
            sizes["language"] = int(self.language_lookup.vocabulary_size())
        return sizes

    def call(
            self, inputs: dict[str, keras.KerasTensor]
    ) -> dict[str, keras.KerasTensor]:
        outputs: dict[str, keras.KerasTensor] = {}
        if "user_id" in inputs:
            outputs["user_idx"] = self.user_lookup(inputs["user_id"])
        if "city" in inputs and self.city_lookup is not None:
            outputs["city_idx"] = self.city_lookup(inputs["city"])
        if "device" in inputs and self.device_lookup is not None:
            outputs["device_idx"] = self.device_lookup(inputs["device"])
        if "language" in inputs and self.language_lookup is not None:
            outputs["language_idx"] = self.language_lookup(inputs["language"])
        return outputs

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "user_vocab": self._user_vocab,
                "city_vocab": self._city_vocab,
                "device_vocab": self._device_vocab,
                "language_vocab": self._language_vocab,
            }
        )
        return config



@keras.saving.register_keras_serializable(package="recommender")
class ItemPreprocessor(keras.layers.Layer):
    
    def __init__(
          self,
          item_df: pl.DataFrame | None = None,
          item_vocab: list[int] | None = None,
          category_vocab: list[int] | None = None,
          root_category_vocab: list[int] | None = None,
          seller_vocab: list[int] | None = None,
          brand_vocab: list[str] | None = None,
          price_stats: tuple[float, float] | None = None,
          ctr_stats: tuple[float, float] | None = None,
          rating_stats: tuple[float, float] | None = None,
          **kwargs: Any,
    ) -> None:
       super().__init__(**kwargs)

       if item_df is not None:
             item_vocab = (item_df["item_id"].drop_nulls().unique().sort().to_list())

             category_vocab = (
                 item_df["category_id"].drop_nulls().unique().sort().to_list()
                 if "category_id" in item_df.columns
                 else None
             )
             root_category_vocab = (
                 item_df["root_category_id"].drop_nulls().unique().sort().to_list()
                 if "root_category_id" in item_df.columns
                 else None
             )
             seller_vocab = (
                 item_df["seller_id"].drop_nulls().unique().sort().to_list()
                 if "seller_id" in item_df.columns
                 else None
             )
             brand_vocab = (
                 item_df["brand"].drop_nulls().cast(pl.Utf8).unique().sort().to_list()
                 if "brand" in item_df.columns
                 else None
             )
       
             if "log_price" in item_df.columns:
               prices = item_df["log_price"].fill_null(0.0).to_numpy()
               price_stats = (
                   float(np.mean(prices)),
                   max(float(np.var(prices)), 1e-8),
               )
             elif "price" in item_df.columns:
               prices = item_df["price"].fill_null(0.0).to_numpy()
               price_stats = (
                   float(np.mean(prices)),
                   max(float(np.var(prices)), 1e-8),
               )
       
             if "train_smoothed_ctr" in item_df.columns:
               ctrs = item_df["train_smoothed_ctr"].fill_null(0.0).to_numpy()
               ctr_stats = (float(np.mean(ctrs)), max(float(np.var(ctrs)), 1e-8))
             elif "train_ctr" in item_df.columns:
               ctrs = item_df["train_ctr"].fill_null(0.0).to_numpy()
               ctr_stats = (float(np.mean(ctrs)), max(float(np.var(ctrs)), 1e-8))
       
             if "train_avg_rating" in item_df.columns:
               ratings = item_df["train_avg_rating"].fill_null(0.0).to_numpy()
               rating_stats = (
                   float(np.mean(ratings)),
                   max(float(np.var(ratings)), 1e-8),
               )
       
       self._item_vocab = [int(x) for x in item_vocab] if item_vocab else None
       self._category_vocab = ([int(x) for x in category_vocab] if category_vocab else None)
       self._root_category_vocab = ([int(x) for x in root_category_vocab] if root_category_vocab else None)
       self._seller_vocab = ([int(x) for x in seller_vocab] if seller_vocab else None)
       self._brand_vocab = [str(x) for x in brand_vocab] if brand_vocab else None
       self._price_stats = price_stats
       self._ctr_stats = ctr_stats
       self._rating_stats = rating_stats
   
       self.item_lookup = (
           build_integer_lookup(self._item_vocab)
           if self._item_vocab
           else layers.IntegerLookup(output_mode="int")
       )
       self.category_lookup = (
           build_integer_lookup(self._category_vocab)
           if self._category_vocab
           else None
       )
       self.root_category_lookup = (
           build_integer_lookup(self._root_category_vocab)
           if self._root_category_vocab
           else None
       )
       self.seller_lookup = (
           build_integer_lookup(self._seller_vocab) if self._seller_vocab else None
       )
       self.brand_lookup = (
           build_string_lookup(self._brand_vocab) if self._brand_vocab else None
       )
   
       self.price_norm = (
           layers.Normalization(
               axis=-1,
               mean=np.array([self._price_stats[0]], dtype=np.float32),
               variance=np.array([self._price_stats[1]], dtype=np.float32),
           )
           if self._price_stats is not None
           else None
       )
       self.ctr_norm = (
           layers.Normalization(
               axis=-1,
               mean=np.array([self._ctr_stats[0]], dtype=np.float32),
               variance=np.array([self._ctr_stats[1]], dtype=np.float32),
           )
           if self._ctr_stats is not None
           else None
       )
       self.rating_norm = (
           layers.Normalization(
               axis=-1,
               mean=np.array([self._rating_stats[0]], dtype=np.float32),
               variance=np.array([self._rating_stats[1]], dtype=np.float32),
           )
           if self._rating_stats is not None
           else None
       )
       self.built = True

    @property
    def vocab_sizes(self) -> dict[str, int]:
        sizes = {"item_id": int(self.item_lookup.vocabulary_size())}
        if self.category_lookup is not None:
           sizes["category_id"] = int(self.category_lookup.vocabulary_size())
        if self.root_category_lookup is not None:
           sizes["root_category_id"] = int(self.root_category_lookup.vocabulary_size())
        if self.seller_lookup is not None:
          sizes["seller_id"] = int(self.seller_lookup.vocabulary_size())
        if self.brand_lookup is not None:
          sizes["brand"] = int(self.brand_lookup.vocabulary_size())
        return sizes
        
    def call(
          self, inputs: dict[str, keras.KerasTensor]
    ) -> dict[str, keras.KerasTensor]:
       outputs: dict[str, keras.KerasTensor] = {}
       if "item_id" in inputs:
        outputs["item_idx"] = self.item_lookup(inputs["item_id"])

       if "category_id" in inputs and self.category_lookup is not None:
          outputs["category_idx"] = self.category_lookup(inputs["category_id"])

       if "root_category_id" in inputs and self.root_category_lookup is not None:
          outputs["root_category_idx"] = self.root_category_lookup(inputs["root_category_id"])

       if "seller_id" in inputs and self.seller_lookup is not None:
          outputs["seller_idx"] = self.seller_lookup(inputs["seller_id"])

       if "brand" in inputs and self.brand_lookup is not None:
          outputs["brand_idx"] = self.brand_lookup(inputs["brand"])
   
       if "log_price" in inputs and self.price_norm is not None:
          outputs["log_price_norm"] = self.price_norm(_ensure_2d_float(inputs["log_price"]))

       elif "price" in inputs and self.price_norm is not None:
          outputs["price_norm"] = self.price_norm(_ensure_2d_float(inputs["price"]))
   
       if "train_smoothed_ctr" in inputs and self.ctr_norm is not None:
          outputs["ctr_norm"] = self.ctr_norm(_ensure_2d_float(inputs["train_smoothed_ctr"]))

       elif "train_ctr" in inputs and self.ctr_norm is not None:
          outputs["ctr_norm"] = self.ctr_norm(_ensure_2d_float(inputs["train_ctr"]))
   
       if "train_avg_rating" in inputs and self.rating_norm is not None:
          outputs["rating_norm"] = self.rating_norm(_ensure_2d_float(inputs["train_avg_rating"]))

       return outputs
   
    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
           {
              "item_vocab": self._item_vocab,
              "category_vocab": self._category_vocab,
              "root_category_vocab": self._root_category_vocab,
              "seller_vocab": self._seller_vocab,
              "brand_vocab": self._brand_vocab,
              "price_stats": self._price_stats,
              "ctr_stats": self._ctr_stats,
              "rating_stats": self._rating_stats,
           }
        )
        return config