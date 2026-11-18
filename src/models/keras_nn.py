
import keras
from keras import ops
from keras import layers
from keras import regularizers
from typing import Any
import polars as pl

from src.config import KerasNNConfig
from src.data.preprocessing import ItemPreprocessor, UserPreprocessor


@keras.saving.register_keras_serializable(package="recommender")
class L2Normalize(layers.Layer):
   
   def __init__(self, epsilon: float = 1e-8, **kwargs: Any) -> None:
     super().__init__(**kwargs)
     self.epsilon = epsilon

   def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
     norm = ops.sqrt(
       ops.maximum(ops.sum(ops.square(inputs), axis=-1, keepdims=True), self.epsilon)
     )
     return inputs / norm
   
   def get_config(self) -> dict[str, Any]:
     config = super().get_config()
     config.update({"epsilon": self.epsilon})
     return config


def build_user_tower(
    user_preprocessor: UserPreprocessor,
    config: KerasNNConfig,
) -> keras.Model:
  
  user_inputs = {
      "user_id": keras.Input(shape=(), dtype="int64", name="user_id"),
      "city": keras.Input(shape=(), dtype="string", name="city"),
      "device": keras.Input(shape=(), dtype="string", name="device"),
      "language": keras.Input(shape=(), dtype="string", name="language"),
  }

  indices = user_preprocessor(user_inputs)
  vocab = user_preprocessor.vocab_sizes
  reg = regularizers.l2(config.l2_reg)

  user_emb = layers.Embedding(
      input_dim=vocab["user_id"] + 1,
      output_dim=config.embedding_dim,
      embeddings_regularizer=reg,
      name="user_id_embedding",
  )(indices["user_idx"])

  feature_list = [layers.Flatten()(user_emb)]

  if "city_idx" in indices:
    city_emb = layers.Embedding(
        input_dim=vocab["city"] + 1,
        output_dim=8,
        embeddings_regularizer=reg,
        name="city_embedding",
    )(indices["city_idx"])
    feature_list.append(layers.Flatten()(city_emb))

  if "device_idx" in indices:
    device_emb = layers.Embedding(
        input_dim=vocab["device"] + 1,
        output_dim=4,
        embeddings_regularizer=reg,
        name="device_embedding",
    )(indices["device_idx"])
    feature_list.append(layers.Flatten()(device_emb))

  if "language_idx" in indices:
    lang_emb = layers.Embedding(
        input_dim=vocab["language"] + 1,
        output_dim=4,
        embeddings_regularizer=reg,
        name="language_embedding",
    )(indices["language_idx"])
    feature_list.append(layers.Flatten()(lang_emb))

  x = (
      layers.Concatenate(name="user_concat")(feature_list)
      if len(feature_list) > 1
      else feature_list[0]
  )

  for i, units in enumerate(config.tower_hidden_units[:-1]):
    x = layers.Dense(
        units,
        activation="relu",
        kernel_regularizer=reg,
        name=f"user_dense_{i}",
    )(x)
    x = layers.BatchNormalization(name=f"user_bn_{i}")(x)
    x = layers.Dropout(config.dropout_rate, name=f"user_drop_{i}")(x)

  user_vec = layers.Dense(
      config.embedding_dim,
      activation=None,
      kernel_regularizer=reg,
      name="user_projection",
  )(x)
  user_vec_norm = L2Normalize(name="user_embedding")(user_vec)

  return keras.Model(
      inputs=user_inputs, outputs=user_vec_norm, name="user_tower"
  )


def build_item_tower(
    item_preprocessor: ItemPreprocessor,
    config: KerasNNConfig,
) -> keras.Model:

  item_inputs = {
      "item_id": keras.Input(shape=(), dtype="int64", name="item_id"),
      "category_id": keras.Input(shape=(), dtype="int64", name="category_id"),
      "root_category_id": keras.Input(
          shape=(), dtype="int64", name="root_category_id"
      ),
      "seller_id": keras.Input(shape=(), dtype="int64", name="seller_id"),
      "log_price": keras.Input(shape=(), dtype="float32", name="log_price"),
      "train_smoothed_ctr": keras.Input(
          shape=(), dtype="float32", name="train_smoothed_ctr"
      ),
      "train_avg_rating": keras.Input(
          shape=(), dtype="float32", name="train_avg_rating"
      ),
  }

  processed = item_preprocessor(item_inputs)
  vocab = item_preprocessor.vocab_sizes
  reg = regularizers.l2(config.l2_reg)

  item_emb = layers.Embedding(
      input_dim=vocab["item_id"] + 1,
      output_dim=config.embedding_dim,
      embeddings_regularizer=reg,
      name="item_id_embedding",
  )(processed["item_idx"])

  feature_list = [layers.Flatten()(item_emb)]

  if "category_idx" in processed:
    cat_emb = layers.Embedding(
        input_dim=vocab["category_id"] + 1,
        output_dim=16,
        embeddings_regularizer=reg,
        name="category_embedding",
    )(processed["category_idx"])
    feature_list.append(layers.Flatten()(cat_emb))

  if "root_category_idx" in processed:
    root_emb = layers.Embedding(
        input_dim=vocab["root_category_id"] + 1,
        output_dim=8,
        embeddings_regularizer=reg,
        name="root_category_embedding",
    )(processed["root_category_idx"])
    feature_list.append(layers.Flatten()(root_emb))

  if "seller_idx" in processed:
    seller_emb = layers.Embedding(
        input_dim=vocab["seller_id"] + 1,
        output_dim=16,
        embeddings_regularizer=reg,
        name="seller_embedding",
    )(processed["seller_idx"])
    feature_list.append(layers.Flatten()(seller_emb))

  for num_key in ("log_price_norm", "ctr_norm", "rating_norm"):
    if num_key in processed:
      feature_list.append(layers.Reshape((1,))(processed[num_key]))

  x = layers.Concatenate(name="item_concat")(feature_list)

  for i, units in enumerate(config.tower_hidden_units[:-1]):
    x = layers.Dense(
        units,
        activation="relu",
        kernel_regularizer=reg,
        name=f"item_dense_{i}",
    )(x)
    x = layers.BatchNormalization(name=f"item_bn_{i}")(x)
    x = layers.Dropout(config.dropout_rate, name=f"item_drop_{i}")(x)

  item_vec = layers.Dense(
      config.embedding_dim,
      activation=None,
      kernel_regularizer=reg,
      name="item_projection",
  )(x)
  item_vec_norm = L2Normalize(name="item_embedding")(item_vec)

  return keras.Model(
      inputs=item_inputs, outputs=item_vec_norm, name="item_tower"
  )


def build_keras_recommender_nn(
    users_df: pl.DataFrame,
    items_df: pl.DataFrame,
    config: KerasNNConfig | None = None,
) -> tuple[keras.Model, keras.Model, keras.Model]:
  
  cfg = config or KerasNNConfig()
  reg = regularizers.l2(cfg.l2_reg)

  user_preprocessor = UserPreprocessor(
      user_df=users_df, name="user_preprocessor"
  )
  item_preprocessor = ItemPreprocessor(
      item_df=items_df, name="item_preprocessor"
  )

  user_tower = build_user_tower(user_preprocessor, cfg)
  item_tower = build_item_tower(item_preprocessor, cfg)

  model_inputs = {
      "user_id": keras.Input(shape=(), dtype="int64", name="user_id"),
      "city": keras.Input(shape=(), dtype="string", name="city"),
      "device": keras.Input(shape=(), dtype="string", name="device"),
      "language": keras.Input(shape=(), dtype="string", name="language"),
      "item_id": keras.Input(shape=(), dtype="int64", name="item_id"),
      "category_id": keras.Input(shape=(), dtype="int64", name="category_id"),
      "root_category_id": keras.Input(
          shape=(), dtype="int64", name="root_category_id"
      ),
      "seller_id": keras.Input(shape=(), dtype="int64", name="seller_id"),
      "log_price": keras.Input(shape=(), dtype="float32", name="log_price"),
      "train_smoothed_ctr": keras.Input(
          shape=(), dtype="float32", name="train_smoothed_ctr"
      ),
      "train_avg_rating": keras.Input(
          shape=(), dtype="float32", name="train_avg_rating"
      ),
      "position": keras.Input(shape=(), dtype="int64", name="position"),
  }

  user_vec = user_tower(
      {k: model_inputs[k] for k in ("user_id", "city", "device", "language")}
  )
  item_vec = item_tower(
      {
          k: model_inputs[k]
          for k in (
              "item_id",
              "category_id",
              "root_category_id",
              "seller_id",
              "log_price",
              "train_smoothed_ctr",
              "train_avg_rating",
          )
      }
  )

  hadamard = layers.Multiply(name="user_item_hadamard")([user_vec, item_vec])
  dot_sim = layers.Dot(axes=-1, name="user_item_dot")([user_vec, item_vec])

  interaction_vec = layers.Concatenate(name="interaction_features")(
      [user_vec, item_vec, hadamard, dot_sim]
  )

  h = interaction_vec
  for i, units in enumerate(cfg.ranker_hidden_units):
    h = layers.Dense(
        units,
        activation="relu",
        kernel_regularizer=reg,
        name=f"ranker_mlp_{i}",
    )(h)
    h = layers.Dropout(cfg.dropout_rate, name=f"ranker_drop_{i}")(h)

  relevance_logit = layers.Dense(1, activation=None, name="relevance_logit")(h)

  pos_bias = layers.Embedding(
      input_dim=cfg.max_position + 2,
      output_dim=1,
      embeddings_initializer="zeros",
      name="position_bias_embedding",
  )(model_inputs["position"])
  pos_bias_flat = layers.Flatten(name="position_bias_logit")(pos_bias)

  total_logit = layers.Add(name="debiased_total_logit")(
      [relevance_logit, pos_bias_flat]
  )
  click_prob = layers.Activation("sigmoid", name="click_prob")(total_logit)

  full_model = keras.Model(
      inputs=model_inputs,
      outputs=click_prob,
      name="keras_recommender_nn",
  )
  full_model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=[
          keras.metrics.AUC(name="roc_auc"),
          keras.metrics.AUC(curve="PR", name="pr_auc"),
          keras.metrics.BinaryAccuracy(name="accuracy"),
      ],
  )
  return full_model, user_tower, item_tower
