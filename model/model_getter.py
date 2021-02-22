from model.tft5 import SnapthatT5
from model.lr_schedule import CosineDecayWithWarmUP
import tensorflow as tf

def get_model(strategy=None,
              metrics=None,
              model_fn=SnapthatT5.from_pretrained,
              P=None):

    with strategy.scope():
        if metrics is None:
            metrics = [tf.keras.metrics.SparseTopKCategoricalAccuracy(name='accuracy', k=1)]

        model = model_fn(P.model_name)

        learning_rate = CosineDecayWithWarmUP(initial_learning_rate=P.initial_learning_rate,
                                              decay_steps=P.epochs * P.steps_per_epoch - P.warm_iterations,
                                              alpha=P.minimum_learning_rate,
                                              warm_up_step=P.warm_iterations)

        optimizer = tf.keras.optimizers.Adam(learning_rate)
        model.compile(optimizer=optimizer, metrics=metrics)

    return model


def load_pretrained_model(model_fn =SnapthatT5.from_pretrained,
                          P = None,
                          weight_path = None
                          ):

    if weight_path is None:
        weight_path = P.weight_path

    model = model_fn(P.model_name)
    model.load_weights(weight_path)

    return model


