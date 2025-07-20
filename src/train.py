from typing import Optional
from tensorflow.keras.optimizers import Adam


def train_model(model, X_train, y_train, X_val, y_val,
                learning_rate: float = 0.001, epochs: int = 10,
                batch_size: int = 32) -> Optional[object]:
    """Compile and train a model."""
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size)
    return history
