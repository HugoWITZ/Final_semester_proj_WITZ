import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv1D, Flatten, concatenate, GlobalAveragePooling1D, MaxPooling1D, LSTM, Input, concatenate, GRU
from keras.optimizers import Adam


from keras.regularizers import l1_l2

from sklearn.metrics import mean_absolute_error, mean_squared_error



def plot_metrics(history):
    """
    Plots the Mean Squared Error loss and Mean Absolute Percentage Error of a model's history.

    Args:
        history (History): The history object of a trained model.

    """
    # Plot the MSE loss for training and validation sets
    plt.figure(figsize=(12, 4))  # Set the figure size
    plt.subplot(1, 2, 1)  # Set the position of the first plot in a grid of 1 row and 2 columns
    plt.plot(history.history['loss'], label='Training MSE loss')
    plt.plot(history.history['val_loss'], label='Validation MSE loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()  # Add a legend
    plt.title('Losses (Mean Squared Error)')  # Set the title of the plot

    # Plot the Mean Absolute Percentage Error for training and validation sets
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mean_absolute_percentage_error'], label='Training MAPE')
    plt.plot(history.history['val_mean_absolute_percentage_error'], label='Validation MAPE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Percentage Error')
    plt.legend()  # Add a legend
    plt.title('Mean Absolute Percentage Error')  # Set the title of the plot

    plt.show()  # Display the figure


def plot_test_performance(model, X_signal_test, X_other_test, y_test):
    """
    Plots the true values vs predicted values for the test set.

    Args:
        model (Model): The trained model.
        X_signal_test (numpy array): The signal test set.
        X_other_test (numpy array): The other test set.
        y_test (numpy array): The labels for the test set.

    """
    # Predict the labels for the test set
    y_pred = model.predict([X_signal_test, X_other_test])

    # Create a scatter plot for each concentration
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Create a new figure and a 1x2 subplot

    # For each concentration
    for i in range(2):
        axes[i].scatter(y_test[:, i], y_pred[:, i], alpha=0.2)  # Scatter plot of true vs predicted values
        axes[i].plot([y_test[:, i].min(), y_test[:, i].max()], [y_test[:, i].min(), y_test[:, i].max()], 'k--', lw=2)  # Plot a line representing perfect predictions
        axes[i].set_xlabel(f"True Concentration {i + 1}")  # Set the x-axis label
        axes[i].set_ylabel(f"Predicted Concentration {i + 1}")  # Set the y-axis label
        axes[i].set_title(f"Concentration {i + 1} Performance")  # Set the title of the subplot

    plt.show()  # Display the figure

def evaluate_test_performance(model, X_signal_test, X_other_test, y_test):
    """
    Evaluates the performance of the model on the test set, returning the Mean Absolute Error and 
    Mean Squared Error for each concentration.

    Args:
        model (Model): The trained model.
        X_signal_test (numpy array): The signal test data.
        X_other_test (numpy array): The other features' test data.
        y_test (numpy array): The test set labels.

    Returns:
        mae (list): A list containing the Mean Absolute Error (MAE) for each concentration.
        mse (list): A list containing the Mean Squared Error (MSE) for each concentration.
    """
    # Make predictions on the test set using the trained model
    y_pred = model.predict([X_signal_test, X_other_test])

    # Calculate the Mean Absolute Error (MAE) for each concentration
    mae = [mean_absolute_error(y_test[:, i], y_pred[:, i]) for i in range(2)]
    
    # Calculate the Mean Squared Error (MSE) for each concentration
    mse = [mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(2)]

    return mae, mse  # Return the MAE and MSE




def first_cnn_model(lr=1e-3, dropout=0.5):
    # Signal input branch (CNN)
    input_signal = Input(shape=(7200, 1))
    x = Conv1D(16, kernel_size=3, activation='relu')(input_signal)
    x = BatchNormalization()(x)
    x = Conv1D(32, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    signal_branch = Dropout(dropout)(x)

    # Other features input branch (Dense)
    input_other = Input(shape=(14,))
    y = Dense(32, activation='relu')(input_other)
    y = BatchNormalization()(y)
    other_branch = Dropout(dropout)(y)

    # Combine branches
    combined = concatenate([signal_branch, other_branch])

    # Add dense layers
    z = Dense(64, activation='relu')(combined)
    z = BatchNormalization()(z)
    z = Dropout(dropout)(z)
    z = Dense(32, activation='relu')(z)
    z = BatchNormalization()(z)
    z = Dropout(dropout)(z)
    output = Dense(2, activation='relu')(z)

    optimizer = Adam(learning_rate=lr)

    model = Model(inputs=[input_signal, input_other], outputs=output)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])

    return model


def reg_cnn_model(n_features, initial_lr=1e-3, decay_rate=0.95, decay_steps=100, dropout=0.5, l1_reg=0.01, l2_reg=0.01):

    # Signal input branch (CNN)
    input_signal = Input(shape=(7200, 1))
    x = Conv1D(16, kernel_size=3, activation='relu')(input_signal)
    x = BatchNormalization()(x)
    x = Conv1D(32, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    signal_branch = Dropout(dropout)(x)

    # Other features input branch (Dense)
    input_other = Input(shape=(n_features,))
    y = Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(input_other)
    y = BatchNormalization()(y)
    other_branch = Dropout(dropout)(y)

    # Combine branches
    combined = concatenate([signal_branch, other_branch])

    # Add dense layers
    z = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(combined)
    z = BatchNormalization()(z)
    z = Dropout(dropout)(z)
    output = Dense(2, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(z)

    # Create an ExponentialDecay learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate
    )

    # Use the learning rate schedule with the Adam optimizer
    optimizer = Adam(learning_rate=lr_schedule)

    model = Model(inputs=[input_signal, input_other], outputs=output)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])

    return model