import pandas as pd
import os
from PIL import Image
import utils as utils
from sklearn.model_selection import train_test_split
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def train_model_with_incremental_data(data_directory, resize_width, resize_height, test_size, random_state, batch_size, epochs):
    # Create an empty DataFrame to store the image data and labels
    df = pd.DataFrame(columns=["Image", "Label"])

    # Loop through each image file in the data directory
    file_list = os.listdir(data_directory)
    for filename in file_list:
        if filename.endswith(".jpeg"):
            # Load the image using PIL
            image_path = os.path.join(data_directory, filename)
            image = Image.open(image_path)

            # Resize the image
            image = image.resize((resize_width, resize_height))

            # Add the class name as a prefix to the label
            if filename.startswith("Banana"):
                label = "Banana"
            elif filename.startswith("Apple"):
                label = "Apple"
            elif filename.startswith("Grape"):
                label = "Grape"

            # Add the image and label to the DataFrame
            df.loc[len(df)] = [image, label]

    # Check the Batched Data
    df_existing = utils.load_json()
    
    if len(df) > (len(df_existing)+100):
        
        # Split the data into training and testing sets
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        train_df, val_df = train_test_split(train_df, test_size=test_size, random_state=random_state)

        # Convert images to numpy arrays
        train_images = np.array([img_to_array(img) for img in train_df['Image']])
        val_images = np.array([img_to_array(img) for img in val_df['Image']])
        test_images = np.array([img_to_array(img) for img in test_df['Image']])

        # Normalize the image pixel values between 0 and 1
        train_images = train_images.astype('float32') / 255.0
        val_images = val_images.astype('float32') / 255.0
        test_images = test_images.astype('float32') / 255.0

        # Convert labels to categorical
        num_classes = df['Label'].nunique()
        train_labels = pd.get_dummies(train_df['Label']).values
        val_labels = pd.get_dummies(val_df['Label']).values
        test_labels = pd.get_dummies(test_df['Label']).values

        # Create the model
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=train_images.shape[1:]))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(val_images, val_labels))

        # Evaluate the model on the testing set
        test_loss, test_accuracy = model.evaluate(test_images, test_labels)
        print("Test Loss:", test_loss)
        print("Test Accuracy:", test_accuracy)

        # Make predictions on the testing set
        predictions = model.predict(test_images)

        # Convert the predictions to labels
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(test_labels, axis=1)

        # Compare the predicted labels with true labels
        correct_predictions = np.sum(predicted_labels == true_labels)
        total_predictions = len(test_labels)
        accuracy = correct_predictions / total_predictions
        print("Accuracy:", accuracy)
        
        # Update the Model 
        model.save(CONFIG_DATA['model_path'])
        
        return model
    else:
        print("Number of new samples is below the threshold. Skipping model update.")
    

if __name__ == '__main__':
    # 1. Load configuration file
    CONFIG_DATA = utils.config_load()
    
    # 2. Running Code
    train_model_with_incremental_data(data_directory=CONFIG_DATA['data_source'], 
                                       resize_width=CONFIG_DATA['resize_width'], 
                                       resize_height=CONFIG_DATA['resize_height'],
                                       test_size=CONFIG_DATA['test_size'], 
                                       random_state=CONFIG_DATA['random_state'], 
                                       batch_size=CONFIG_DATA['batch_size'], 
                                       epochs=CONFIG_DATA['epochs']
                                       )
    