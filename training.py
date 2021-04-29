#!/usr/bin/python3.7
from data_utils import *
from training_utils import *
from datetime import datetime
from tensorflow.keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split


def main():
    # Import data info, select which folders to import
    path = 'Training_Data'
    data = import_data_info(path=path, start_folder=0, end_folder=1)
    # print(f'{data.head()}\n\n{data.tail()}')
    # pd.set_option('display.max_rows', None)
    # print(data)

    # Visualize and balance data
    data = visualize_balance_data(data, display=False, balance=True)

    # Convert data frame to list
    images_path, steerings = load_data(path, data)
    # cv2.imshow('Test Image', cv2.imread(images_path[0]))
    # cv2.waitKey(0)

    # Split data for training and validation(x for training and y for validation)
    x_train, y_train, x_test, y_test = train_test_split(images_path, steerings, test_size=0.2, random_state=42)
    print(f'\nTotal Training Images: {len(x_train)}\nTotal Validation Images: {len(y_train)}\n')

    # Create model, by nvidia: https://developer.nvidia.com/blog/deep-learning-self-driving-cars/
    model = create_model()

    # Select if you want to save the model and if you want to add a comment to the name
    save_model = False
    comment = ''

    timestamp = datetime.now().strftime("%d_%m_%Y-%H:%M:%S")
    name = timestamp if comment == '' else f'{comment}_{timestamp}'

    csv_logger = CSVLogger(f'Models/log_{name}.csv', append=True, separator=';')

    # Training, for better results increase data, balance better, change augmentation method
    history = model.fit(data_gen(x_train, x_test, 100, 1),
                        steps_per_epoch=100,    # How many times in every epoch will take batch_size random images to train (100)
                        epochs=15,              # Set steps_per_epoch value so in 10, 15 epochs have good results           (15)
                        validation_data=data_gen(y_train, y_test, 50, 0),
                        validation_steps=50,
                        callbacks=[csv_logger])

    if save_model:
        model.save(f'Models/model_{name}.h5')
        print('Model Saved')
    else:
        os.remove(f'Models/log_{timestamp}.csv')

    # Plot results of training
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.show()


if __name__ == '__main__':
    main()


# Shuffle the training dataset because they are in order
# keys = np.array(range(x_train.shape[0]))
# np.random.shuffle(keys)
# x_train = x_train[keys]
# y_train = y_train[keys]

# BATCHE SIZE TEST #####################################################################################
# batch_size = 100
# steps_per_epoch = len(x_train) // batch_size
# steps_per_epoch += 1 if steps_per_epoch * batch_size < len(x_train) else steps_per_epoch
#
# validation_batch_size = 50
# validation_steps = len(y_train) // validation_batch_size
# validation_steps += 1 if validation_steps * validation_batch_size < len(y_train) else validation_steps
#
# history = model.fit(data_gen(x_train, x_test, batch_size, 1),
#                     steps_per_epoch=steps_per_epoch,
#                     epochs=20,
#                     validation_data=data_gen(y_train, y_test, validation_batch_size, 0),
#                     validation_steps=validation_steps)
