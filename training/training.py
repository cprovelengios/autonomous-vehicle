#!/usr/bin/python3.7
from training_utils import *
from datetime import datetime
from tensorflow.keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split


def main():
    # Import data info, select which folders to import
    path = '../data/training_data'
    data = import_data_info(path=path, start_folder=folders[0], end_folder=folders[1])
    # print(f'{data.head()}\n\n{data.tail()}')

    # Visualize and balance data
    data = visualize_balance_data(data, display=True, balance=balance_data)

    # Convert data frame to list
    images_path, steerings = load_data(data)

    # Split data for training and validation(x for training and y for validation)
    x_train, y_train, x_test, y_test = train_test_split(images_path, steerings, test_size=0.2, random_state=10)
    print(f'\nTotal Training Images: {len(x_train)}\nTotal Validation Images: {len(y_train)}\n')

    # Create model, by Nvidia: https://developer.nvidia.com/blog/deep-learning-self-driving-cars/
    model = create_model()

    timestamp = datetime.now().strftime("%d_%m_%Y-%H:%M:%S")
    name = timestamp if comment == '' else f'{comment}_{timestamp}'

    csv_logger = CSVLogger(f'../data/models/log_{name}.csv', append=True, separator=';')

    # Training, for better results increase data, balance better, change augmentation method
    history = model.fit(data_gen(x_train, x_test, 100, 1),
                        steps_per_epoch=100,    # How many times in every epoch will take batch_size random images to train (100)
                        epochs=10,              # Set steps_per_epoch value so in 10, 15 epochs have good results           (10)
                        validation_data=data_gen(y_train, y_test, 50, 0),
                        validation_steps=50,
                        callbacks=[csv_logger])

    if save_model:
        model.save(f'../data/models/model_{name}.h5')
        print('Model Saved')
    else:
        os.remove(f'../data/models/log_{name}.csv')

    # Plot results of training
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.show()


if __name__ == '__main__':
    try:
        folders = list(map(int, sys.argv[1].split('-')))
        balance_data = int(sys.argv[2])
        save_model = True if int(sys.argv[3]) == 1 else False

        try:
            comment = sys.argv[4]
        except IndexError:
            comment = ''
    except (IndexError, ValueError):
        print(f'Give required arguments: Start folder-End folder (0-0), Balance (0 or Number), Save model (0 or 1) and Comment if want for name model')
        sys.exit()

    main()
