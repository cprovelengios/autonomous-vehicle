#!/usr/bin/python3.7
import os
import sys
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


# Import data info
def import_data_info(*, path, start_folder, end_folder):
    columns = ['Image', 'Steering']
    data = pd.DataFrame()

    for x in range(start_folder, end_folder + 1):
        data_new = pd.read_csv(os.path.join(path, f'log_{x}.csv'), names=columns)
        data_new['Image'] = data_new['Image'].str.split('/', 5).str[-1]
        print(f'Folder {x}: {data_new.shape[0]} Images', end=', ')
        data = data.append(data_new, True)  # True need to continue row index from last append

    print(f'\nTotal: {end_folder - start_folder + 1} Folder(s), {data.shape[0]} Images Imported\n')

    return data


# Visualize and balance data
def visualize_balance_data(data, display=True, balance=False):
    number_of_bins = 201    # It defines the number of equal-width bins in the given range. Steering take values from -1 to 1, so (2 / 0.01) + 1 = 201, + 1 for 0 value
    samples_per_bin = 300   # It defines how many samples it will keep after balance for every bin
    remove_index_list = []

    hist, bins = np.histogram(data['Steering'], number_of_bins)
    center = np.round(np.arange(-1, 1.01, 0.01), 2)
    bins = np.round(bins, 2)

    if display:
        plt.bar(center, hist, width=0.03)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samples_per_bin, samples_per_bin))
        plt.title('Data Visualisation')
        plt.xlabel('Steering Angle')
        plt.ylabel('No of Samples')
        plt.show()

    if balance:
        flag = False    # This flag and following check need because bins have 2 times value 0

        for j in range(number_of_bins + 1):
            if bins[j] == 0 and not flag:
                flag = True
                continue

            bin_data_list = []

            for i in range(len(data['Steering'])):
                if data['Steering'][i] == bins[j]:
                    bin_data_list.append(i)

            bin_data_list = shuffle(bin_data_list)
            bin_data_list = bin_data_list[samples_per_bin:]
            remove_index_list.extend(bin_data_list)

        data.drop(data.index[remove_index_list], inplace=True)
        print(f'Removed Images: {len(remove_index_list)}\nRemaining Images: {len(data)}')

        if display:
            hist, bins = np.histogram(data['Steering'], number_of_bins)
            plt.bar(center, hist, width=0.03)
            plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samples_per_bin, samples_per_bin))
            plt.title('Balanced Data')
            plt.xlabel('Steering Angle')
            plt.ylabel('No of Samples')
            plt.show()

    return data


# Convert data frame to list
def load_data(data):
    images_path = []
    steering = []

    for i in range(len(data)):
        indexed_data = data.iloc[i]
        images_path.append(indexed_data[0])
        steering.append(float(indexed_data[1]))

    images_path = np.asarray(images_path)
    steering = np.asarray(steering)

    return images_path, steering


# Preprocess image for neural network
def pre_process(img):
    img = img[54:120, :, :]                     # Crop image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Convert image to YUV colorspace
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255                             # Normalization

    return img


# Check images and append or delete
def check_images(path, data):
    images_path, steerings = load_data(data)
    index = 0
    previous_index = -1
    length = len(images_path)
    option = 'no'
    append_index_list = []
    remove_index_list = []

    while True:
        if previous_index != index:
            img = cv2.imread(images_path[index])
            img = pre_process(img)
            img = cv2.resize(img, (640, 360))

            print(f'Image: {index:>4}  -  Steering: {steerings[index]:>5}')
            cv2.imshow('Test Image', img)
            previous_index = index

        key = cv2.waitKey(0)

        if key == 83:                       # ->
            index = (index + 1) % length
        elif key == 81:                     # <-
            index = (index - 1) % length
        elif key == 82:                     # ^
            append_index_list.append(index)
            print(f'Append Image: {index}')
        elif key == 84:                     # v
            remove_index_list.append(index)
            print(f'Remove Image: {index}')
        elif key == 27:
            cv2.destroyAllWindows()
            break

    # Set removes duplicates
    remove_index_list = list(set(remove_index_list))

    if len(append_index_list) or len(remove_index_list):
        print(f'\nYou have select {len(append_index_list)} images to append and {len(remove_index_list)} images to remove')
        option = input('Do you want to proceed(yes or no): ')

        if option != 'yes':
            option = input('Images will be lost, do you want to proceed(yes or no): ')

    if option == 'yes':
        count_folder = 0
        img_list = []
        steering_list = []
        img_list_new = []
        steering_list_new = []

        while os.path.exists(os.path.join(path, f'IMG{str(count_folder)}')):
            count_folder += 1

        new_path = path + '/IMG' + str(count_folder)
        os.makedirs(new_path)

        data.drop(data.index[remove_index_list], inplace=True)

        for i in append_index_list:
            img_list.append(data['Image'][i])
            steering_list.append(data['Steering'][i])

        raw_data = {'Image': img_list, 'Steering': steering_list}
        data_new = pd.DataFrame(raw_data)
        data = data.append(data_new, True)  # True need to continue row index from last append

        for i in range(len(data)):
            img = cv2.imread(data.iloc[i][0])
            timestamp = str(datetime.timestamp(datetime.now())).replace('.', '')
            file_name = os.path.join(os.getcwd(), new_path, f'Image_{timestamp}.jpg')
            cv2.imwrite(file_name, img)

            img_list_new.append(file_name)
            steering_list_new.append(data.iloc[i][1])

        raw_data = {'Image': img_list_new, 'Steering': steering_list_new}
        data_new = pd.DataFrame(raw_data)
        data_new.to_csv(os.path.join(path, f'log_{str(count_folder)}.csv'), index=False, header=False)
        print(f'\nNew Log Saved, total images: {len(img_list_new)}')


def main():
    path = 'Training_Data'
    data = import_data_info(path=path, start_folder=folders[0], end_folder=folders[1])

    # Visualize and balance data
    if visualize_data:
        visualize_balance_data(data, display=True, balance=True)

    # Check images and append or delete
    if check_data:
        check_images(path, data)


if __name__ == '__main__':
    try:
        folders = list(map(int, sys.argv[1].split('-')))
        visualize_data = True if int(sys.argv[2]) == 1 else False
        check_data = True if int(sys.argv[3]) == 1 else False
    except (IndexError, ValueError):
        print(f'Give required arguments: Start folder-End folder(0-0), Visualize data(0 or 1) and Check data(0 or 1)')
        sys.exit()

    main()
