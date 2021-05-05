#!/usr/bin/python3.7
# Shuffle the training dataset because they are in order
# keys = np.array(range(x_train.shape[0]))
# np.random.shuffle(keys)
# x_train = x_train[keys]
# y_train = y_train[keys]

# Batche size
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

# Balance
# x = max(hist[0], hist[200])
# cut = (x // 50 + 1) * 50 if x % 50 else x // 50 * 50
#
# prin(cut)
