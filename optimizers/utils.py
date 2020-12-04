import time
import os
import matplotlib.pyplot as plt
import tensorflow as tf


def get_data(url):
    dataset_url = url
    dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(dataset_url), origin=dataset_url)
    print("Local copy of the dataset file: {}".format(dataset_fp))
    return dataset_fp
    
    
def preprocess_dataset_csv(batch_size, data, names, label):
    return tf.data.experimental.make_csv_dataset(
        data,
        batch_size,
        column_names=names,
        label_name=label,
        num_epochs=1
    )
    

def pack_features_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)
    return features, labels
    
    
def prepare_csv(data_patch, batch_size, names, label):
    _ = preprocess_dataset_csv(batch_size, data_patch, names, label)
    return _.map(pack_features_vector)
    
    
def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(3)
    ])

def pack_features_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)
    return features, labels


class lets_play():
    def __init__(self, data_path, names, label, batch_size=10, epochs=201, optimizer='sgd', **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
#         self.features, self.labels = None, None   
        self.num_epochs = epochs
        self.names = names
        self.label = label
        self.model = build_model()
        self.train_loss_results, self.train_accuracy_results = [], []
        self.train_dataset = prepare_csv(data_path, batch_size, self.names, self.label)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#         self.features, self.labels = next(iter(self.train_dataset))
        if optimizer == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=0.01,
                name='SGD'
            )
        elif optimizer == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.001, 
                beta_1=0.9, 
                beta_2=0.999, 
                epsilon=1e-07, 
                amsgrad=False,
                name='Adam'
            )
        elif optimizer == 'rmsprop':
            self.optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=0.001, 
                rho=0.9, 
                momentum=0.0, 
                epsilon=1e-07, 
                centered=False,
                name='RMSprop'
            )
        else:
            print('optimizer unavailable')

    def plot_data_points(self):
        features, labels = next(iter(self.train_dataset))
        plt.scatter(features[:, 2],
            features[:, 3],
            c=labels,
            cmap='viridis')

        plt.xlabel("Petal length")
        plt.ylabel("Sepal length")
        plt.show()
       
        
    def loss(self, model, x, y, training):
        y_ = model(x, training=training)
        return self.loss_object(y_true=y, y_pred=y_)
    
    def gradient(self, model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(model, inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)
     
    def train(self):
        start_time = time.time()
        for epoch in range(self.num_epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
            for x, y in self.train_dataset:
                loss_value, grads = self.gradient(self.model, x, y)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                # Track progress
                epoch_loss_avg.update_state(loss_value)  # Add current batch loss
                epoch_accuracy.update_state(y, self.model(x, training=True))

              # End epoch
            self.train_loss_results.append(epoch_loss_avg.result())
            self.train_accuracy_results.append(epoch_accuracy.result())


            if epoch % 50 == 0:
                print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                            epoch_loss_avg.result(),
                                                                            epoch_accuracy.result()))
        print('execution_time: ', time.time() - start_time)
    
    def plot_metrics(self):
        fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
        fig.suptitle('Training Metrics')

        axes[0].set_ylabel("Loss", fontsize=14)
        axes[0].plot(self.train_loss_results)

        axes[1].set_ylabel("Accuracy", fontsize=14)
        axes[1].set_xlabel("Epoch", fontsize=14)
        axes[1].plot(self.train_accuracy_results)
        plt.show()
        
    def predict(self, test_data_path, batch_size=30):
        test_accuracy = tf.keras.metrics.Accuracy()
        test_dataset = prepare_csv(test_data_path, batch_size, self.names, self.label)
        for (x, y) in test_dataset:
            logits = self.model(x, training=False)
            prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
            test_accuracy(prediction, y)
            print("labels: ", y.numpy(), "| prediction: ", prediction.numpy())
        print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
    
