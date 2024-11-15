import gin
import tensorflow as tf
import logging
import keras


@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, ds_info, run_paths, total_steps, log_interval, ckpt_interval):
        # Summary Writer
        self.summary_writer = tf.summary.create_file_writer(run_paths["path_logs_train"])

        # Checkpoint Manager
        self.checkpoint = tf.train.Checkpoint(optimizer=keras.optimizers.Adam(), model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, run_paths["path_ckpts_train"],
                                                             max_to_keep=3)

        # Loss objective
        self.loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = keras.optimizers.Adam()

        # Metrics
        self.train_loss = keras.metrics.Mean(name='train_loss')
        self.train_accuracy = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.val_loss = keras.metrics.Mean(name='val_loss')
        self.val_accuracy = keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def val_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.val_loss(t_loss)
        self.val_accuracy(labels, predictions)

    def train(self):
        for idx, (images, labels) in enumerate(self.ds_train):

            step = idx + 1
            self.train_step(images, labels)

            if step % self.log_interval == 0:

                # Reset test metrics
                self.val_loss.reset_states()
                self.val_accuracy.reset_states()

                for val_images, val_labels in self.ds_val:
                    self.val_step(val_images, val_labels)

                template = 'Step {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'
                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.val_loss.result(),
                                             self.val_accuracy.result() * 100))

                # Write summary to tensorboard
                with self.summary_writer.as_default():
                    tf.summary.scalar('train_loss', self.train_loss.result(), step = step)
                    tf.summary.scalar('train_accuracy', self.train_accuracy.result(), step = step)
                    tf.summary.scalar('val_loss', self.val_loss.result(), step = step)
                    tf.summary.scalar('val_accuracy', self.val_accuracy.result(), step = step)

                # Reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

                yield self.val_accuracy.result().numpy()

            if step % self.ckpt_interval == 0:
                logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
                # Save checkpoint
                self.checkpoint_manager.save()

            if step % self.total_steps == 0:
                logging.info(f'Finished training after {step} steps.')
                # Save final checkpoint
                self.checkpoint_manager.save()
                return self.val_accuracy.result().numpy()