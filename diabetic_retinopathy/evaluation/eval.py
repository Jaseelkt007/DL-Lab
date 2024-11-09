import tensorflow as tf
import logging
import keras

def evaluate(model, checkpoint, ds_test, ds_info, run_paths):
    # Load checkpoint
    if checkpoint:
        checkpoint.restore(tf.train.latest_checkpoint(run_paths['path_ckpts_train']))
        logging.info('Checkpoint restored for evaluation')

    # Define evaluation metrics
    loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits = True)
    test_loss = keras.metrics.Mean(name = 'test_loss')
    test_accuracy = keras.metrics.SparseCategoricalAccuracy(name = 'test_accuracy')

    # Evaluate on the test dataset
    for images, labels in ds_test:
        predictions = model(images, training = False)
        test_loss = loss_object(labels, predictions)

        # Update metrics
        test_loss(test_loss)
        test_accuracy(labels, predictions)

    # Log and return results
    logging.info(f'Test Loss: {test_loss.result():.4f}, Test Accuracy: {test_accuracy.result() * 100:.2f}%')
    return {
        "test_loss": test_loss.result().numpy(),
        "test_accuracy": test_accuracy.result().numpy()
    }