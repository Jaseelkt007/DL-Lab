import tensorflow as tf
images_ds = tf.data.Dataset.list_files(r'F:\学校\课程文件\dl lab\idrid\IDRID_dataset\images\train\*', shuffle = False)
images_ds = images_ds.shuffle(buffer_size = 200)
for file in images_ds.take(5):
  print(file.numpy().decode('utf-8'))
image_count = len(images_ds)
print(image_count)