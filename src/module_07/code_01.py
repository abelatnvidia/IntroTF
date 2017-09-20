import os,tensorflow as tf

# the full file path for an image
file_path = os.path.join('data','notMNIST','A','MDEtMDEtMDAudHRm.png')

# lets slurp up the data in this file as generic binary content
with tf.gfile.FastGFile(file_path, 'rb') as f: image_data = f.read()

# decode the binary data as a png image and return a tensor
image_tensor = tf.image.decode_png(image_data,channels=1)

# remember we just have ops, have to run to get result
with tf.Session() as sess:

    sfw = tf.summary.FileWriter(logdir=os.getcwd(),graph=sess.graph)

    imgT = sess.run(image_tensor)

    sfw.close()

print('image data type: {}'.format(type(image_data)))
print('image tensor type: {}'.format(type(image_tensor)))
print('evaluated image tensor type: {}'.format(type(imgT)))
print('evaluated image tensor shape:{}'.format(imgT.shape))

'''
    Tensorflow has all the utilities built-in to work with images 
    such as jpg/jpeg/png etc.  The tf.gfile provides file I/O 
    without thread locking based on C++ FileSystem API.
    
    The C++ FileSystem API supports multiple file system 
    implementations, including local files, Google Cloud Storage 
    (using a gs:// prefix), and HDFS (using an hdfs:// prefix). 
    TensorFlow exports these as tf.gfile so that you can uses these 
    implementations for saving and loading checkpoints, writing 
    TensorBoard logs, and accessing training data (among other uses). 
    However, if all of your files are local, you can use the regular 
    Python file API without any issues.
'''