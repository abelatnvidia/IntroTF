from tensorflow.python.client import device_lib

# get list of locally avaliable compute devices
devices = device_lib.list_local_devices()

# blab about them
for dev in devices: print(dev)
