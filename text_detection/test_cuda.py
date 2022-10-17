import torch
# import tensorflow as tf
if __name__ == "__main__":
    # check with tensorflow
    # print(tf.__version__)
    # print("gpu_device_name =>" + tf.test.gpu_device_name())
    # print("Num GPUs Available: ", len(
    #     tf.config.experimental.list_physical_devices('GPU')))

    # check with torch
    isAvailable = torch.cuda.is_available()
    print("cuda is available ?",  isAvailable)
    if isAvailable:
        count = torch.cuda.device_count()
        print("cuda number of device =", count)
        for index in range(0, count):
            print("cuda device_name(0)", torch.cuda.get_device_name(0))
        print('End of Program')
        pass
