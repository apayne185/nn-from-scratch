import numpy as np
import gzip 
import urllib.request
import os



#urls,file names for MNIST dataset
base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"             # base_url = "http://yann.lecun.com/exdb/mnist/"    
files = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",  
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz"   
}



def download_mnist(destination="mnist_data"):
    os.makedirs(destination, exist_ok=True)         #if doesnt already exist

    for key, filename in files.items():
        path = os.path.join(destination, filename)
        if not os.path.exists(path):
            print(f"Downloading {filename}")
            urllib.request.urlretrieve(base_url + filename, path)   


#parse IDX files 
def load_images(path):
    with gzip.open(path, 'rb') as f:
        _ = int.from_bytes(f.read(4), 'big')    #read/ignore magic number

        #reads number of imgs, rows, cols
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4),'big')
        cols = int.from_bytes(f.read(4),'big')
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)     #reshapes data

        return data   
    
def load_labels(path):
    with gzip.open(path, 'rb') as f:   
        _ = int.from_bytes(f.read(4),'big')    
        num_labels = int.from_bytes(f.read(4),'big')  
        labels= np.frombuffer(f.read(), dtype=np.uint8)   

        return labels  
    



#final loading function for all MNIST data
def load_mnist_data():
    download_mnist()      #ensures downloaded
    path = "mnist_data"   

    x_train = load_images(os.path.join(path, files["train_images"]))
    y_train = load_labels(os.path.join(path, files["train_labels"]))
    x_test = load_images(os.path.join(path, files["test_images"]))
    y_test = load_labels(os.path.join(path, files["test_labels"]))  

    return (x_train, y_train), (x_test, y_test)



#loads the data
(x_train, y_train), (x_test, y_test)= load_mnist_data()

print("Training data shape: ",x_train.shape)
print("Training labels shape: ", y_train.shape)   

