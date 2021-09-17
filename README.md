# CNN to detect leaf diseases in Maize plant

## Overview
A Convolutional Neural Net (CNN/ConvNet) is a Deep Neural Network, that takes inputs as images, assigns weights and biases to various aspects of the image and
then differentiates among different class of images. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms.
A ConvNet is able to successfully capture the Spatial and Temporal dependencies in an image through the application of relevant filters. The architecture performs a better fitting to the image dataset,
when compared to a standard neural network, due to the reduction in the number of parameters involved and reusability of weights.

## About this Project
This repository contains a self-made ConvNet (CNN.py) implemented using Keras, an open source neural network library in Python. The model is
computationally cheaper than other popular and successful implementations of CNNs such as the VGG16 and ResNet50, having only
77,633 trainable parameters compared to the 138,000,000 parameters of VGG16 and 23,000,000 parameters of ResNet50.

Despite it's low number of parameters, it manages a **training set accuracy ~ 96%** and **test set accuracy ~ 93%**.

It uses 7 Convolution and Pooling layers along with Spatial Dropout to prevent overfitting, followed by 3 Dense and Dropout layers to classify the maize plant images
as having disease (Cercospora, common rust, and northern leaf blight) or Healthy.

## About the Dataset
The PlantVillage dataset used is avaiable on [Kaggle](www.kaggle.com) and consists of 3852 images divided into two classes. For the analysis of maize leaf disease images, all the diseased leaf images were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system.

You can access the dataset [here](https://www.kaggle.com/abdallahalidev/plantvillage-dataset)

<table>
  <tr>
   <td align="center"><img src="https://storage.googleapis.com/kagglesdsdata/datasets/277323/658267/color/Corn_%28maize%29___healthy/06fc07d3-2c82-4538-9d0e-bff9808cd3fe___R.S_HL%200639%20copy.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20210917%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210917T200944Z&X-Goog-Expires=345599&X-Goog-SignedHeaders=host&X-Goog-Signature=29c1d2e64cf2671961513cdc9aa63a26f7edf7f1de3397421679dab9a1b0931434701a0b64e5bee9d347fee5304bfee8becae05bceefdcd16b0561e12f335448098c744bfa0d2bf440b802c274dfc0683db3772595cc0581786524fb3211b5c251e0258cf954952d5ddd5606b9d754bcf161705511eb5c7db8ea815477f96cdfa3330d10434b4d08ddfb3eb17a50fc126b753fd3fd8aef023f3f0d26dbd76644eb2a72862df1fbd03d921882f188246e605171ec84b4df51274e8269db7a0c6cb4483ffdf8a9436cab29cc361c57d97d4783e0166d72110cb9d4f735501c46776bb96dd5c5f5a977a8b0ee16c081dd805a64671efe363a5f5fbcfa030bd56309" width="100px;" alt="Healthy"/><br /><sub><b>Healthy</b></sub></a><br /></td>
   <td align="center"><img src="https://storage.googleapis.com/kagglesdsdata/datasets/277323/658267/color/Corn_%28maize%29___Cercospora_leaf_spot%20Gray_leaf_spot/00a20f6f-e8bd-4453-9e25-36ea70feb626___RS_GLSp%204655.JPG?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20210917%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210917T200750Z&X-Goog-Expires=345599&X-Goog-SignedHeaders=host&X-Goog-Signature=1d253177c76527130fb4da612f6cbae5822f2c7eb61fb55110823ed8ba620e4a99e4a74dfb351516119522b49fb3966dfb9a689e73ffb287ac8ad853d8ed92e6b53d8ffc7f0d5b8e9113c545dd6a99d4580817cf7696320ef0bf53958510eb4d6867babd17f3274efa173a46eea91afba7e546ddf6fe936aa385e6c0a0e939ede6594c701a610277afaa1ec517e2ede8b7026fee5de6d3ec20866229f994d632a5d2a147e24d8d7bcf02e87ec96685c54214d8528bb8f432c9f26079c01b6b8203b945fda18b97ee768098ef94d1f5397eaf1a55c757349d7fbdc41a39c8f2d7b6eae71f3aba82a86687eb33854b0166caa32d08dacf03c6da5f43e32abf6cae" width="100px;" alt="Cercospora"/><br /><sub><b>Cercospora</b></sub></a><br /></td>
   <td align="center"><img src="https://storage.googleapis.com/kagglesdsdata/datasets/277323/658267/color/Corn_%28maize%29___Common_rust_/RS_Rust%201566.JPG?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20210917%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210917T200830Z&X-Goog-Expires=345599&X-Goog-SignedHeaders=host&X-Goog-Signature=1a5fab01a968b9b73e510432e94bf33d52fde9c4829401f6bf2e429b0ddfc65674a93b4ca4ec3322420d5561acbc0f71352497e45fda38d4646965215b0a7e8af626db14b11bcf1951812fc860120a6e1f518b4c0d800fbf3b96f73b47ddefb1996b9514125dd097e6b27d867b3af462fe693a74d6aa40b7b39174d11edd45e1384d91acc217283ca73e36b6b1a85c47e484498bdc82995c88ad7395bc1262fd9267712c97235eef68d783ef3bbf1d17115bc4d30503483a5a23714829d5f78505f696d5266e089e71d4ffdb7eb59c96325892e84a751465945e8e782ad0b23b4acb9905ce5d2e15f7a708821669d2c08766a0b679fbb672542f17c73b0a11d3" width="100px;" alt="Common rust"/><br /><sub><b>Common rust</b></sub></a><br /></td>
   <td align="center"><img src="https://storage.googleapis.com/kagglesdsdata/datasets/277323/658267/color/Corn_%28maize%29___Northern_Leaf_Blight/005318c8-a5fa-4420-843b-23bdda7322c2___RS_NLB%203853%20copy.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20210917%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210917T200921Z&X-Goog-Expires=345599&X-Goog-SignedHeaders=host&X-Goog-Signature=96d6a1c3dad277ede72c92b4149d37d2aa10d73d567df525eabe4dff138d0da1d8811dd5781956f7481cb6a5c9c5884364ec8c7a39b9fc3075f54a580c6649b46354cf35934f03aca5e3f7d12f452a61c059b343a368456dbf1799749081f41e32c4868fad3a9526e6f08455390d1ef7a42dac3332f6adf54429d11ce77e4fc39dbeb482bee4b100d26614ddd4854b4638e9bc57d2f6febb914adf7e53ba8690fed56f39803dffd7fc9c60a39923f6f2bdfcf1f2f137a5792b6f20022b2bcd7b3b493972305d84ae967b686d2f8e065186d9bf3038f3846cd92c03eaf04ecb6f3301f83fdfa20f6baf7473431d876ac43500ed905b8584685fbcdc01140a739a" width="100px;" alt="Northern leaf blight"/><br /><sub><b>Northern leaf blight</b></sub></a><br /></td>
 </tr>
</table>

This figure illustrates examples of maize leaf images being healthy and diseased.

## How to Train the Model?
You can use your system installed Python interpreter or IDE, or [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb) to run this model.
It was originally trained on the Colab Notebook, due to its availability of high speed GPUs. The step-by-step procedure for both the methods has been explained below:

* ***Using Google Colab***
1. Create an API key in Kaggle.To do this, go to the Kaggle [website](www.kaggle.com/) and open your user settings page.
![image](https://i.stack.imgur.com/jxGQv.png)

2. Next, scroll down to the API access section and click generate to download an API key.
![image](https://i.stack.imgur.com/Hzlhp.png)
This will download a file called kaggle.json to your computer. You'll use this file in Colab to access Kaggle datasets and competitions.

3. Navigate to https://colab.research.google.com/

4. Upload your kaggle.json file using the following snippet in a code cell:
```python
from google.colab import files
files.upload()
```

5. Install the kaggle API using:
```python
!pip install kaggle
```

6. Move the kaggle.json file into ~/.kaggle, which is where the API client expects your token to be located:
```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
```

7. Now, download the dataset using:
```python
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
```

8. Unzip the data using the following code snippet:
```python
from zipfile import ZipFile
file_name='chest-xray-pneumonia.zip'
with ZipFile(file_name,'r') as zip:
  zip.extractall()
  print('Done')
```

Now your dataset is uploaded and ready for use!

To train the model, simply paste the code in CNN.py into a code cell after completing the above process and run it.

For my complete Google Colab implementation, follow this [link](https://colab.research.google.com/drive/1YQ-QXX2xribxGC0C_a-EfTAu_7sfGQbK?usp=sharing)

* ***Using system installed Python interpreter or IDE***

1. Download the dataset to your system using this [link](https://www.kaggle.com/abdallahalidev/plantvillage-dataset), then unzip the image dataset.

2. Make sure that you have the Keras library installed. Run the following code in your Python environment:

```python
import keras
```
If the import is successful with no error message, the library is already installed.

3. In case of error message on import, follow this [link](https://www.tutorialspoint.com/keras/keras_installation.htm) for instructions to install Keras.
Then run the above code cell again to ensure proper installation.

4. Now, to train the model, set the folder containing the downloaded dataset as the working directory. Then simply paste the code available in CNN.py
and run it.


* To view the model summary and list of layers used, check Model Summary file from the repository.


* To check for individual images, uncomment the last section of the code, titled ' Checking for individial images' and replace 'enter image name' by the name of the image you wish to test.


## Using the model for other image classification tasks
Finally, this model can be used for any image classification task without effecting the accuracy. To use it for another purpose,
replace the images in the downloaded folder with the classes of the images you wish to classify or differentiate between, say cat or dog, keeping
the structure of the model as it is. In case , more than 2 classes of images are present, edit the final layer of the CNN and the compile command as follows:

```python
model.add(Dense(output_dim= 'number of output classes', activation='softmax')
```

```python
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
```



##### Copyright (c) 2020 Pranav-Jain
