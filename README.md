# hist-doc-segm

**Semantic Segmentation of Historical Documents using Deep Learning Architectures**

* Prediction of Semantic Segmentation masks for images of the **[Eparchos Dataset](https://zenodo.org/record/4095301#.YsV2rHZBzDc)**.
* Deep Learning Architectures used: [**dhSegment**](https://github.com/dhlab-epfl/dhSegment), [**U-Net**](https://github.com/zhixuhao/unet), [**VGG16**](https://github.com/machrisaa/tensorflow-vgg)
* This consists a demo of my [**Graduate Thesis**](https://drive.google.com/file/d/1MoOnG4wPs2h1XBP2vNGyT-Y3iS9OFyBp/view?usp=sharing).

![Demo Screenshot 2](https://github.com/anastzel/hist-doc-segm/blob/main/2.png)
![Demo Screenshot 3](https://github.com/anastzel/hist-doc-segm/blob/main/3.png)

## Authors

[Anastasios Tzelepakis](https://github.com/anastzel)

## Installation

1.It is recommended to install tensorflow (or tensorflow-gpu) independently using Anaconda distribution, in order to make sure all dependencies are properly installed.

2.Clone the repository using ```git clone https://github.com/anastzel/hist-doc-segm.git```

3.Install Anaconda or Miniconda 

4.Create a virtual environment and activate it

```
conda create -n hist_doc_segm python=3.6
source activate hist_doc_segm
```
5.Install dhSegment dependencies with ```pip install -r requirements.txt```

6.Install TensorFlow 1.13 with conda ```conda install tensorflow-gpu=1.13.1```

## Download Models

You can download the models needed from my [here](https://drive.google.com/drive/folders/1ixB7ifTz0YeIFowU_G59eujb5FUfB_7O?usp=sharing) (models have a large size, so GitHub doesn't allowed me to upload them).

After you have downloaded the models, place them inside the repository folder.

## Demo

To run the Demo open Anaconda prompt and navigate to the repository folder.

Then run ```streamlit run appication.py```.

A browser's tab should open containing the application's demo.

![Demo Screenshot 3](https://github.com/anastzel/hist-doc-segm/blob/main/1.png)
