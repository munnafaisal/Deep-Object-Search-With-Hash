# Deep-Object-Search-With-Hash

## A Short Description Of Project

This repo contains an object search mechanism with hashing techniques using deep features of available pretrained object detection models. It can detect multiple objects from a given video then read features of those objects from a chosen feature extractor. Afterwards those features are transformed into hash codes using LSH(Locality Sensitive Hashing) technique and indexed into a python “Dictionary”.Finally for a given video it takes objects from the video as a query image and displays the close matched images which features have been previously indexed with hash codes.      

![Test Image](https://github.com/munnafaisal/Deep-Object-Search-With-Hash/blob/master/Demo_Video/Shotwell%20%20query_result.mp4%20-%203.jpg)

## Demo Video Link

[click here](https://drive.google.com/file/d/15LqCcMkJmii4LY-Nds70ZwMXON6RDLyA/view?usp=sharing)

## Acknowledgements 

To learn more about LSH(Locality Sensitive Hashing) 

visit:

https://github.com/kayzhu/LSHash 

and

https://github.com/pixelogik/NearPy


## Environment Setup:


### Installation instructions

_Run the commands in a terminal or command-prompt.

- Install `Python 3.6 or >3.6` for your operating system, if it does not already exist.

 - For [Mac](https://www.python.org/ftp/python/3.6.8/python-3.6.8-macosx10.9.pkg)

 - For [Windows](https://www.python.org/ftp/python/3.6.8/python-3.6.8-amd64.exe)

 - For Ubuntu/Debian

 ```bash
 sudo apt-get install python3.6
 ```

 Check if the correct version of Python (3.6) is installed.

 ```bash
 python --version
 ```

**Make sure your terminal is at the root of the project i.e. where 'README.md' is located.**

* Get `virtualenv`.

 ```bash
 pip install virtualenv
 ```

* Create a virtual environment named `.env` using python `3.6` and activate the environment.

 ```bash
 # command for gnu/linux systems
 virtualenv -p $(which python3.6) .env

 source .env/bin/activate
 ```
* If any error occurs to install a virtual environment you can see this [link](https://github.com/anisrfd/Python-Virtualenv-Setup/blob/master/Python_virtualenv_setup.md)

 
* Install python dependencies from requirements.txt.
 ```bash
  pip install -r requirements.txt
  ```


## How to run

After installing all the required libraries run the following commands in the terminal. The details of required input parameters have been described in the following section.



* Step 1: 
First run object detection script from terminal using following command 

```bash
python object_detection_YOLO.py --video_dir <video file path>
 ```
Example : 
```bash
python object_detection_YOLO.py --video_dir VideoFileDirectory
 ```
This will create a directory named “temp” in your project directory and under this directory, subdirectories will also be created according to object class or category.
Object class wise cropped images will be saved into those subdirectories. 
	
Currently 7 object classes are available. You can use other object classes by changing the parameter name and value.
	
    Person
    Bicycle
    Car
    Bike
    Bus
    truck 
    Chair
            

* Step 2 : 
After the completion of step 1 , run the the following command from terminal

Example :
```bash
python start.py --range 350 --hash_length 48 --type discrete --function pca --n_of_HPT 5 --n_of_NN 20  --DSF 16 --QOC person --RNF True --TVD VideoFileDirectory
 ```
Options for feature extraction from pretrained models will appear on terminal.

Currently 3 pretrained models are available:

    Resnet-50
    VGG-19
    MobileNet-SSD

For the first time select Resnet-50 and hit Enter Button then you will see features of object images saved in “temp/person” being extracted and indexed.  

After indexing all features a video(path given in the input arguments) will start to play and press 'Q' to pause the video you will see another gallery window showing Query Image and corresponding Query Results. Then again press 'Q' to continue.

## Description of Input arguments

##### range : 
Number of images to be read from directory

##### hash_length : 
Length of hash key to be generated from features
##### type :
Type of hash keys, currently "Binary and Discrete" hash types are available

##### functions :
Type of hash functions

    --n_of_HPT :
    Number of hash function per table (Python dictionary object)
    
    --n_of_NN
    Number of nearest neighbour for NN search
    
    --RNF
    Read New Features (RNF) from object image directory

	--DSF
	Downsampling Factor
	
	--QOC
	Query Object Class

	--TVD
	Test Video Directory/Path
	
	--OID
	Object Image Directory
	
	--OFD
	Object Features Directory
	

## Contacts

1. Md. Faisal Ahmed Siddiqi (ahmedfaisal.fa21@gmail.com)

2. Anis-Ul-Isalm Rafid (au.i.rafid15@gmail.com)
    [linkedin Profile](https://www.linkedin.com/in/anis-ul-islam-rafid-54s18m/)


 

