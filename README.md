# Deep learning recognition of handprinted letters

To use you need install [anaconda](https://www.anaconda.com/products/individual). 
Clone repository and download [model](https://drive.google.com/file/d/1Pwn_XnIs_D8vp5x66cgKtNlHHb7zd-3Y/view?usp=sharing) (and move it in project folder), then go to folder with project and run:

```
conda env create -f environment.yml
conda activate ml_test_task
python ml.py [path_to_image] [path_to_model | empty (standart "model.h5" near py file)]
```
