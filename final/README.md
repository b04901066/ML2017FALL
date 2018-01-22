# ML2017 Fall Final
## library used in training
* pyTorch
* jieba
* tensorboard_logger
* tqdm
* spacy
* six
* numpy
* pandas
* Python Standard Lib

## Sliding Window
### Test
Run

    cd src
    python3 HL2.py <path-to-test-v1.1.json> <output.csv>

## R-Net
### Required packages
You need `pytorch` and `tqdm`. The version tested is `torch==0.3.0.post4` and `tqdm==4.19.5`.

Also we use a customized version of `jieba`. Please run
    
    cd src/R-Net
    git clone https://github.com/ldkrsi/jieba-zh_TW
    
to fetch it.

### Running the program
Run

    cd src/R-Net
    python main.py model.pt <path-to-test-v1.1.json>

to generate output. The file `output-<date_time>.csv` is for uploading to Kaggle, and the file `text-<date_time>.csv` contains human-readable text.

### Since we use random initialize in pointer network,the variance of the output might be high...
### Please try more times if the F1-score is low in reproducing.
