
TEST = False

# 'WIKI', 'MIRFlickr' or 'NUSWIDE'
DATASET = 'WIKI'


if DATASET == 'WIKI':


    alpha = 0.3
    beta = 0.4
    lamb = 0.3

    pho3 = 0.15
    gamma = 0.45
    mu = 1.5

    INTRA = 0.3

    LR_IMG = 0.001
    LR_TXT = 0.01

    DATA_DIR = 'dataset/WIKI/images'
    LABEL_DIR = 'dataset/WIKI/raw_features.mat'
    TRAIN_LABEL = 'dataset/WIKI/trainset_txt_img_cat.list'
    TEST_LABEL = 'dataset/WIKI/testset_txt_img_cat.list'



if DATASET == 'MIRFlickr':


    alpha = 0.5
    beta = 0.1
    lamb = 0.4

    pho3 = 0.15
    gamma = 0.45
    mu = 1.5

    INTRA = 0.1

    LR_IMG = 0.001
    LR_TXT = 0.01

    LABEL_DIR = 'dataset/Flickr/mirflickr25k-lall.mat'
    TXT_DIR = 'dataset/Flickr/mirflickr25k-yall.mat'
    IMG_DIR = 'dataset/Flickr/mirflickr25k-iall.mat'


if DATASET == 'NUSWIDE':


    alpha = 0.4
    beta = 0.3
    lamb = 0.3

    pho3 = 0.25
    gamma = 0.5
    mu = 1.32
    INTRA = 0.1

    LR_IMG = 0.001
    LR_TXT = 0.01

    LABEL_DIR = 'dataset/NUS-WIDE/nus-wide-tc10-lall.mat'
    TXT_DIR = 'dataset/NUS-WIDE/nus-wide-tc10-yall.mat'
    IMG_DIR = 'dataset/NUS-WIDE/nus-wide-tc10-iall.mat'



HASH_BIT = 128

BATCH_SIZE = 32

MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

GPU_ID = 0
NUM_WORKERS = 8
EPOCH_INTERVAL = 2
NUM_EPOCH = 600
EVAL_INTERVAL = 40

MODEL_DIR = './checkpoint'

