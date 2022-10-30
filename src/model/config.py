import os

BETA_START = 1e-4
BETA_END = 0.02
NOISE_STEPS = 1000
N_HEADS = 2
TIME_EMBEDDING_SIZE = 256

LEARNING_RATE = 1e-4
EMA_DECAY = 0.995
EMA_START_STEP = 2000

RUN_EAGERLY = True #False
EPOCHS = 100
BATCH_SIZE = 4

IMAGE_RESIZE = (64,64)
#IMAGE_CROP = (64,64)

#Saving/Loading
SAVE_MODEL = True
LOAD_MODEL = True
EPOCHS_BEFORE_SAVE = 5

#Paths
WEIGHTS_PATH = "weights"
UNET_WEIGHTS_PATH = os.path.join(WEIGHTS_PATH,"unet_{}.npy")

TRAIN_PATH = os.path.join("data","training dataset")

N_SAMPLES = 2
SAMPLES_PATH = os.path.join("data","samples","sample_ep{}_{}.jpg")