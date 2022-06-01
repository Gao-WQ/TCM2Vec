import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data  as tud
from torch.autograd import Variable
torch.manual_seed(100)
device = torch.device("cpu")




import os
import re
import sys
import math
import time
import copy
import random
random.seed(100)
import gensim
import sklearn
import numpy as np
np.random.seed(100)
from numpy import linalg as LA

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from datetime import timedelta
from ast import literal_eval
from tkinter import _flatten
from tensorboardX import SummaryWriter
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from collections import Counter





random.seed(100)
np.random.seed(100)
torch.manual_seed(100)
torch.cuda.manual_seed(100)
torch.cuda.manual_seed_all(100)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] =str(100)