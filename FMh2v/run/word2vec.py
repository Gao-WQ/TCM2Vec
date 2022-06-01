
from models.CBOW_model import CBOWModel
from utils.input_data import InputData
import torch.optim as optim
from tqdm import tqdm

for dim in [50, 100, 150, 200, 250, 300]:
    # hyper parameters
    WINDOW_SIZE = 3
    BATCH_SIZE = 64
    MIN_COUNT = 0
    EMB_DIMENSION = dim
    LR = 0.0001
    NEG_COUNT = 2


    class Word2Vec:
        def __init__(self, input_file_name, output_file_name):
            self.output_file_name = output_file_name
            self.data = InputData(input_file_name, MIN_COUNT)
            self.model = CBOWModel(self.data.word_count, EMB_DIMENSION, MIN_COUNT)
            self.lr = LR
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

        def train(self):
            print("CBOW Training......")
            pairs_count = self.data.evaluate_pairs_count(WINDOW_SIZE)
            print("pairs_count", pairs_count)
            batch_count = pairs_count / BATCH_SIZE
            print("batch_count", batch_count)
            process_bar = tqdm(range(int(batch_count)))
            for i in process_bar:
                pos_pairs = self.data.get_batch_pairs(BATCH_SIZE, WINDOW_SIZE)
                pos_u = [pair[0] for pair in pos_pairs]
                pos_w = [int(pair[1]) for pair in pos_pairs]
                neg_w = self.data.get_negative_sampling(pos_pairs, NEG_COUNT)

                self.optimizer.zero_grad()
                loss = self.model.forward(pos_u, pos_w, neg_w)
                loss.backward()
                self.optimizer.step()

                if i * BATCH_SIZE % 100000 == 0:
                    self.lr = self.lr * (1.0 - 1.0 * i / batch_count)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr

            self.model.save_embedding(self.data.id2word_dict, self.output_file_name)
    fn = '_S'
    w2v = Word2Vec(input_file_name='./data/StackOverflow(1040).txt', output_file_name='./data/embed/'+"word2vec_embed_"+str(EMB_DIMENSION)+ fn +".txt")
    w2v.train()
