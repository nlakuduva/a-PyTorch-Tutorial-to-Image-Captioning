import os
import numpy as np
import h5py
import json
import torch
import string
from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import ast

def read_files(split, train_image_paths, train_image_captions, word_freq):
    bad_images = ['ac80c5633d76c27b352ee6352ddbb3', 'ac88d66ad654f2739bbfdfbe55c2bdb', 'ac8b1beeb050fa26b970dc2fee5ef539', 'ac81b9680ba5cab4436ad2528555da', 'ac8788acffc6226816827a943e69bbf', 'ac8d98a8fab765d92a678295fe9b2d', 'ac866ff82c6319994c8568a91f8aa2a', 'ac845e9d3081d9415d8a4b49c7dca7', 'ac8aaf341c293ed43ce33358c4801c', 'ac8b31a49a2ed5375724ad7e8fd80ff', 'ac827b33adaaefb430ce867c4fec7732', 'ac84b4dc481dd318ae3977a755bd3742', 'ac83a308c357da220a394e7164da956', 'ac8cfa458c978d903e87459ffd66c2a', 'ac84b63a2b27ddc08dd7e769593b25c2', 'ac83a907bc318b96b466179876ad093', 'ac854f4a9d99e34ec5223b991aa2c887', 'ac80835da2c2e5f021dd63ed56d0be93', 'ac8a4530c32027b32d52bc899697d8', 'ac8ea6f73a10a31c3d4920c174fe96', 'ac882dd35d8bf3cee5efeddbe1f399', 'ac8d22b10c9dee47015761898ae75', 'ac8ba82a7ae6d761e1d3582dddbdecdf', 'ac87681e52e3a709e48cb40ce18bedf', 'ac827d5424278a62aa6aad18d439a', 'ac83f9f225f695c2a633d44dcbbce55', 'ac833f723b92e13b6e314d644f76837', 'ac8a241d2041cb26eba548ec1e7d128', 'ac849ffe6d25ee9bb21787a39c1926', 'ac8287a425804c7c8cc56ca590de1435', 'ac8567fbe08f7d825bf47ea6846693dc', 'ac85d4417242b9a36c3f45a6a32a138', 'ac8bcdc098f3698665b91ab9146cc3', 'ac8e6eca5713ee25f631e79d9a3355ac', 'ac817cd3ccfec3358265dee15ec616af', 'ac84ecd0fc1f3e17772d8f561a11add', 'ac822b755268b2a6ce231cc0e1ad588', 'ac83cdb55a9e6a79ecc879451dacb3', 'ac8d7d1ce0f32d1e7ee4d838c9b1b94', 'ac80a9f51a66169dda8eec89cda2a289', 'ac8ddea4829c7eba37ac0c81a7ef634', 'ac83184fd40c475501ef12160eefa1c', 'ac85a84da55bfb3497f038822344596d', 'ac86f15b386ea87d3d240fac81f166c', 'ac8dd7f5795743061f480a56aec7c97', 'ac841340786841ee7e665051fe58d9', 'ac8176a2fb143c79c22488d104ece72', 'ac89925318fa67f3e018da2547bbe2', 'ac87838e7846735b27d54a7d5dbc4ee', 'ac8628ffeed36884336ceab1586cb1', 'ac81db90ac691dfdd275b2e6ec299ca4', 'ac8f9ff369308ac4d3643d3114c6718b', 'ac8ee3225ea20433642b347f8fa8d81', 'ac83d37bf87d21f31bcbc3c3f7714f99', 'ac85ef996efd9aed1f91293c1552e9e5']
    # split_path = '/Users/Nandhini/ParlAI/data/personality_captions/'
    # split_path = '/Users/Nandhini/ParlAI/data/personality_captions/baby/'
    # image_path = '/Users/Nandhini/Documents/School-Junk/UT-Senior:Master-Year/GNLP/final/images'
    # image_path = '/work/06117/lakuduva/maverick2/ParlAI/data/yfcc_images'

    # TACC
    split_path = '/work/06117/lakuduva/maverick2/ParlAI/data/personality_captions'
    image_path = '/work/06117/lakuduva/maverick2/ParlAI/data/yfcc_images/'

    f = open("{split_path}/{split}.json".format(split_path=split_path, split=split))
    data = next(f)
    data = ast.literal_eval(data)
    if split is 'train':
        print("TAKING LAST FEW")
    data = data[0:155000] if split is 'train' else data
    for triplet in data:
        if triplet["image_hash"] in bad_images:
            continue
        train_image_paths.append("{image_path}/{img}.jpg".format(image_path=image_path, img=triplet["image_hash"]))
        toks = [tok.strip(string.punctuation).lower() for tok in triplet["comment"].split(" ")]
        train_image_captions.append([toks])
        word_freq.update(toks)

def create_input_files(dataset, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'coco', 'flickr8k', 'flickr30k', 'personality', 'baby'}

    # Read Karpathy JSON
    # with open(karpathy_json_path, 'r') as j:
    #     data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    read_files("train", train_image_paths, train_image_captions, word_freq)
    read_files("val", val_image_paths, val_image_captions, word_freq)
    read_files("test", test_image_paths, test_image_captions, word_freq)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    print(len(train_image_paths))
    assert len(val_image_paths) == len(val_image_captions)
    len(val_image_paths) 
    assert len(test_image_paths) == len(test_image_captions)
    len(test_image_paths) 

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images
                try:
                    img = imread(impaths[i])
                    if len(img.shape) == 2:
                        img = img[:, :, np.newaxis]
                        img = np.concatenate([img, img, img], axis=2)
                    img = imresize(img, (256, 256))
                    img = img.transpose(2, 0, 1)
                    assert img.shape == (3, 256, 256)
                    assert np.max(img) <= 255

                    # Save image to HDF5 file
                    images[i] = img

                    for j, c in enumerate(captions):
                        # Encode captions
                        enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                            word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                        # Find caption lengths
                        c_len = len(c) + 2
                        enc_captions.append(enc_c)
                        caplens.append(c_len)
                except OSError:
                    print("Skipping image", impaths[i])

            # Sanity check
            # assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            print("HOW MANY CAPTIONS", len(enc_captions))
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)

