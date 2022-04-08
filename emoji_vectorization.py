import gensim.models as gsm
import numpy as np


def vectorize_emoji(emoji_list):
    """
    Will output a vectorized version of the emoji list. 300 dimensions for every row.
    """
    e2v = gsm.KeyedVectors.load_word2vec_format('emoji2vec.bin', binary=True)

    # change sequence of emoji into a list of individual emojis
    emooo = []
    for i in range(len(emoji_list)):
        emo = []
        for j in range(len(emoji_list[i])):
            emo.append(emoji_list[i][j])
        emooo.append(emo)

    # remove unwanted characters
    for i in range(len(emooo)):
        for j in emooo[i]:
            if j == '🏼':
                emooo[i].remove(j)
            if j == '':
                emooo[i].remove(j)
            if j == ',':
                emooo[i].remove(j)
    
    # get vector representation of each emoji
    emo_vecs = []
    for i in range(len(emooo)):
        emo_vecs.append([])
        for j in range(len(emooo[i])):
            try:
                emo_vecs[i].append(e2v[emooo[i][j]])
            except:
                emo_vecs[i].append(np.zeros(300))

    # get average of each emoji
    emo_vecs_avg = []
    for i in range(len(emo_vecs)):
        emo_vecs_avg.append(np.mean(emo_vecs[i], axis=0))

    return emo_vecs_avg