# Copyright(c) 2023, KaoruXun(尹喆勋)
# Developed for the "digital intelligence" empowerment rural revitalization
# practice group of North China Electric Power University

import jieba
import jieba.analyse
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model
import openpyxl as xl
import re
import pickle
import operator

import matplotlib.pyplot as plt
from wordcloud import WordCloud

def extract_keywords(text, top_k=9):
    keywords = jieba.analyse.textrank(text, topK=top_k, withWeight=False)
    return keywords


# 使用jieba进行中文分词 并移除标点符号
def tokenize(text):
    # return list(jieba.cut_for_search(text))
    # return [
    #     word for word in jieba.cut(text, cut_all=False) if re.match(r"^[\w]+$", word)
    # ]
    return extract_keywords(text)


def get_corpus(sentences):
    # 对文本进行分词
    tokenized_sentences = [tokenize(sentence) for sentence in sentences]
    return tokenized_sentences


def gen_word2index(words):
    # 建立词汇表
    vocab = sorted(set(words))
    word_to_index = {word: index for index, word in enumerate(vocab)}
    index_to_word = {index: word for word, index in word_to_index.items()}
    vocab_size = len(vocab)
    return vocab, word_to_index


def training(sentence):
    tokenized_sentences = get_corpus(sentence)

    # 将分词后的文本转换为单词列表
    words = []
    for sentence in tokenized_sentences:
        words.extend(sentence)

    # 建立词汇表
    vocab, word_to_index = gen_word2index(words)
    vocab_size = len(vocab)

    # 构建Word2Vec词向量模型
    embedding_dim = 100
    input_target = Input(shape=(1,))
    input_context = Input(shape=(1,))
    embedding_layer = Embedding(vocab_size, embedding_dim)
    target_embed = embedding_layer(input_target)
    context_embed = embedding_layer(input_context)

    # 使用点乘计算两个词向量的相似度
    dot_product = tf.keras.layers.Dot(axes=2)([target_embed, context_embed])

    # 定义神经网络模型
    x = GlobalAveragePooling1D()(dot_product)
    x = Dense(64, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=[input_target, input_context], outputs=output)

    # 编译模型
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", "mse"]
    )

    # 创建训练数据
    positive_pairs = []
    negative_pairs = []
    for i, sentence in enumerate(tokenized_sentences):
        for j in range(len(sentence)):
            for k in range(j + 1, len(sentence)):
                positive_pairs.append(
                    (word_to_index[sentence[j]], word_to_index[sentence[k]])
                )
                # 随机选择一个负样本
                negative_word_index = np.random.randint(0, vocab_size)
                while negative_word_index == word_to_index[sentence[j]]:
                    negative_word_index = np.random.randint(0, vocab_size)
                negative_pairs.append((word_to_index[sentence[j]], negative_word_index))

    target_words = np.array(
        [pair[0] for pair in positive_pairs + negative_pairs], dtype=np.int32
    )
    context_words = np.array(
        [pair[1] for pair in positive_pairs + negative_pairs], dtype=np.int32
    )
    labels = np.array(
        [1] * len(positive_pairs) + [0] * len(negative_pairs), dtype=np.int32
    )

    # 训练模型
    model.fit(x=[target_words, context_words], y=labels, epochs=10, batch_size=16)

    # 获取单词的词向量
    # word_vectors = embedding_layer.get_weights()[0]

    # 打印词向量结果
    # for word, index in word_to_index.items():
    #     print(f"{word}: {word_vectors[index]}")

    # 保存模型到文件
    model.save("trained_model/rnn.keras")

    # # 保存词向量数据
    # with open("trained_model/word_vectors.pkl", "wb") as f:
    #     pickle.dump(word_vectors, f)

    # # 保存词汇表
    # with open("trained_model/vocab.pkl", "wb") as f:
    #     pickle.dump(vocab, f)

    # 保存词汇索引
    with open("trained_model/word_to_index.pkl", "wb") as f:
        pickle.dump(word_to_index, f)

    print(f"模型词汇索引量为: {len(word_to_index)}")

    return


def test(input_sentence):
    # 加载词汇索引
    with open("trained_model/word_to_index.pkl", "rb") as f:
        word_to_index = pickle.load(f)

    # 加载已保存的模型
    loaded_model = tf.keras.models.load_model("trained_model/rnn.keras")

    loaded_model.summary()

    # 获取 embedding_layer
    embedding_layer = loaded_model.get_layer("embedding")

    # 获取词汇表和词向量
    word_vectors = embedding_layer.get_weights()[0]
    # vocab = list(embedding_layer.get_vocabulary())

    # 使用训练好的模型获取单词的词向量
    # word = "喜欢"
    # if word in word_to_index:
    #     word_index = word_to_index[word]
    #     word_vector = loaded_model.layers[2].get_weights()[0][word_index]
    #     print(f"{word} 的词向量：{word_vector}")
    # else:
    #     print(f"{word} 不在词汇表中。")

    # 提取句子关键词
    def extract_keywords(sentence, word_vectors):
        words = tokenize(sentence)
        word_vectors = [
            word_vectors[word_to_index[word]] for word in words if word in word_to_index
        ]
        if word_vectors:
            sentence_vector = np.mean(word_vectors, axis=0)
            similarities = np.dot(word_vectors, sentence_vector) / (
                np.linalg.norm(word_vectors, axis=1) * np.linalg.norm(sentence_vector)
            )
            top_indices = np.argsort(-similarities)[:8]
            keywords = [words[i] for i in top_indices]
            return keywords
        else:
            return []

    all_keywords = []

    # 测试句子关键词提取
    for sentence in input_sentence:
        keywords = extract_keywords(sentence, word_vectors)
        print(f"{sentence} \n关键词: {keywords}")
        all_keywords += keywords

    return all_keywords


def count_word_frequency(text):
    # 使用空格分隔文本中的单词
    words = text
    # 创建一个空字典来存储单词和它们的频率
    word_freq = {}
    # 遍历文本中的每个单词
    for word in words:
        # 去除标点符号
        word = word.strip(".,!?\"'()[]{}:;")
        # 将单词转换为小写，以便忽略大小写
        word = word.lower()
        # 更新字典中的单词频率
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    return word_freq


if __name__ == "__main__":
    training_sentences = []

    training_max_row = 5000
    training_file = "训练评论集.xlsx"

    wb = xl.load_workbook(training_file)
    # 选择默认的工作表
    sheet = wb.active
    for row in sheet.iter_rows(
        min_row=2, max_row=training_max_row, min_col=1, max_col=3, values_only=True
    ):
        comment, comment_time, comment_star = row
        comment = re.sub(r"\n", "", comment)
        training_sentences.append(comment)

    training(training_sentences)
