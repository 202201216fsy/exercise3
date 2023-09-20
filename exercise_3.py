import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

# 读取Gutenberg数据集中的Moby Dick文件
moby_dick = gutenberg.raw('melville-moby_dick.txt')

# Tokenization - 分词
tokens = word_tokenize(moby_dick)

# Stop-words filtering - 停用词过滤
stop_words = set(stopwords.words('english'))
filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalnum()]

# Parts-of-Speech (POS) tagging - 词性标注
pos_tags = nltk.pos_tag(filtered_tokens)

# POS frequency - 词性频率
tag_fd = FreqDist(tag for (word, tag) in pos_tags)
tag_counts = tag_fd.items()

# 打印所有词性和它们的出现次数
print("POS frequency:")
for tag, count in tag_counts:
    print(tag + ": " + str(count))

# Lemmatization - 词形归并
lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(word, pos) for (word, pos) in pos_tags[:20]]
print("Lemmatized tokens (top 20):")
print(lemmas)

# Plotting frequency distribution - 绘制词性频率分布图
tags = [tag for (word, tag) in pos_tags]
freq_dist = nltk.FreqDist(tags)
freq_dist.plot(title="POS Frequency Distribution", cumulative=False)
plt.show()