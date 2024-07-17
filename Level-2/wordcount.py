import re
from collections import defaultdict

def wordcount(text):
    # 使用正则表达式匹配单词
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = defaultdict(int)
    
    # 统计每个单词出现的次数
    for word in words:
        word_count[word] += 1
    
    return dict(word_count)

# 测试例子
text = """
Got this panda plush toy for my daughter's birthday,
who loves it and takes it everywhere. It's soft and
super cute, and its face has a friendly look. It's
a bit small for what I paid though. I think there
might be other options that are bigger for the
same price. It arrived a day earlier than expected,
so I got to play with it myself before I gave it
to her.
"""
result = wordcount(text)
print(result)
