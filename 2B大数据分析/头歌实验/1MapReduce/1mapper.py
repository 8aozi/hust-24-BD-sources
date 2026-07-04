# Mapper 1
'''
使用 9 个并发线程分别处理 9 个不同的输入文件（source01 ~ source09）。
每个线程，将原始文本转换成 <单词, 1> 形式的中间键值对，并写入临时文件，
供后续的 Combiner / Reducer 使用。
'''

import threading
import os
from time import time
from collections import defaultdict


class MapperNode(threading.Thread):
    def __init__(self, seq):
        """
        :param seq:线程序号（1-9）
        :var self.combier_dict:存储combiner结果的字典
        """
        self.seq = seq
        self.combiner_dict = defaultdict(int)
        self.check_dir()
        threading.Thread.__init__(self)

    def check_dir(self):
        """
        检查文件目录
        """
        if not os.path.exists('./mapper'):
            os.makedirs('./mapper')

    def run(self):
        start_time = time()
        self.simple_mapper()
        end_time = time()
        #print("Thread %d done, use time: %f" % (self.seq, end_time-start_time))

    def simple_mapper(self):
        """
        对source文件进行map操作，得到<token, 1>元组，存储在temp文件中
        为了模拟大数据情景进行了I/O操作，产生的temp文件在combiner步骤中删除

        读入sourcename对应文件，写入到filename对应的文件，请按读入顺序写入以方便检查
        """
        sourcename = '/data/workspace/myshixun/MapReduce实验/source/source0' + str(self.seq)
        filename = './mapper/mapper0' + str(self.seq) + '_temp'

        ##########begin#############
        # 读取源文件
        with open(sourcename, 'r', encoding='utf-8') as infile:
            content = infile.read()
        # 分词器：以逗号和换行符作为分隔符分割单词
        words = content.replace('\n', ',').split(',')
        with open(filename, 'w', encoding='utf-8') as outfile:
            for word in words:
                if word:   # 过滤空字符串
                   # 输出成 <key,value>
                    outfile.write(f"{word}\t1\n")
        ##########end###############


if __name__ == '__main__':
    # 维护9个线程，模拟9个mapper节点
    threads = []
    for _ in range(9):
        new_thread = MapperNode(_ + 1)
        threads.append(new_thread)
        new_thread.start()

    # 等待所有 mapper 线程执行完成
    for t in threads:
        t.join()

    # 按顺序输出保存的 9 个文件的第一行
    for i in range(1, 10):
        filename = f'./mapper/mapper0{i}_temp'
        print(filename, end=': ')
        with open(filename, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            print(first_line if first_line else '(空文件)')