# shuffler 3
'''
将combiner预聚合后的键值对按单词进行哈希分区，供reducer使用
分区内排序，写入不同文件（每个 Mapper 产生 3 个文件，共 27 个）
'''

import threading
import os
from time import time
from collections import defaultdict
import hashlib

class MapperNode(threading.Thread):
    def __init__(self, seq):
        """
        :param seq:线程序号（1-9）
        :var self.combiner_dict:存储combiner结果的字典
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
        self.combiner()
        self.shuffler()
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
        with open(sourcename, 'r', encoding='utf-8') as infile, \
             open(filename, 'w', encoding='utf-8') as outfile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                # 按逗号分割一行中的多个单词
                words = line.split(',')
                for word in words:
                    if word:
                        outfile.write(f"{word}\t1\n")
        ##########end###############

    def combiner(self):
        """
        对temp文件中的元组进行combine操作，得到<token, value>元组
        由于combine后的数据规模会成倍缩小，故将结果直接存储在内存中
        将元组哈希存储在字典self.combiner_dict中
        """
        filename = './mapper/mapper0' + str(self.seq) + '_temp'

        ##########begin#############
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 格式为 "word\t1"
                parts = line.split('\t')
                if len(parts) == 2:
                    word, count = parts[0], int(parts[1])
                    self.combiner_dict[word] += count
        ##########end###############
        
        os.remove(filename)
    
    def shuffler(self):
        """
        将combiner步骤中得到的元组按token进行分片
        请使用hash函数对token进行分配，将token分入三片，写入w1，w2，和w3中，对应三个reducer（总共会有27份临时文件）
        请对token按照字母进行排序后写入
        hash方式请使用int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)对word进行哈希，方便测试结果

        """
        filename = './mapper/mapper0' + str(self.seq)
        w1 = open(filename + '_part1', 'w')
        w2 = open(filename + '_part2', 'w')
        w3 = open(filename + '_part3', 'w')
        
        ##########begin#############
        # 按分区收集 (word, count) 对
        partitions = {1: [], 2: [], 3: []}
        # 计算单词的 MD5 哈希值,为 32 位十六进制字符串，并转换成int
        for word, count in self.combiner_dict.items():
            h = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
            # hash 分区
            part = (h % 3) + 1   # 0->1, 1->2, 2->3
            partitions[part].append((word, count))
        
        # 对每个分区内的单词按字母排序
        for part in [1, 2, 3]:
            partitions[part].sort(key=lambda x: x[0])       
            # 排序后的键值对写入对应分区文件
            for word, count in partitions[part]:
                if part == 1:
                    w1.write(f"{word}\t{count}\n")
                elif part == 2:
                    w2.write(f"{word}\t{count}\n")
                else:
                    w3.write(f"{word}\t{count}\n")

        ##########end###############
        
        w1.close()
        w2.close()
        w3.close()

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

    # 按顺序输出 shuffle 保存的 27 个文件的第一行
    for i in range(1, 10):
        for j in range(1, 4):
            filename = f'./mapper/mapper0{i}_part{j}'
            print(filename, end=': ')
            with open(filename, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                print(first_line if first_line else '(空文件)')