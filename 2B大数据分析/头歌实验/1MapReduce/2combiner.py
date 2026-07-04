# combiner 2
'''
读取 Mapper 输出的临时文件（含有大量 <word, 1> 记录）。
对每个单词出现的次数进行累加（求和）。
将聚合结果保存在内存
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
        self.combiner()
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
        with open(sourcename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 处理分隔符，统一按逗号分割单词
        content = content.replace('\n', ',')
        words = content.split(',')
        
        # 写入格式：单词 1，按读入顺序写入
        with open(filename, 'w', encoding='utf-8') as f:
            for word in words:
                word = word.strip()
                if word:
                    f.write(f"{word} 1\n")
        ##########end###############

    def combiner(self):
        """
        对temp文件中的元组进行combine操作，得到<token, value>元组
        由于combine后的数据规模会成倍缩小，故将结果直接存储在内存中
        将元组哈希存储在字典self.combiner_dict中
        """
        filename = './mapper/mapper0' + str(self.seq) + '_temp'

        ##########begin#############
        # 读取临时文件，聚合词频
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                #去除行首尾空白，跳过空行。
                line = line.strip()   
                if not line:
                    continue
                word, count_str = line.split()
                # 对单个线程内（mapper节点）的单词，累加计数
                self.combiner_dict[word] += int(count_str)   
        ##########end###############
        
        os.remove(filename)
    
        
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

    # 严格匹配预期输出格式
    for i, thread in enumerate(threads, 1):
        print(f"Mapper {i} Combiner 结果:")
        # 按字典序输出前3个词，格式和预期完全一致
        for word, count in sorted(thread.combiner_dict.items())[:3]:
            print(f"  {word}: {count}")
        # 去掉多余换行，避免Mapper之间出现空行
        print(f"  ... 共 {len(thread.combiner_dict)} 个词")