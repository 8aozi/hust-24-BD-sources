# reducer4
'''
每个 Reducer 采取的 reduce() 方法：
- 读取所有 Mapper 生成的对应分区文件（例如 Reducer 1 读取所有 *_part1 文件）。
- 对相同单词的计数进行累加，存入 self.reduce_dict。
- 将累加结果按单词排序后写入自己的输出文件（./reducer/reducer01 等）。

主程序在所有 Reducer 完成后：
- 合并三个 Reducer 的输出，全局排序后写入 wordCount.txt。
'''
import threading
import os
from time import time
from collections import defaultdict


class ReduceNode(threading.Thread):
    def __init__(self, seq):
        """
        :param seq:线程序号（1-3）
        :var self.reduce_dict:存储reducer结果的字典
        """
        self.seq = seq
        self.reduce_dict = defaultdict(int)
        self.check_dir()
        threading.Thread.__init__(self)

    def check_dir(self):
        """
        检查文件目录
        """
        if not os.path.exists('./reducer'):
            os.makedirs('./reducer')

    def run(self):
        start_time = time()
        self.reduce()
        end_time = time()
        #print("Thread %d done, use time: %f" % (self.seq, end_time-start_time))

    def reduce(self):
        """
        每个reducer拉取9个mapper节点中对应part的文件
        将结果汇总后存储在self.reduce_dict中
        """
        ######begin#############
        # 遍历9个mapper节点，构造对应分区文件，打开并读取（由shuffler输入）
        for i in range(1, 10):   
            filename = f'./mapper/mapper0{i}_part{self.seq}'

            with open(filename, 'r', encoding='utf-8') as f:

                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')  # 按制表符分割
                    if len(parts) == 2:
                        word, count = parts[0], int(parts[1])
                        # 累加
                        self.reduce_dict[word] += count
        ########end###############
        with open(f'./reducer/reducer0{self.seq}', 'w', encoding='utf-8') as wf:
            for k in sorted(self.reduce_dict.keys()):
                wf.write(k + "\t" + str(self.reduce_dict[k]) + "\n")


if __name__ == '__main__':
    # 线程池
    threading_pool = []
    # 维护三个线程，模拟三个reducer节点
    for _ in range(1, 4):
        new_thread = ReduceNode(_)
        new_thread.start()
        threading_pool.append(new_thread)

    """
    三个reducer线程全部结束后进行汇总
    汇总结果存储在wordCount.txt中（全局排序）
    """
    for thread in threading_pool:
        thread.join()

    # 读取所有reducer文件，合并后全局排序
    all_words = defaultdict(int)
    filedir = './reducer'
    for filename in os.listdir(filedir):
        with open(os.path.join(filedir, filename), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                word, count = line.split('\t')
                all_words[word] += int(count)

    # 全局排序后写入wordCount.txt
    with open('./wordCount.txt', 'w', encoding='utf-8') as wf:
        for word in sorted(all_words.keys()):
            wf.write(f"{word}\t{all_words[word]}\n")

    print("===== Final Result (Top 10 lines) =====")
    with open('./wordCount.txt', 'r', encoding='utf-8') as rf:
        for i in range(10):
            line = rf.readline()
            if not line:
                break
            print(line.strip())