import random

import numpy as np


class Vertex:
    def __init__(self, vid, nodes):
        # 节点编号
        self._vid = vid
        self._nodes = nodes

    def get_nodes(self):
        return len(self._nodes)

    def propagation(self):
        return self._nodes

    @property
    def nodes(self):
        return self._nodes


class CDEPS:
    def __init__(self, g):
        self._G = g
        self._vid_vertex = {}  # 需维护的关于结点的信息(结点编号，相应的Vertex实例)

        for vid in self._G.keys():
            # 刚开始社区编号就是节点编号
            self._vid_vertex[vid] = Vertex(vid, {vid})

    def merged1(self, element, element_neighbor):
        """
        合并度为1的节点
        :param element: 度为1的节点
        :param element_neighbor: 节点的邻居
        :return:
        """
        self._vid_vertex[element_neighbor].nodes.update(self._vid_vertex[element].nodes)
        self._vid_vertex.pop(element)
        self._G.pop(element)
        self._G[element_neighbor].pop(element)

    def merged2(self, element, element_neighbor):
        """
        合并度为2的节点
        :param element: 要合并的节点i
        :param element_neighbor: i的两个邻居j，k
        :return:
        """
        if len(self._G[element_neighbor[0]].keys()) > len(self._G[element_neighbor[1]].keys()):
            self._vid_vertex[element_neighbor[0]].nodes.update(self._vid_vertex[element].nodes)
            if self._G[element_neighbor[0]][element_neighbor[1]]:
                self._G[element_neighbor[0]][element_neighbor[1]] = self._G[element_neighbor[0]][
                                                                        element_neighbor[1]] + (0.5 * self._G[element][
                    element_neighbor[0]] * self._G[element][element_neighbor[1]])
            else:
                self._G[element_neighbor[0]][element_neighbor[1]] = (
                        0.5 * self._G[element][element_neighbor[0]] * self._G[element][element_neighbor[1]])
            self._G[element_neighbor[1]][element_neighbor[0]] = self._G[element_neighbor[0]][element_neighbor[1]]
            self._vid_vertex.pop(element)
            self._G.pop(element)
            self._G[element_neighbor[0]].pop(element)
            self._G[element_neighbor[1]].pop(element)
        else:
            self._vid_vertex[element_neighbor[1]].nodes.update(self._vid_vertex[element].nodes)
            if self._G[element_neighbor[0]][element_neighbor[1]]:
                self._G[element_neighbor[0]][element_neighbor[1]] = self._G[element_neighbor[0]][
                                                                        element_neighbor[1]] + (0.5 * self._G[element][
                    element_neighbor[0]] * self._G[element][element_neighbor[1]])
            else:
                self._G[element_neighbor[0]][element_neighbor[1]] = (
                        0.5 * self._G[element][element_neighbor[0]] * self._G[element][element_neighbor[1]])
            self._G[element_neighbor[1]][element_neighbor[0]] = self._G[element_neighbor[0]][element_neighbor[1]]
            self._vid_vertex.pop(element)
            self._G.pop(element)
            self._G[element_neighbor[0]].pop(element)
            self._G[element_neighbor[1]].pop(element)

    def compression(self):
        """
        图压缩阶段
        :return:
        """
        d1 = set()
        d2 = set()
        for node in self._G.keys():
            if len(self._G[node].keys()) == 1:
                d1.add(node)
            if len(self._G[node].keys()) == 2:
                d2.add(node)
        while d1 or d2:
            # 对度为1的节点进行删除
            d1_remove = set()
            d1_add = set()
            for element in d1:
                # 获取度为1的邻居节点
                # 对于非联通的数据集进行异常环绕
                try:
                    element_neighbors = list(self._G[element].keys())
                    element_neighbor = element_neighbors[0]
                    self.merged1(element, element_neighbor)
                    if len(self._G[element_neighbor].keys()) == 1:
                        d1_add.add(element_neighbor)
                        if element_neighbor in d2:
                            d2.remove(element_neighbor)
                    if len(self._G[element_neighbor].keys()) == 2:
                        d2.add(element_neighbor)
                    d1_remove.add(element)
                except IndexError:
                    d1_remove.add(element)
            d1.update(d1_add)
            d1.difference_update(d1_remove)
            # 对度为2的节点进行删除
            d2_add = set()
            d2_remove = set()
            for element in d2:
                # 动态改变
                if element in d2_remove:
                    continue
                element_neighbor = list(self._G[element].keys())
                # 对于非联通图的错误排查
                if len(element_neighbor) == 0:
                    d2_remove.add(element)
                    continue
                # 判断是否为桥节点
                try:
                    bridge = self._G[element_neighbor[0]][element_neighbor[1]]
                    self.merged2(element, element_neighbor)
                    d2_remove.add(element)
                    if len(self._G[element_neighbor[0]].keys()) == 1:
                        d1.add(element_neighbor[0])
                        if element_neighbor[0] in d2:
                            d2_remove.add(element_neighbor[0])
                    if len(self._G[element_neighbor[0]].keys()) == 2:
                        d2_add.add(element_neighbor[0])
                    if len(self._G[element_neighbor[1]].keys()) == 1:
                        d1.add(element_neighbor[1])
                        if element_neighbor[1] in d2:
                            d2_remove.add(element_neighbor[1])
                    if len(self._G[element_neighbor[1]].keys()) == 2:
                        d2_add.add(element_neighbor[1])
                except KeyError:
                    d2_remove.add(element)
            d2.update(d2_add)
            d2.difference_update(d2_remove)
        return self._G

    def cal_local_superiority(self):
        """
        结算节点的局部优势
        :return: lsi字典
        """
        # 计算节点的局部优势
        lsi = {}
        for i in self._G.keys():
            # 邻居节点度的和
            sum_i_j = 0
            for j in self._G[i].keys():
                sum_i_j += len(self._G[j])
            # 计算每个节点的LSI
            try:
                lsi_i = (len(self._G[i]) - (sum_i_j / len(self._G[i]))) / (
                        len(self._G[i]) + (sum_i_j / len(self._G[i])))
                # 只考虑大于0的局部中心点
                if lsi_i >= 0:
                    lsi[i] = lsi_i
            except ZeroDivisionError:
                pass
        return lsi

    def determination(self, lsi):
        """
        初始化社区种群
        :return: css
        """
        # 计算相对距离
        dis = {}
        for center in lsi.keys():
            add = True
            # 一阶邻居
            one = set()
            for i in self._G[center].keys():
                if i in lsi.keys() and lsi[i] > lsi[center]:
                    dis[center] = 0
                    add = False
                one.add(i)
            # 二阶邻居
            two = set()
            for i in one:
                for j in self._G[i].keys():
                    if j in lsi.keys() and lsi[j] > lsi[center]:
                        if add:
                            dis[center] = 0.5
                            add = False
                    two.add(j)
            # 三阶
            for i in two:
                for j in self._G[i].keys():
                    if j in lsi.keys() and lsi[j] > lsi[center]:
                        if add:
                            dis[center] = 0.8
                            add = False
            # 三阶以外邻居
            if add:
                dis[center] = 1
        gama = {k: dis[k] for k, v in lsi.items()}
        gi = list(sorted(gama.values(), reverse=True))
        d1 = np.diff(gi)
        d2 = abs(np.diff(d1))
        # 有多个最大值时取最大的索引
        max_indices = [i for i, x in enumerate(d2) if x == max(d2)]
        kp = max(max_indices)
        gkp = gi[kp]
        # 候选种子节点
        seeds = []
        ban_neighbors = set()
        # 相邻的两个种子节点只能取一个
        for key, value in gama.items():
            if value >= gkp:
                if key not in ban_neighbors:
                    seeds.append(key)
                    ban_neighbors.update(self._G[key].keys())
        return seeds

    def expansion(self, sc):
        """
        种子扩展阶段
        :return: 社区划分
        """
        partitions = []
        for seed in sc:
            partitions.append([seed])
        nodes = [i for i in self._G.keys()]
        unsigned_nodes = set(nodes) - set(sc)
        while unsigned_nodes:
            cv = set()
            for community in partitions:
                cn = set()
                for i in community:
                    a = set(self._G[i].keys())
                    cn.update(a)
                cv.update(cn)
            cv = list(cv)
            random.shuffle(cv)
            """访问顺序的不同，会导致结果的不同"""
            for u in cv:
                if u in unsigned_nodes:
                    u_neighbor = set(self._G[u].keys())
                    inter = 0
                    # 节点可能连接不同的社区
                    for community in partitions:
                        if u_neighbor.intersection(set(community)):
                            inter += 1
                    # 之和一个社区有联系就直接合并
                    if inter == 1:
                        for community in partitions:
                            if u_neighbor.intersection(set(community)):
                                community.append(u)
                                break
                    # 不止和一个社区有联系，选相似度最大的合并
                    else:
                        sim = {}
                        for index, community in enumerate(partitions):
                            if u_neighbor.intersection(set(community)):
                                tot_weight = 0
                                tot_weight_n = 0
                                for v in community:
                                    tot_weight_v1 = 0
                                    if v in u_neighbor:
                                        tot_weight += self._G[u][v]
                                        v_neighbor = set(self._G[v].keys())
                                        v_1 = u_neighbor.intersection(v_neighbor)
                                        v_2 = set()
                                        for v1 in v_1:
                                            v_2.update(set(self._G[v1].keys()))
                                            tot_weight_v2 = 0
                                            for v2 in v_2:
                                                try:
                                                    tot_weight_v2 += self._G[v1][v2]
                                                except KeyError:
                                                    pass
                                            tot_weight_v1 += 1 / tot_weight_v2
                                    tot_weight_n += tot_weight_v1
                                sim[index] = tot_weight + tot_weight_n
                        max_index = max(sim, key=sim.get)
                        partitions[max_index].append(u)
                    unsigned_nodes.remove(u)
        return partitions

    def propagation(self, partitions):
        """
        将图压缩阶段节点的内部所有节点添加至该节点所在的社区
        :return: 最终的社区划分
        """
        for community in partitions:
            community_to_add = []
            for node in community:
                try:
                    inter_nodes = self._vid_vertex[node].propagation()
                    for inter_node in inter_nodes:
                        if inter_node not in community:
                            community_to_add.append(inter_node)
                except KeyError:
                    pass
            community.extend(community_to_add)
        return partitions

    def execute(self):
        self.compression()
        lsi = self.cal_local_superiority()
        sc = self.determination(lsi)
        partitions = self.expansion(sc)
        communities = self.propagation(partitions)
        return communities
