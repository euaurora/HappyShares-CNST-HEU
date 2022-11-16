import numpy as np
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
import copy

class Pip_arrangement:
    def __init__(self,s1,s2,s3,e,r,N,ArrangeType):
        #初始参数
        self.s1 = s1        # 管心水平间距
        self.s2 = s2        # 管心竖直间距
        self.s3 = s3        # 两侧管间距
        self.e = e          # 最外侧管与套桶距离条件
        self.r = r          # 单管外径
        self.N = N          # 总管数
        self.ty = ArrangeType #排管形式，传入"Squar"时为方形，其它参数时为正三角形
        #计算结果
        self.PipeNum = 0
        self.Pippos = 0
        self.R = 0
        self.R_part = 0

    #方形排布：生产初始点
    #默认s1 == s2
    def point_squar(self):
        pos = np.zeros([1,2])
        for i in range(int(2*np.sqrt(self.N))):
            for j in range(int(2*np.sqrt(self.N))):
                pos = np.append(pos,[[self.s1*j,0.5*self.s3+self.s1*(i+1)]],axis=0)
        pos = pos[1:]
        return pos

    # 正三角形排布：生成初始点
    def point_RegularTriangle(self):
        pos = np.zeros([1, 2])
        for i in range(int(2 * np.sqrt(self.N))):
            if i%2 != 0:
                for j in range(int(2 * np.sqrt(self.N))):
                    pos = np.append(pos, [[self.s1 * j, 0.5*self.s3 + self.s2 * (i + 1)]], axis=0)
            else:
                for j in range(int(2 * np.sqrt(self.N))):
                    pos = np.append(pos, [[0.5*self.s1+self.s1 * j, 0.5 * self.s3 + self.s2 * (i + 1)]], axis=0)
        pos = pos[1:]
        return pos

    #自适应R迭代补偿：依据s1的数量级判断R的迭代补偿，设定为s1的小一数量级（3.2->1）
    def minstep(self):
        s = self.s1
        n = 0
        while np.power(10,n)*s < 1:
            n = n+1
        return 1/np.power(10,n)

    #在第一象限中找到合适的点
    def pos_stard(self):
        if self.ty == 'Squar':
            pos = copy.deepcopy(self.point_squar())
        else:
            pos = copy.deepcopy(self.point_RegularTriangle())
        R = self.s1 + self.e
        count_tot = 0
        count_Y = 0
        step = self.minstep()
        while count_tot < (self.N-2*count_Y)/4+count_Y:
            count_Y = 0
            pos_stard = np.zeros([1, 2])
            for p in pos:
                if p[0]**2 + p[1]**2 <= (R-self.e-self.r)**2:
                    pos_stard = np.append(pos_stard,[p],axis=0)
            for pp in pos_stard:
                if pp[0] == 0.:
                    count_Y = count_Y+1
            pos_stard = copy.deepcopy(pos_stard[1:])
            count_tot = len(pos_stard)
            R = R + step #mm
        self.R = R
        return pos_stard

    def arrangement(self):
        pos = copy.deepcopy(self.pos_stard())
        pos_first_with_axis_y = copy.deepcopy(pos)
        pos_second_without_axis_y = np.zeros([1,2])
        for ppp in pos_first_with_axis_y:
            if ppp[0] != 0:
                ppp[0] = -ppp[0]
                pos_second_without_axis_y = np.append(pos_second_without_axis_y,[ppp],axis=0)
        pos_second_without_axis_y = pos_second_without_axis_y[1:]
        pos_up = np.append(pos,pos_second_without_axis_y,axis=0)
        ppos = np.append(pos,pos_second_without_axis_y,axis=0)
        pos_down = pos_up
        for q in pos_down:
            q[1] = -q[1]
        pos = np.append(ppos,pos_down,axis=0)
        L = len(pos)
        self.PipeNum = L
        self.Pippos = pos
        R_part = 0
        for p in pos:
            R_part = R_part + 0.5*np.pi * abs(p[1])
        self.R_part = R_part


    def visualize(self):
        R = self.R
        dr = []
        figure, axes = plt.subplots()
        for xx in self.Pippos:
            dr.append(plt.Circle(xx, self.r, fill=False, lw=0.25))
        dr.append(plt.Circle(np.array([0.,0.]), R, fill=False, lw=0.25))
        axes.set_aspect(1)
        for pic in dr:
            axes.add_artist(pic)
        #自适应坐标轴
        Artist.set(axes, xlabel='X-Axis', ylabel='Y-Axis',
                   xlim=(-1.1*R, 1.1*R), ylim=(-1.1*R, 1.1*R),
                   title='Arrangement of heat transfer tubes')
        plt.savefig('test111.png', dpi=150, bbox_inches='tight')
        plt.title('Arrangement of heat transfer tubes')
        plt.show()
