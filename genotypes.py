from collections import namedtuple   # 该函数为元组的升级版，是一个特殊的数据结构

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')   # 设置搜索空间元组
# 备选操作
PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]
# 用DARTS的表示方法定义NASNet的最终结果架构
NASNet = Genotype(
    # 由于DARTS特殊的设置，只有两个输入，所以以数组顺序，每两个元素作为一个节点的输入。
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 1),],
    normal_concat=[2, 6],
    reduce=[
        ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 0),
    ],
    reduce_concat=[2, 6]
)
# 定义AmoebaNet的Cell结构
AmoebaNet = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0),
            ('dil_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ],
    normal_concat=[2, 6],
    reduce=[
        ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0),
        ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 0),
    ],
    reduce_concat=[2, 6]
)
# DARTS得到的结果
DARTS_V2= Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0),
                            ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('dil_conv_5x5', 3)], normal_concat=range(2, 6),
                    reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0),
                            ('sep_conv_5x5', 2), ('max_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))
DARTS_NTK_1 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))
DARTS_NTK_2 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1),
                               ('sep_conv_3x3', 3), ('sep_conv_3x3', 2), ('dil_conv_5x5', 4), ('dil_conv_3x3', 3)], normal_concat=range(2, 6),
                       reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2),
                               ('max_pool_3x3', 0), ('sep_conv_3x3', 3), ('dil_conv_5x5', 4), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))
DARTS_NTK_3=Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 1),
                             ('dil_conv_3x3', 3), ('dil_conv_3x3', 2), ('dil_conv_5x5', 4), ('dil_conv_3x3', 3)],
                     normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0),
                                                        ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2),
                                                        ('dil_conv_5x5', 4), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
DARTS_synflow_1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1),
                                   ('dil_conv_5x5', 3), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4), ('dil_conv_5x5', 2)],
                           normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1),
                                                              ('max_pool_3x3', 0), ('dil_conv_5x5', 3), ('max_pool_3x3', 0),
                                                              ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

DARTS_synflow_2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2),
                                   ('sep_conv_5x5', 3), ('dil_conv_5x5', 2), ('dil_conv_5x5', 4), ('dil_conv_3x3', 2)],
                           normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0),
                                                              ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2),
                                                              ('dil_conv_5x5', 2), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))
DARTS_VAS = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2),
                              ('dil_conv_5x5', 3), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 3)],
                      normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2),
                                                         ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
                                                         ('max_pool_3x3', 0), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))
DARTS_entmax = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2),
                                ('dil_conv_5x5', 3), ('sep_conv_3x3', 2), ('sep_conv_3x3', 4), ('max_pool_3x3', 0)],
                        normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
                                                           ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 2),
                                                           ('dil_conv_5x5', 3), ('sep_conv_5x5', 4)], reduce_concat=range(2, 6))
DARTS_entmax_2 =  Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0),
                                   ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1)],
                           normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 1),
                                                              ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 2),
                                                              ('skip_connect', 1), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))

DARTS = DARTS_synflow_2
