from collections import namedtuple

Genotype = namedtuple('Genotype', 'cell cell_concat')

# PRIMITIVES = [
#     'skip_connect',
#     '3d_conv_3x3',
#     '3d_conv_5x5',
#     '3d_conv_7x7',
# ]

# PRIMITIVES = [
#     'skip_connect',
#     '3d_conv_3x3',
#     '3d_conv_5x5'
# ]

PRIMITIVES = [
    'skip_connect',
    '3d_conv_3x3'
]