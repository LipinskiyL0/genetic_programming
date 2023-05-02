'''

01.05.2023 
Класс реализует дерево для реализации расчетов генетического программирования
в данном файле проверяем построение дерева вычисляющее RMSProp

'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from gp_list_tensor_TF import *
from gp_tree import gp_tree


tree_111=gp_tree( type_ini='manual', limit_level=10, childs=[], cur_list=list_variable(name='grad'))
tree_11=gp_tree(type_ini='manual', limit_level=10, childs=[tree_111], cur_list=list_x_multiply_minus_one())
tree_1212=gp_tree(  type_ini='manual', limit_level=10, childs=[], cur_list=list_const([1e-7]))
tree_1211112=gp_tree(type_ini='manual', limit_level=10, childs=[], cur_list=list_variable(name='s0'))
tree_1211111=gp_tree(type_ini='manual', limit_level=10, childs=[], cur_list=list_variable(name='b'))
tree_121111=gp_tree( type_ini='manual', limit_level=10, childs=[tree_1211111,tree_1211112 ], cur_list=list_multiply())

tree_12111211=gp_tree(type_ini='manual', limit_level=10, childs=[], cur_list=list_variable(name='b'))
tree_1211121=gp_tree( type_ini='manual', limit_level=10, childs=[tree_12111211 ], cur_list=list_one_minus_x())
tree_12111221=gp_tree(type_ini='manual', limit_level=10, childs=[], cur_list=list_variable(name='grad'))
tree_1211122=gp_tree( type_ini='manual', limit_level=10, childs=[tree_12111221 ], cur_list=list_square())
tree_121112=gp_tree( type_ini='manual', limit_level=10, childs=[tree_1211121,tree_1211122 ], cur_list=list_multiply())

tree_12111=gp_tree(type_ini='manual', limit_level=10, childs=[tree_121111, tree_121112 ], cur_list=list_sum())
tree_1211=gp_tree(type_ini='manual', limit_level=10, childs=[tree_12111 ], cur_list=list_slot_F(name='s'))
tree_121=gp_tree( type_ini='manual', limit_level=10, childs=[tree_1211, tree_1212 ], cur_list=list_sum())
tree_12=gp_tree(type_ini='manual', limit_level=10, childs=[tree_121], cur_list=list_sqrt())

tree_RMSProp=gp_tree(type_ini='manual', limit_level=10, childs=[tree_11, tree_12], cur_list=list_divide())

tree_RMSProp.update_numper(level=0, nom_list='1')

if __name__=='__main__':
    #проверяем дерево
    
    grad=tf.random.uniform((3,), minval=0, maxval=1)
    
    print(grad.numpy())

    params={'s0':tf.constant(0.), 'grad':grad, 'b':tf.constant(0.9) }
    
    rez1=tree_RMSProp.eval(params).numpy()
    print('rez: ')
    print(rez1)

    print(params)
    print('================================================================================')
    print(tree_RMSProp.print_tree())
    print('================================================================================')
    print(tree_RMSProp.get_tree_number())
    print('================================================================================')
    

