'''

01.05.2023 
Класс реализует дерево для реализации расчетов генетического программирования
в данном файле проверяем построение дерева вычисляющее AdaGrad

'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from gp_list_tensor_TF import *
from gp_tree import gp_tree



tree_111=gp_tree( type_ini='manual', limit_level=10, childs=[], cur_list=list_variable(name='grad'))
tree_11=gp_tree(type_ini='manual', limit_level=10, childs=[tree_111], cur_list=list_x_multiply_minus_one())
tree_1212=gp_tree(  type_ini='manual', limit_level=10, childs=[], cur_list=list_const([1e-7]))
tree_121112=gp_tree(type_ini='manual', limit_level=10, childs=[], cur_list=list_variable(name='s0'))
tree_1211111=gp_tree(type_ini='manual', limit_level=10, childs=[], cur_list=list_variable(name='grad'))
tree_121111=gp_tree( type_ini='manual', limit_level=10, childs=[tree_1211111 ], cur_list=list_square())
tree_12111=gp_tree(type_ini='manual', limit_level=10, childs=[tree_121111, tree_121112 ], cur_list=list_sum())
tree_1211=gp_tree(type_ini='manual', limit_level=10, childs=[tree_12111 ], cur_list=list_slot_F(name='s'))
tree_121=gp_tree( type_ini='manual', limit_level=10, childs=[tree_1211, tree_1212 ], cur_list=list_sum())
tree_12=gp_tree(type_ini='manual', limit_level=10, childs=[tree_121], cur_list=list_sqrt())
tree_AdaGrad=gp_tree(type_ini='manual', limit_level=10, childs=[tree_11, tree_12], cur_list=list_divide())

tree_AdaGrad.update_numper(level=0, nom_list='1')

if __name__=='__main__':
    #проверяем дерево
    
    grad=tf.random.uniform((3,), minval=0, maxval=1)
    s=tf.constant(0.)
    print(grad.numpy())

    params={'s0':s, 'grad':grad, }
    
    rez1=tree_AdaGrad.eval(params).numpy()
    print('rez: ')
    print(rez1)

    print(params)
    print('================================================================================')
    print(tree_AdaGrad.print_tree())
    print('================================================================================')
    print(tree_AdaGrad.get_tree_number())
    print('================================================================================')

    

