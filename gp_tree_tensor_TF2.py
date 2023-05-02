'''

28.01.2023 
Класс реализует дерево для реализации расчетов генетического программирования
в данном файле проверяем построение дерева вычисляющее обычный градиент
    

'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from gp_list_tensor_TF import *
from gp_tree import gp_tree
t11=list_const([-1.])
tree11=gp_tree( level=1, nom_list='1.1', type_ini='manual', limit_level=10, childs=[], cur_list=t11)

t12=list_variable(name='grad')
tree_12=gp_tree( level=1, nom_list='1.2', type_ini='manual', limit_level=10, childs=[], cur_list=t12)

t1=list_multiply()
tree_grad=gp_tree( level=0, nom_list='1', type_ini='manual', limit_level=10, childs=[tree11, tree_12], cur_list=t1)

if __name__=='__main__':
    #проверяем дерево
    m0=tf.random.uniform((3,), minval=0, maxval=1)
    grad=tf.random.uniform((3,), minval=0, maxval=1)
    b=tf.constant(0.9)
    lr=tf.constant(0.01)

    params={'m0':m0, 'grad':grad, 'b':b, 'lr':lr}
    m01=m0.numpy()
    
    grad1=grad.numpy()
    print('grad: ')
    print(grad1)
    
    rez1=tree_grad.eval(params).numpy()
    print('rez: ')
    print(rez1)
    

