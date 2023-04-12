'''

28.01.2023 
Класс реализует дерево для реализации расчетов генетического программирования
в данном файле проверяем построение дерева вычисляющего момент в методе моментной оптимизации
    

'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from gp_list_tensor_TF import *
from gp_tree import gp_tree

#создаем с помощью тензоров TF вычисление момента для метода моментной оптимизации
#m1=b*m0-(1-b)*grad
#w1=w0+m1*lr
#здесь grad - вектор градиентов, b - параметр алгоритма, lr - скорость обучения, m0 значение момента на предыдущем этапе
#m1 - значение момента на текущем этапе, w0 - прошлые веса, w1 - обновленные веса

#вычисление происходят в 5 уровней. 
# уровень 5 создаем узел с константой 1 и переменной b
t12211=list_const([1.])
tree12211=gp_tree( level=4, nom_list='1.2.2.2.1', type_ini='manual', limit_level=10, childs=[], cur_list=t12211)
t12212=list_variable(name='b')
tree12212=gp_tree( level=4, nom_list='1.2.2.2.2', type_ini='manual', limit_level=10, childs=[], cur_list=t12212)

#Уровень 4: создаем узел с переменной b, переменной m0 (на следующем уровен будет b*mo),
#создаем поддерево (1-b) и узел grad
t1211=list_variable(name='b')
tree1211=gp_tree( level=3, nom_list='1.2.1.1', type_ini='manual', limit_level=10, childs=[], cur_list=t1211)

t1212=list_variable(name='m0')
tree1212=gp_tree( level=3, nom_list='1.2.1.1', type_ini='manual', limit_level=10, childs=[], cur_list=t1212)

t1221=list_subtract()
tree1221=gp_tree( level=3, nom_list='1.2.2.1', type_ini='manual', limit_level=10, childs=[tree12211, tree12212], cur_list=t1221)

t1222=list_variable(name='grad')
tree1222=gp_tree( level=3, nom_list='1.2.1.1', type_ini='manual', limit_level=10, childs=[], cur_list=t1222)

#на третьем уровне создаем поддеревья: b*m0 и ((1-b)*grad)
t121=list_multiply()
tree121=gp_tree( level=2, nom_list='1.2.1', type_ini='manual', limit_level=10, childs=[tree1211, tree1212], cur_list=t121)

t122=list_multiply()
tree122=gp_tree( level=2, nom_list='1.2.2', type_ini='manual', limit_level=10, childs=[tree1221, tree1222], cur_list=t122)

#на втором уровне создаем переменную lr и вычисляем поддерево (b*m0)-((1-b)*grad)
t11=list_variable(name='lr')
tree11=gp_tree( level=1, nom_list='1.1', type_ini='manual', limit_level=10, childs=[], cur_list=t11)

t12=list_subtract()
tree12=gp_tree( level=1, nom_list='1.2', type_ini='manual', limit_level=10, childs=[tree121, tree122], cur_list=t12)
#на первом уровне вычисляем lr*(b*m0-(1-b)*grad)
t1=list_multiply()
tree1=gp_tree( level=0, nom_list='1.2.2', type_ini='manual', limit_level=10, childs=[tree11, tree12], cur_list=t1)
   
if __name__=='__main__':
    #проверяем дерево
    m0=tf.random.uniform((3,), minval=0, maxval=1)
    grad=tf.random.uniform((3,), minval=0, maxval=1)
    b=tf.constant(0.9)
    lr=tf.constant(0.01)

    params={'m0':m0, 'grad':grad, 'b':b, 'lr':lr}
    m01=m0.numpy()
    print('m0: ')
    print(m01)
    grad1=grad.numpy()
    print('grad: ')
    print(grad1)
    b1=b.numpy()
    print('b: ',b1)
    rez1=tree1.eval(params).numpy()
    print('rez: ')
    print(rez1)
    rez=lr*(b*m0-((1-b)*grad))
    rez=rez.numpy()
    print('контрольный результат: ',rez)



    