'''
28.01.2023 
В данном файле будет реализован типовой лист для дерева GP
В листе только реализуется вычисление текущего узла. Механика вычисления 
Всего дерева реализуется в другом файле

структура терминального узла:
    self.name = имя узла
    self.value = значение для константного узла
    self.num_childs=0 т.к. у терминального узла нет потомков
    eval - функция возвращает значение терминального узла. Если константа, то 
           само значение, если переменная, то подставляем значение переменной из param,
           где params - словарь, который содержит ключ=имя переменной, 
           значение=значение переменной. 
    copy - функция создает полную копию узла
    get_name - функция используется при распечатке (вывода в строку) дерева

структура функционального узла:
    self.name - имя узла
    self.num_childs - количество потомков характерное для этой функции
    eval - функция вычисляет значение функционального узла. в зависимости от 
           типа функции из param подставляем параметры params, если это требуется и 
           childs - подставляем значение дочерних узлов
    


'''


# class base_list:
#     def __init__(self, type='T',  ) -> None:
#         pass
import numpy as np
import tensorflow as tf

class list_const:
    def __init__(self, value=0) -> None:
        self.name='const'
        self.value=tf.constant(value)
        self.num_childs=0
    def eval(self, childs=None, params=None):
        #для общности в каждый узел передаем и потомков и параметры
        return self.value
    def copy(self):
        #функция выполняет полную копию узла
        rez=list_const()
        rez.name=self.name
        rez.value=self.value
        rez.num_childs=self.num_childs
        return rez
    def get_name(self):
        return self.name+'_'+str(self.value.numpy())
    
#==============================================================================
class list_variable:
    def __init__(self, name='x') -> None:
        self.name=name
        self.num_childs=0
    def eval(self, childs=None, params=None):
        #для общности в каждый узел передаем и потомков и параметры
        try:
            return params[self.name]
        except:
            
            raise RuntimeError("Ошибка вычисления узла типа list_variable: name={0}, param={1}, ".format(self.name, params))
        return False
    def copy(self):
        #функция выполняет полную копию узла
        rez=list_variable()
        rez.name=self.name
        rez.num_childs=self.num_childs
        return rez
    
    def get_name(self):
        return self.name
        
#==============================================================================
class list_sum:
    def __init__(self) -> None:
        self.name='sum'
        self.num_childs=2
    def eval(self, childs=None, params=None):
        #для общности в каждый узел передаем и потомков и параметры
        try:
            
            return tf.add(childs[0], childs[1])
        except:
            
            raise RuntimeError("Ошибка вычисления узла типа list_sum: name={0}, childs={1}".format(self.name,childs ))
        return False
    def copy(self):
        #функция выполняет полную копию узла
        rez=list_sum()
        rez.name=self.name
        rez.num_childs=self.num_childs
        return rez
    
    def get_name(self):
        return self.name+str(self.num_childs)
#==============================================================================
class list_subtract:
    def __init__(self) -> None:
        self.name='subtract'
        self.num_childs=2
    def eval(self, childs=None, params=None):
        #для общности в каждый узел передаем и потомков и параметры
        try:
            return tf.subtract(childs[0], childs[1])
        except:
            
            raise RuntimeError("Ошибка вычисления узла типа list_subtract: name={0}, childs={1}".format(self.name,childs ))
        return False
    def copy(self):
        #функция выполняет полную копию узла
        rez=list_subtract()
        rez.name=self.name
        rez.num_childs=self.num_childs
        return rez
    
    def get_name(self):
        return self.name+str(self.num_childs)
#==============================================================================
class list_one_minus_x:
    def __init__(self) -> None:
        self.name='list_one_minus_x'
        self.num_childs=2
    def eval(self, childs=None, params=None):
        #для общности в каждый узел передаем и потомков и параметры
        try:
            return tf.subtract(tf.constant(1.), childs[0])
        except:
            
            raise RuntimeError("Ошибка вычисления узла типа list_one_minus_x: name={0}, childs={1}".format(self.name,childs ))
        return False
    def copy(self):
        #функция выполняет полную копию узла
        rez=list_one_minus_x()
        rez.name=self.name
        rez.num_childs=self.num_childs
        return rez
    
    def get_name(self):
        return self.name+str(self.num_childs)
#==============================================================================
class list_multiply:
    def __init__(self ) -> None:
        self.name='multiply'
        self.num_childs=2
    def eval(self, childs=None, params=None):
        #для общности в каждый узел передаем и потомков и параметры
        try:
            return tf.multiply(childs[0], childs[1])
        except:
            
            raise RuntimeError("Ошибка вычисления узла типа list_multiply: name={0}, childs={1}".format(self.name,childs ))
        return False
    def copy(self):
        #функция выполняет полную копию узла
        rez=list_multiply()
        rez.name=self.name
        rez.num_childs=self.num_childs
        return rez
    
    def get_name(self):
        return self.name+str(self.num_childs)    
#==============================================================================
class list_x_multiply_minus_one:
    def __init__(self ) -> None:
        self.name='list_multiply_minus_one'
        self.num_childs=1
    def eval(self, childs=None, params=None):
        #для общности в каждый узел передаем и потомков и параметры
        try:
            return tf.multiply(tf.constant(-1.), childs[0])
        except:
            
            raise RuntimeError("Ошибка вычисления узла типа list_x_multiply_minus_one: name={0}, childs={1}".format(self.name,childs ))
        return False
    def copy(self):
        #функция выполняет полную копию узла
        rez=list_x_multiply_minus_one()
        rez.name=self.name
        rez.num_childs=self.num_childs
        return rez
    
    def get_name(self):
        return self.name+str(self.num_childs)    
#==============================================================================
class list_divide:
    def __init__(self ) -> None:
        self.name='divide'
        self.num_childs=2
    def eval(self, childs=None, params=None):
        #для общности в каждый узел передаем и потомков и параметры
        try:
            return tf.divide(childs[0], childs[1])
        except:
            
            raise RuntimeError("Ошибка вычисления узла типа list_divide: name={0}, childs={1}".format(self.name,childs ))
        return False
    def copy(self):
        #функция выполняет полную копию узла
        rez=list_divide()
        rez.name=self.name
        rez.num_childs=self.num_childs
        return rez
    
    def get_name(self):
        return self.name+str(self.num_childs) 
#==============================================================================
class list_square:
    def __init__(self ) -> None:
        self.name='square'
        self.num_childs=1
    def eval(self, childs=None, params=None):
        #для общности в каждый узел передаем и потомков и параметры
        try:
            return tf.square(childs[0])
        except:
            
            raise RuntimeError("Ошибка вычисления узла типа list_square: name={0}, childs={1}".format(self.name,childs ))
        return False
    def copy(self):
        #функция выполняет полную копию узла
        rez=list_square()
        rez.name=self.name
        rez.num_childs=self.num_childs
        return rez
    
    def get_name(self):
        return self.name+str(self.num_childs) 
#==============================================================================
class list_sqrt:
    def __init__(self ) -> None:
        self.name='sqrt'
        self.num_childs=1
    def eval(self, childs=None, params=None):
        #для общности в каждый узел передаем и потомков и параметры
        try:
            return tf.sqrt(childs[0])
        except:
            
            raise RuntimeError("Ошибка вычисления узла типа list_sqrt: name={0}, childs={1}".format(self.name,childs ))
        return False
    def copy(self):
        #функция выполняет полную копию узла
        rez=list_sqrt()
        rez.name=self.name
        rez.num_childs=self.num_childs
        return rez
    
    def get_name(self):
        return self.name+str(self.num_childs) 
#==============================================================================
class list_slot_F:
    #Слот используется для сохранения значений промежуточных вычислений для того, что бы использовать их 
    #на следующих итерациях. Слот - функциональное звено и используется в паре с терминальным узлом - типа list_variable. 
    #если имя слота name, то имя терминального узал = переменной будет name+'0', а выходом из функционального узла будет 
    #name+'1'. При обсчете в терминальное звено подставляют то значение, которое было выходом из функционального узла
    #на предыдущей итерации. 
    def __init__(self, name ) -> None:
        self.name=name
        self.num_childs=1
    def eval(self, childs=None, params=None):
        #для общности в каждый узел передаем и потомков и параметры
        try:
            params[self.name+'1']=childs[0]
            return params[self.name+'1']
        except:
            
            raise RuntimeError("Ошибка вычисления узла типа list_slot: name={0}, childs={1}".format(self.name,childs ))
        return False
    def copy(self):
        #функция выполняет полную копию узла
        rez=list_slot_F()
        rez.name=self.name
        rez.num_childs=self.num_childs
        return rez
    
    def get_name(self):
        return self.name+str(self.num_childs) 
#==============================================================================
# class list_sin:
#     def __init__(self) -> None:
#         self.name='sin'
#         self.num_childs=1
#     def eval(self, childs=None, params=None):
#         #для общности в каждый узел передаем и потомков и параметры
#         try:
#             return np.sin(childs[0])
#         except:
            
#             raise RuntimeError("Ошибка вычисления узла типа list_sin: name={0}, childs={1}".format(self.name,childs ))
#         return False
#     def copy(self):
#         #функция выполняет полную копию узла
#         rez=list_sin()
#         rez.name=self.name
#         rez.num_childs=self.num_childs
#         return rez
#     def get_name(self):
#         return self.name

if __name__=='__main__':
    t1=tf.random.uniform((3,), minval=0, maxval=1)
    t2=tf.random.uniform((3,), minval=0, maxval=1)
    t3=list_const([2.])
    t4=list_variable(name='grad')
    params={'grad':tf.ones(3,), 's0':0}
    


    print('исходный тензор 1:', t1.numpy())
    print('исходный тензор 2:', t2.numpy())
    print('константный тензор :', t3.get_name())
    print('переменная :', t4.eval(params=params).numpy())


    childs=[ t1, t3.eval()]
    l1=list_sum()
    l2=list_subtract()
    l3=list_multiply()
    l4=list_divide()
    l5=list_square()
    l6=list_sqrt()
    l7=list_slot_F(name='s')
    l8=list_one_minus_x()
    l9=list_x_multiply_minus_one()


    print('результат сложения: ', l1.eval(childs=childs).numpy())
    print('результат вычитание: ', l2.eval(childs=childs).numpy())
    print('результат умножение: ', l3.eval(childs=childs).numpy())
    print('результат деление: ', l4.eval(childs=childs).numpy())
    print('результат возведение в квадрат: ', l5.eval(childs=[t1]).numpy())
    print('результат вычисления корня: ', l6.eval(childs=[t1]).numpy())

    print('вычисление слота: ', l7.eval(childs=[t1], params=params).numpy())
    params['s0']=params['s1']
    print('вычисление слота: ', l7.eval(childs=[t1], params=params).numpy())
   
    print('вычисление слота list_one_minus_x:  ', l8.eval(childs=[t1], params=params).numpy())
    print('вычисление слота list_x_multiply_minus_one: ', l9.eval(childs=[t1], params=params).numpy())
    




































