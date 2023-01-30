'''

28.01.2023 
Класс реализует дерево для реализации расчетов генетического программирования


'''

import numpy as np
from gp_list import *

class gp_tree:
    def __init__(self, list_T=None, list_F=None, level=0, nom_list='1', type_ini='full',
                 limit_level=2 ) -> None:
        
        #происходит инициализация дерева рекурсивным способом
        if type_ini=='full':
            #инициализируем дерево методом полного роста
            if level<limit_level:
                #если не достигли еще глубины инциализируем функциональным узлом
                i=np.random.randint(len(list_F))
                self.list=list_F[i].copy()
            else:
                i=np.random.randint(len(list_T))
                self.list=list_T[i].copy()
        else:
            raise RuntimeError("Ошибка определения метода инициализации дерева {0}".format(type_ini))
            return False
        
        
        self.level=level
        self.nom_list=str(nom_list)
        self.childs=[]
        self.num_childs=self.list.num_childs
        for i in range(self.num_childs):
            сhild=gp_tree(list_T=list_T, list_F=list_F, level=level+1, nom_list=nom_list+'.'+str(i+1),
                          type_ini=type_ini, limit_level=limit_level)
            self.childs.append(сhild)
        
        return
    
    def print_tree(self):
        #отображение дерева в строку
        if len(self.childs)==0:
            return self.list.get_name()
        else:
            rez=''
            rez+=self.list.get_name()+'('
            for i in range(len(self.childs)):
                rez+=self.childs[i].print_tree()
                
                if i!=len(self.childs)-1:
                    rez+=', '
                
            rez+=')'
        return rez
                
    def eval(self, params):
        #отображение дерева в строку
        if len(self.childs)==0:
            return self.list.eval(params=params)
        else:
            childs=[]
            for i in range(len(self.childs)):
                childs.append(self.childs[i].eval(params=params))
                
                
        return self.list.eval(childs=childs, params=params)
                            
    
if __name__=='__main__':
    list_F=[]
    list_F.append(list_sum(num_childs=1))
    list_F.append(list_sum(num_childs=2))
    list_F.append(list_sum(num_childs=3))
    
    list_F.append(list_sin())
    
    list_T=[]
    list_T.append(list_const(0))
    list_T.append(list_const(1))
    list_T.append(list_const(3.14))
    list_T.append(list_const(2.71))
    list_T.append(list_variable(name='x1'))
    list_T.append(list_variable(name='x2'))
    list_T.append(list_variable(name='x3'))
    list_T.append(list_variable(name='x4'))
    
    tree=gp_tree(list_T=list_T, list_F=list_F, level=0, nom_list='1', type_ini='full',
                 limit_level=4)
    str_tree=tree.print_tree()
    
    params={'x1':1,'x2':2, 'x3':3, 'x4':4 }
    rez=tree.eval(params)
    
    
    
    
    