'''

28.01.2023 
Класс реализует дерево для реализации расчетов генетического программирования
gp_tree:
    def __init__(self, list_T=None, list_F=None, level=0, nom_list='1', type_ini='full',
                 limit_level=2) -> None: - инициализация класса
                list_T - массив терминальных листов
                list_F - массив функциональных листов
                level - уровень узла в дереве
                nom_list - номер листа в дереве. номер устанавливается так: берется номер
                            родительского узла и через "." добавляется порядковый номер текущего 
                            узла как потомка родительского узла. 
                type_ini - способ инициализации: full - все узлы функциональные кроме последнего уровня
                            на котором узлы терминальные, null - пустой узел без роста поддеревьев
                limit_level - предельный уровень роста дерева. 
    def print_tree(self): - печатает дерево в строку
    def copy(self): - создает копию дерева
    def eval(self, params): - вычисляем дерево. params - словарь параметров
    

'''

import numpy as np
from gp_list import *

class gp_tree:
    def __init__(self, list_T=None, list_F=None, level=0, nom_list='1', type_ini='full',
                 limit_level=2) -> None:
        
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
        elif type_ini=='null':
            #инициируем только один  пустой узел без вызова дочерних узлов
            self.level=level 
            self.nom_list=nom_list
            self.list=None
            self.num_childs=0
            self.childs=[]
            return
            
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
    #--------------------------------------------------------------------------
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
    #--------------------------------------------------------------------------            
    def eval(self, params):
        #отображение дерева в строку
        if len(self.childs)==0:
            return self.list.eval(params=params)
        else:
            childs=[]
            for i in range(len(self.childs)):
                childs.append(self.childs[i].eval(params=params))
        return self.list.eval(childs=childs, params=params)
    #--------------------------------------------------------------------------
    def copy(self):
        #copy текущего узла и всего поддерева
        
        tree_list=gp_tree(level=self.level, nom_list=self.nom_list,
                      type_ini='null', limit_level=limit_level)
        tree_list.list=self.list.copy()
            
        if len(self.childs)!=0:
            childs=[]
            for i in range(len(self.childs)):
                childs.append(self.childs[i].copy())
            tree_list.childs=childs
            tree_list.num_childs=len(childs)
        return tree_list
    #--------------------------------------------------------------------------
    def recombination(self, gp_tree, old ):
        #для текущего узла и его подузлов находим узел с номером old
        #заменяем его содержимое содержимым  gp_tree включая все поддерево. 
        #затем обновляем нумерацию
        
        tree_list=gp_tree(level=self.level, nom_list=self.nom_list,
                      type_ini='null', limit_level=limit_level)
        tree_list.list=self.list.copy()
            
        if len(self.childs)!=0:
            childs=[]
            for i in range(len(self.childs)):
                childs.append(self.childs[i].copy())
            tree_list.childs=childs
            tree_list.num_childs=len(childs)
        return tree_list

                            
    
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
                 limit_level=10)
    str_tree=tree.print_tree()
    
    params={'x1':1,'x2':2, 'x3':3, 'x4':4 }
    rez=tree.eval(params)
    
    copy_tree=tree.copy()
    
    str__copy_tree=copy_tree.print_tree()
    rez_copy=copy_tree.eval(params)
    
    
    
    
    