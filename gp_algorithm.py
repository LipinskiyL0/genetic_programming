# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 20:22:58 2023

@author: Leonid

Генетическое программирование 
"""


import numpy as np
from gp_list import *
from gp_tree import *
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

class gp_algorithm:
    def __init__(self, n_ind=10, n_iter=5, selection="tournament", 
                 recombination="one_point",list_T=None, list_F=None, 
                  type_ini='full', limit_level=2, params=None ) -> None:
        
        #сохраняем стартовые параметры
        self.n_ind=n_ind
        self.n_iter=n_iter
        self.selection=selection
        self.recombination=recombination
        self.list_T=list_T 
        self.list_F=list_F
        self.type_ini=type_ini
        self.limit_level=limit_level
        self.params=params
        #создаем популяцию индивидов
        self.parents=[]
        for i in range(self.n_ind):
            self.parents.append(gp_tree(list_T=self.list_T, list_F=self.list_F, level=0, nom_list='1',
                                type_ini=self.type_ini, limit_level=self.limit_level))
            print(self.parents[i].print_tree())
            print(self.parents[i].get_tree_number())
            
        #производим оценку индивидов
        self.fit_parents=[]
        for i in range(len(self.parents)):
            self.fit_parents.append(self.fit_function(self.parents[i], self.params))
        self.fit_parents=np.array(self.fit_parents)    
        
        return
    
    def fit_function(self, tree, params):
        #вычисление пригодности индивида
        y_pred=tree.eval(params=params)
        if type(y_pred)!=np.ndarray:
            y_pred=np.array([y_pred]*len(params['y']))
        # fit=r2_score(params['y'], y_pred)
        fit=mean_absolute_error(params['y'], y_pred)
        fit=1/(1+fit)
        
        return fit

if __name__=='__main__':
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  
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
    # list_T.append(list_variable(name='x2'))
    # x1=np.random.rand(100)
    # x2=np.random.rand(100)
    x1=np.arange(-1,1,0.01)
    y=x1+x1
    params={'x1':x1, 'y':y}
    gp=gp_algorithm(n_ind=50, list_T=list_T, list_F=list_F, type_ini='full', limit_level=1, params=params)
    print(gp.fit_parents)
    
    i=np.argmax(gp.fit_parents)
    
    print(gp.fit_parents[i])
    print(gp.parents[i].print_tree())
    
    
    
    