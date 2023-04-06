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
from selections import *
from recombination import *

class gp_algorithm:
    def __init__(self, n_ind=10, n_iter=5, selection="tournament", n_tur=5,
                 recombination="standart", p_mut=0.1, list_T=None, list_F=None, 
                  type_ini='full', limit_level=2, params=None, transfer_best=True ) -> None:
        
        #сохраняем стартовые параметры
        self.n_ind=n_ind
        self.n_iter=n_iter
        self.selection=selection
        self.n_tur=n_tur
        self.recombination=recombination
        self.list_T=list_T 
        self.list_F=list_F
        self.type_ini=type_ini
        self.limit_level=limit_level
        self.params=params
        self.transfer_best=transfer_best
        self.p_mut=p_mut
        mas_for_mut=[]
        for l in list_F+list_T:
            mas_for_mut.append([l,l.get_name(), l.num_childs])
        self.mas_for_mut=pd.DataFrame(mas_for_mut, columns=['list', 'list_name', 'num_childs'])
        
        return
    
    def opt(self):
        #создаем популяцию индивидов
        self.parents=[]
        for i in range(self.n_ind):
            self.parents.append(gp_tree(list_T=self.list_T, list_F=self.list_F, level=0, nom_list='1',
                                type_ini=self.type_ini, limit_level=self.limit_level))
            # print(self.parents[i].print_tree())
            # print(self.parents[i].get_tree_number())
            
        #производим оценку индивидов
        self.fit_parents=[]
        for i in range(len(self.parents)):
            self.fit_parents.append(self.fit_function(self.parents[i], self.params))
        self.fit_parents=np.array(self.fit_parents)    
        print("generation {3}, min={0}, mean={1}, mas={2}".format( np.min(self.fit_parents),
          np.mean(self.fit_parents), np.max(self.fit_parents), 0))
        
        for i in range(1,self.n_iter):
            i_parents=selection(self.fit_parents, type_selection=self.selection, n_tur=self.n_tur)
            self.childs=[]
            self.fit_childs=[]
            for i_par in i_parents:
                self.childs.append(recombination(self.parents[i_par[0]],self.parents[i_par[1]],self.recombination))
                self.childs[-1].mutation(self.p_mut, self.mas_for_mut)
                self.fit_childs.append(self.fit_function(self.childs[-1], self.params))
            self.fit_childs=np.array(self.fit_childs)
            
            if self.transfer_best:
                #перенос лучшего в следующее поколение
                i_best=np.argmax(self.fit_parents)
                i_random=np.random.randint(len(self.childs))
                self.childs[i_random]=self.parents[i_best].copy()
                self.fit_childs[i_random]=self.fit_parents[i_best]
            #потомки становятся родителями
            self.parents=self.childs
            self.fit_parents=self.fit_childs
            print("generation {3}, min={0}, mean={1}, mas={2}".format( np.min(self.fit_parents),
              np.mean(self.fit_parents), np.max(self.fit_parents), i))
            
            
            
            
                
            
            
        i_best=np.argmax(self.fit_parents)    
        return {'fit':self.fit_parents[i_best], 'individ':self.parents[i_best].copy()}
    
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
    
    list_F.append(list_prod(num_childs=1))
    list_F.append(list_prod(num_childs=2))
    list_F.append(list_prod(num_childs=3))
    
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
    y=2*(x1**2)+2*x1
    params={'x1':x1, 'y':y}
    gp=gp_algorithm(n_ind=100, n_iter=20, list_T=list_T, list_F=list_F, type_ini='full', limit_level=3, params=params)
    
    rez=gp.opt()
    print('лучшая пригодность: ', rez['fit'])
    print('лучшее решение: ', rez['individ'].print_tree())
    
    # i=np.argmax(gp.fit_parents)
    
    # print(gp.fit_parents[i])
    # print(gp.parents[i].print_tree())
    
    
    
    