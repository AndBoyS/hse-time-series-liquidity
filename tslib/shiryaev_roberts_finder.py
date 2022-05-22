import numpy as np
from scipy.stats import norm

# грубо говоря, критерий Неймана-Пирсона для среднего
def normal_likelihood(value, mean_0, mean_8, std):
    return np.log(norm.pdf(value, mean_0, std) / 
                  norm.pdf(value, mean_8, std))

class CusumFinder():
    
    def __init__(self, alpha, beta, sigma_diff, ceil = 1e6, trsh=2, breaks_max=5,
                slice_length=15):
        
        self.mean_hat = 0
        self.std_hat = 1
        
        self.alpha = alpha
        self.beta = beta
        
        self.metric = 0
        
        # ГИПЕРПАРАМЕТР: дельта для альтернативный гипотезы
        self.sigma_diff = sigma_diff
        # ГИПЕРПАРАМЕТР: верхняя граница для значения критерия
        self.ceil = ceil
        # ГИПЕРПАРАМЕТР: порог для критерия
        self.trsh = trsh
        # ГИПЕРПАРАМЕТРЫ: макс. число разладок для принятия мер
        self.breaks_max = breaks_max
        
        self.states = []
        self.breakpoints = []
        # ГИПЕРПАРАМЕТР: глубина среза значений метрики
        self.slice_length = slice_length
        self.colors=['blue', 'red']
        
    def get_values(self):
        # оцениваем mean и std^2
        try:
            self.mean_hat = self.mean_values_sum / self.mean_weights_sum
            self.var_hat = self.var_values_sum / self.var_weights_sum
        except AttributeError:
            self.mean_hat = 0
            self.var_hat = 1
    
    def update(self, new_value):

        self.get_values()
        
        # Считаем обновлённое значение std и среднее
        self.predicted_diff_value = (new_value - self.mean_hat) ** 2
        self.predicted_diff_mean = self.var_hat
        

        # обновляем значения mean и std^2
        try:
            self.mean_values_sum = (1 - self.alpha) * self.mean_values_sum + new_value
            self.mean_weights_sum = (1 - self.alpha) * self.mean_weights_sum + 1.0
        except AttributeError:
            self.mean_values_sum = new_value
            self.mean_weights_sum = 1.0 
        
        # новое значение std^2
        new_value_var = (new_value - self.mean_hat)**2
        
        try:
            self.var_values_sum = (1 - self.beta) * self.var_values_sum + new_value_var
            self.var_weights_sum = (1 - self.beta) * self.var_weights_sum + 1.0
        except:
            self.var_values_sum = new_value_var
            self.var_weights_sum = 1.0      

    def count_metric(self):
        
        # проверка гипотезы о том, что среднее у разницы между std действительно = 0
        adjusted_value = self.predicted_diff_value - self.predicted_diff_mean
        likelihood = np.exp(self.sigma_diff * (adjusted_value - self.sigma_diff / 2.))
        self.metric = min(self.ceil, (1. + self.metric) * likelihood)
        

#         if self.metric > self.trsh:            
#             self.states.append(1)
#         else:
#             self.states.append(0)
            
#         if (np.array(self.states[-self.slice_length:]) == 1).sum() > self.breaks_max:
#             self.breakpoints.append('red')
#         else:
#             self.breakpoints.append('blue')