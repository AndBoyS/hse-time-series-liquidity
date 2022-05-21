import numpy as np
from scipy.stats import norm

# грубо говоря, критерий Неймана-Пирсона для среднего
def normal_likelihood(value, mean_0, mean_8, std):
    return np.log(norm.pdf(value, mean_0, std) / 
                  norm.pdf(value, mean_8, std))

class MeanExp():
    
    def __init__(self, alpha, beta, mean_diff, trsh=2, breaks_max=5, norm_max=5):
        
        self.mean_hat = 0
        self.std_hat = 1
        
        self.alpha = alpha
        self.beta = beta
        
        self.metric = 0
        self.mean_diff = mean_diff 
        
        self.break_counts = 0
        self.trsh = trsh
        self.breaks_max = breaks_max
        self.norm_max = norm_max
        self.breakpoints = []

        
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
        
        # нормализация и стандартизация
        self.new_value_normalized = (new_value - self.mean_hat) / np.sqrt(self.std_hat)
        
        # обновляем значения mean и std^2
        try:
            self.mean_values_sum = (1 - self.alpha) * self.mean_values_sum + new_value
            self.mean_weights_sum = (1 - self.alpha) * self.mean_weights_sum + 1.0
        except AttributeError:
            self.mean_values_sum = new_value
            self.mean_weights_sum = 1.0 
        
        # новое значение std^2
        new_value_var = (self.new_value_normalized - self.mean_hat)**2
        
        try:
            self.var_values_sum = (1 - self.beta) * self.var_values_sum + new_value_var
            self.var_weights_sum = (1 - self.beta) * self.var_weights_sum + 1.0
        except:
            self.var_values_sum = new_value_var
            self.var_weights_sum = 1.0      

    def count_metric(self):
        # проверка гипотезы о том, что среднее действительно = 0, с учётом того, что std = 1
        zeta_k = normal_likelihood(self.new_value_normalized, self.mean_diff, 0., 1)
        self.metric = max(0, self.metric + zeta_k)
        
        if self.metric > self.trsh:
            self.break_counts += 1

        if self.break_counts > self.breaks_max:
            self.breakpoints.append('red')
        else:
            self.breakpoints.append('blue')