import numpy as np

class goodwin_model:
    def __init__(self, alpha_m: float, beta_m: float, alpha_e: float, beta_e: float, alpha_p: float, beta_p: float, a: float, b: float, c: float, h: float):
        self.alpha_m = alpha_m
        self.beta_m = beta_m
        self.alpha_e = alpha_e
        self.beta_e = beta_e
        self.alpha_p = alpha_p
        self.beta_p = beta_p
        self.a = a
        self.b = b
        self.c = c
        self.h = h

    def g_R(self, p):
        return self.a / (self.b + self.c * p**self.h)

    def model(self, y, t):
        m, e, p = y
        dm_dt = self.alpha_m * self.g_R(p) - self.beta_m * m
        de_dt = self.alpha_e * m - self.beta_e * e
        dp_dt = self.alpha_p * e - self.beta_p * p
        return [dm_dt, de_dt, dp_dt]

class switch_genetico:
    def __init__(self, alpha_m: float, beta_m: float, alpha_p: float, beta_p: float, a: float, b: float, c: float, h: float):
        self.alpha_m = alpha_m
        self.beta_m = beta_m
        self.alpha_p = alpha_p
        self.beta_p = beta_p
        self.a = a
        self.b = b
        self.c = c
        self.h = h

    def g_R(self, p):
        return self.a / (self.b + self.c * p**self.h)

    def model_reducido(self, y, t):
        p1, p2 = y
        m1 = self.alpha_m * self.g_R(p2) / self.beta_m
        m2 = self.alpha_m * self.g_R(p1) / self.beta_m
        dp1_dt = self.alpha_p * m1 - self.beta_p * p1
        dp2_dt = self.alpha_p * m2 - self.beta_p * p2
        return [dp1_dt, dp2_dt]

    def nullclina_p1(self, p2):
        return self.alpha_m * self.a * self.alpha_p / (self.beta_m * self.beta_p * (self.b + self.c * p2**self.h))

    def nullclina_p2(self, p1):
        return self.alpha_m * self.a * self.alpha_p / (self.beta_m * self.beta_p * (self.b + self.c * p1**self.h))

