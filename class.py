class Adam():
    def __init__(self,n1,n2):
        self.suma = n1+n2
        self.resta = n1-n2

    def __repr__(self, n3):
        return ("La suma és"+str(self.suma)+" i la resta és" + str(self.resta))

    def __str__(self):
        return self.__repr__()
    def func(self):
        return 2*self.suma
