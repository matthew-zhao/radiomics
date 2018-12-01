import numpy as np

class Welford(object):
    """Knuth implementation of Welford algorithm.
    """

    def __init__(self, x=None, slices=False):
        self._K = np.float64(0.) 
        self.n = np.float64(0.)
        self._Ex = np.float64(0.)
        self._Ex2 = np.float64(0.)
        self.shape = None
        #self._min = None
        #self._max = None
        self._init = False
        self.__call__(x, slices)
         
    def add_data(self, x, slices=False):
        """Add data.
        """
        if x is None:
            return
        
        x = np.array(x)
        if slices:
            self.shape = x.shape[:2]
            #self.n += float(x.shape[2])
            num_slices = x.shape[2]
        else:
            self.shape = x.shape
            num_slices = 1

        if not self._init:
            self._init = True
            if slices:
                self._K = x[...,0]
            else:
                self._K = x

        for slice_num in range(num_slices):
            self.n += 1.0
            x_slice = x[...,slice_num]
            self._Ex += (x_slice - self._K) / self.n
            self._Ex2 += (x_slice - self._K) * (x_slice - self._Ex)
            self._K = self._Ex
    
    def __call__(self, x, slices=False):
        self.add_data(x, slices)
    def mean(self, axis=None):
        """Compute the mean of accumulated data.
           
           Parameters
           ----------
           axis: None or int or tuple of ints, optional
                Axis or axes along which the means are computed. The default is to
                compute the mean of the flattened array.
        """
        if self.n < 1:
            return None

        val = np.array(self._K + self._Ex / np.float64(self.n))
        if axis:
            return val.mean(axis=axis)
        else:
            return val.mean()

    def sum(self, axis=None):
        """Compute the sum of accumulated data.
        """
        return self.mean(axis=axis)*self.n

    def var(self):
        """Compute the variance of accumulated data.
        """
        if self.n <= 1:
            return  np.zeros(self.shape)
            
        val = np.array((self._Ex2 - (self._Ex*self._Ex)/np.float64(self.n)) / np.float64(self.n-1.))

        return val

    def std(self):
        """Compute the standard deviation of accumulated data.
        """
        return np.sqrt(self.var())

#    def __add__(self, val):
#        """Add two Welford objects.
#        """
#

    def __str__(self):
        if self._init:
            return "{} +- {}".format(self.mean(), self.std())
        else:
            return "{}".format(self.shape)

    def __repr__(self):
        return "< Welford: {:} >".format(str(self))

