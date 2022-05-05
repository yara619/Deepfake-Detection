# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 04:18:01 2022

@author: mehmoodyar.baig
"""

import tensorflow as tf
devices = tf.config.list_physical_devices('GPU')
print(len(devices)) 


print(tf.test.is_built_with_cuda())

