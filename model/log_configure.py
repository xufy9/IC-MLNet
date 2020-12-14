# -*- coding:utf-8 -*-
import logging
import os
import time


logger = logging.getLogger()  # 'updateSecurity'
logger.setLevel(logging.DEBUG)  # logging.INFO


root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(root, time.strftime("%H%M%S", time.localtime()) +'.log')
file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding=None, delay=False) # Relative paths do not work !!!
# console_handler = logging.StreamHandler()
file_handler.setLevel('DEBUG')
# console_handler.setLevel('DEBUG')


formatter = logging.Formatter(fmt='[%(asctime)s.%(msecs)03d] %(levelname)s [%(filename)s %(lineno)d: %(funcName)s]:\n' 
                                  '%(message)s\n' 
                                  '------------------------------------------------------------',
                              datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
# console_handler.setFormatter(formatter)


logger.addHandler(file_handler)
# logger.addHandler(console_handler)