
import datetime

from modeling import Model

if __name__ == '__main__':
    # create untrained Model object based on config.py
    from config import config
    model = Model(config=config)

    model.create_usarname_tag_onehot_tables(usernames=["summer", "autumn", "winter", "newseason"], tags=["tag1", "tag2","newtag"])
    # model._dictionary.test()
    average_user =  model._dictionary.get_average_user(2, datetime.datetime.fromtimestamp(0))
    pass

# # mem usage
# import os
# import psutil
# process = psutil.Process(os.getpid())
# print(process.memory_info().rss)
# pass



