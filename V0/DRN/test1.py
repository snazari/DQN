
from config import config_processing
from processing import Processing

config_processing['initial_form'] = True

text = "This is the random text. Some words of this text are repeated, such as words 'words', 'text'. Some of them appear in different forms. It is ment for test purpose"

processor = Processing(**config_processing)
_, words = processor(text)

print(str(words))

