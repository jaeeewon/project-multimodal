import string

puncs = string.punctuation + " "

remove_puncs = lambda x: x.replace("</s>", "").strip(puncs)