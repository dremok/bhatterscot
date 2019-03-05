import re

x = 'hej [ta bort mig] max heter jag (det här också)'
print(re.sub(r'([(\[]).*?([)\]])', "\g<1>\g<2>", x))

print(x)
