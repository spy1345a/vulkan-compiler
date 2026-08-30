from  compiler  import Parser , Lexer , Flattener , vm
from compiler import Compiler ,Cpu 
# example code
code = "1 + 2 * 3"

# tokonizer of the code
token = Lexer.tokenize(code)
print(token ,"\n")

# prashing building an ats tree 
ats = Parser.parse(token)
print (ats,"\n")

Cpu.run(program="program.toy")
