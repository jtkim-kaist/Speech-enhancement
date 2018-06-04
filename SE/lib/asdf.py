code_list = []
read_flag = False

with open('./trnmodel.py', 'r') as f:
    while True:
        line = f.readline()
        if "def inference(self, inputs):" in line:
            read_flag = True

        if "return fm" in line:

            code_list.append('<br>'+line.replace('\n', '')+'</br>')

            read_flag = False
            break

        if read_flag:
            code_list.append('<br>' + line.replace('\n', '') + '</br>')

code_list = "".join(code_list)

a = 11