import os
dir = os.path.dirname(__file__)
filename = os.path.join(dir, 'temp/')

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)



# createFolder(filename+'faces/')