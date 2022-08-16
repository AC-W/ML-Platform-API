import pickle

class FileManager():
    def __init__(self):
        self.data = {}
        self.file = None

    def storeData(self,data,file_name):
        self.data = data
        self.file = open(file_name, "wb")
        pickle.dump(self.data, self.file)
        self.file.close()
    
    def loadData(self,file_name):
        self.file = open(file_name, "rb")
        self.data = pickle.load(self.file)
        self.file.close()
        return self.data