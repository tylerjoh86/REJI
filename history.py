class History:
    def __init__(self):
        self.context = ""

    def add(self, text):
        self.context += text
    
    def get(self):
        return self.context
    
    def clear_history(self):
        self.context = ""