class Message(object):
    def __init__(self, sender, title, content=None):
        self.sender = sender
        self.title = title
        self.content = content