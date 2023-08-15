
class TreeNode:
    def __init__(self, id, name, parentId, layer):
        self.id = id
        self.name = name
        self.parentId = parentId
        self.layer = layer
        self.childList = []
        self.questionList = [[] for _ in range(13)]
        
    def addChild(self, childId):
        if childId not in self.childList:
            self.childList.append(childId)
    
    def addQues(self, qid, type):
        if qid not in self.questionList[type]:
            self.questionList[type].append(qid)



