class Node:
    def __init__(self, question=None, answers=None, history=None, current_classes=None):
        # questions is like: "Does the object's outline predominantly consist of straight lines?"
        # answers is like: ["Yes", "No"]
        # history is a list of previous questions +answers asked
        # current_classes is a list of class indices that are still in the game like [0, 1, 8, 9]
        # children is dictionary of answer as key and K Nodes as value e.g. {"Yes": [ChileNode(...), ..], "No": [ChildNode(...), ..]}
        self.question: str = question
        self.answers: list[str] = answers if answers is not None else []
        self.history: list[str] = history if history is not None else []
        self.current_classes = current_classes if current_classes is not None else []
        self.children = {}

    def __repr__(self):
        return f"Node(question={self.question}, answers={self.answers}, history={self.history}, current_classes={self.current_classes}, children={self.children})"
class Tree:
    def __init__(self, root=None):
        self.root:list[Node] = root if root is not None else []
