import json
class TreeNode:
    def __init__(self, question=None, answers=None, message=None, possible_classes=None):
        self.question = question
        self.answers = answers or []
        self.message = message
        self.possible_classes = possible_classes or []
        self.children = {}  # Map answers to child nodes

def build_mnist_tree():
    # ==============================================
    # Zero Closed Loops Branch [1, 2, 3, 4, 5, 7]
    # ==============================================
    
    # Has curved segment [2, 3, 5]
    has_curved_segment = TreeNode(
        question="Does the digit contain a straight horizontal line segment?",
        answers=["0", "1"],
        message="Horizontal line identification"
    )
    
    # Has horizontal line [2, 5]
    has_horizontal_line = has_curved_segment.children["1"] = TreeNode(
        question="Does the digit have a curved segment at the top?",
        answers=["0", "1"],
        message="Top curve identification"
    )
    has_horizontal_line.children["1"] = TreeNode(message="2", possible_classes=[2])
    has_horizontal_line.children["0"] = TreeNode(message="5", possible_classes=[5])
    
    # No horizontal line [3]
    has_curved_segment.children["0"] = TreeNode(message="3", possible_classes=[3])
    
    # No curved segment [1, 4, 7]
    no_curved_segment = TreeNode(
        question="Does the digit contain a closed or nearly closed triangular shape?",
        answers=["0", "1"],
        message="Triangular shape identification"
    )
    no_curved_segment.children["1"] = TreeNode(message="4", possible_classes=[4])
    
    # No triangular shape [1, 7]
    no_triangular_shape = no_curved_segment.children["0"] = TreeNode(
        question="Does the digit have a horizontal stroke at the top?",
        answers=["0", "1"],
        message="Top horizontal stroke identification"
    )
    no_triangular_shape.children["1"] = TreeNode(message="7", possible_classes=[7])
    no_triangular_shape.children["0"] = TreeNode(message="1", possible_classes=[1])
    
    # Main zero loops branch
    zero_loops = TreeNode(
        question="Does the digit have a curved segment?",
        answers=["0", "1"],
        message="Curved segment identification"
    )
    zero_loops.children["0"] = has_curved_segment
    zero_loops.children["1"] = no_curved_segment
    
    # ==============================================
    # One Closed Loop Branch [0, 6, 9]
    # ==============================================
    
    one_loop = TreeNode(
        question="Does the digit consist of just a loop, or a loop with an attached stem/tail?",
        answers=["Just a loop", "A loop with a stem/tail"],
        message="Loop structure identification"
    )
    one_loop.children["Just a loop"] = TreeNode(message="0", possible_classes=[0])
    
    # Loop with stem/tail [6, 9]
    loop_with_stem = one_loop.children["A loop with a stem/tail"] = TreeNode(
        question="Where is the closed loop located?",
        answers=["At the bottom", "At the top"],
        message="Loop position identification"
    )
    loop_with_stem.children["At the bottom"] = TreeNode(message="6", possible_classes=[6])
    loop_with_stem.children["At the top"] = TreeNode(message="9", possible_classes=[9])
    
    # ==============================================
    # Two Closed Loops Branch [8]
    # ==============================================
    
    two_loops = TreeNode(message="8", possible_classes=[8])
    
    # ==============================================
    # Main Tree Structure
    # ==============================================
    root = TreeNode(
        question="How many closed loops does the digit have?",
        answers=["0", "1", "2"],
        message="Root classification - Count closed loops"
    )
    
    root.children["0"] = zero_loops
    root.children["1"] = one_loop
    root.children["2"] = two_loops
    
    return root

def print_tree(node, level=0):
    if node is not None:
        prefix = "  " * level
        if node.question:
            print(f"{prefix}[L{level}] Q: {node.question}")
            for ans in node.answers:
                if ans in node.children:
                    print(f"{prefix}  -> {ans}:")
                    print_tree(node.children[ans], level+1)
        else:
            print(f"{prefix}[L{level}] Leaf Node: {node.message} (Class: {node.possible_classes[0] if node.possible_classes else 'None'})")

def save_tree_structure(node):
    # save the printed tree to a text file
    with open("mnist-tree.txt", "w") as f:
        def write_tree(node, level=0):
            if node is not None:
                prefix = "  " * level
                if node.question:
                    f.write(f"{prefix}[L{level}] Q: {node.question}\n")
                    for ans in node.answers:
                        if ans in node.children:
                            f.write(f"{prefix}  -> {ans}:\n")
                            write_tree(node.children[ans], level+1)
                else:
                    f.write(f"{prefix}[L{level}] Leaf Node: {node.message} (Class: {node.possible_classes[0] if node.possible_classes else 'None'})\n")
        write_tree(node)
    
if __name__ == "__main__":
    print("MNIST Digit Classification Tree:")
    tree = build_mnist_tree()
    print_tree(tree)
    save_tree_structure(tree)