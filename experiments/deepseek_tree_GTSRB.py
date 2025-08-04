class TreeNode:
    def __init__(self, question=None, answers=None, message=None, possible_classes=None):
        self.question = question
        self.answers = answers or []
        self.message = message
        self.possible_classes = possible_classes or []
        self.children = {}  # Map answers to child nodes

def build_traffic_sign_tree():
    # ==============================================
    # Warnings Subtree (Red and White Triangles)
    # ==============================================
    warnings = TreeNode(
        question="Does the triangle have an exclamation mark?",
        answers=["yes", "no"],
        message="Warning sign classification"
    )
    warnings.children["yes"] = TreeNode(message="Exclamation mark warning", possible_classes=[18])
    
    # Build warning chain for "no" answers
    current = warnings.children["no"] = TreeNode(
        question="Does it depict a left curve?",
        answers=["yes", "no"],
        message="Curve warning"
    )
    current.children["yes"] = TreeNode(message="Left curve warning", possible_classes=[19])
    
    warning_features = [
        ("double curve", 21),
        ("right curve", 20),
        ("rough/bumpy road", 22),
        ("slippery road", 23),
        ("merging/narrow lanes", 24),
        ("construction/road work", 25),
        ("traffic light", 26),
        ("both child and pedestrian", 28),
        ("pedestrian", 27),
        ("bicycle", 29),
        ("ice/snow", 30),
        ("deer", 31)
    ]
    
    for feature, class_id in warning_features:
        new_node = TreeNode(
            question=f"Does it depict {feature}?",
            answers=["yes", "no"],
            message=f"{feature.replace('/', '/ ')} warning"
        )
        new_node.children["yes"] = TreeNode(message=f"{feature} warning", possible_classes=[class_id])
        current.children["no"] = new_node
        current = new_node
    
    # Final fallback
    current.children["no"] = TreeNode(message="Right-of-way at intersection", possible_classes=[11])

    # ==============================================
    # Speed Limit Subtree (Red and White Circles)
    # ==============================================
    speed_limit = TreeNode(
        question="What is the number?",
        answers=["20", "30", "50", "60", "70", "80", "100", "120"],
        message="Speed limit sign"
    )
    speed_classes = [0, 1, 2, 3, 4, 5, 7, 8]
    for ans, cls in zip(speed_limit.answers, speed_classes):
        speed_limit.children[ans] = TreeNode(message=f"{ans} kph limit", possible_classes=[cls])

    # ==============================================
    # Prohibition Subtree (Red and White Circles)
    # ==============================================
    prohibition = TreeNode(
        question="Does it have a diagonal strike bar?",
        answers=["yes", "no"],
        message="Prohibition sign"
    )
    prohibition.children["yes"] = TreeNode(message="End of restriction", possible_classes=[6])
    
    no_passing = prohibition.children["no"] = TreeNode(
        question="What is inside the circle?",
        answers = ["red truck and black car", "red car and black car", "empty circle", "truck", "horizontal white bar"],
        message="Prohibition details"
    )
    no_passing.children.update({
        "red truck and black car": TreeNode(message="No trucks passing", possible_classes=[10]),
        "red car and black car": TreeNode(message="No passing zone sign", possible_classes=[9]),
        "empty circle": TreeNode(message="No vehicles sign", possible_classes=[15]),
        "truck": TreeNode(message="No vehicles over 3.5 t", possible_classes=[16]),
        "horizontal white bar": TreeNode(message="Do not enter road sign", possible_classes=[17])
    })

    # ==============================================
    # Blue Circle Subtree (Mandatory Signs)
    # ==============================================
    blue_circle = TreeNode(
        question="What arrow is shown?",
        answers=["right", "left", "forward", "forward-right", 
                "forward-left", "keep-right", "keep-left", "circular"],
        message="Mandatory direction"
    )
    blue_classes = [33, 34, 35, 36, 37, 38, 39, 40]
    for ans, cls in zip(blue_circle.answers, blue_classes):
        blue_circle.children[ans] = TreeNode(message=f"{ans} arrow", possible_classes=[cls])

    # ==============================================
    # Main Tree Structure
    # ==============================================
    root = TreeNode(
        question="What's the sign's primary shape?",
        answers=["triangle", "circle", "diamond", "inverted-triangle", "octagon"],
        message="Root classification"
    )
    
    # Triangle branch
    root.children["triangle"] = warnings
    
    # Circle branch
    circle = root.children["circle"] = TreeNode(
        question="What's the circle color?",
        answers=["red-white", "blue", "white-gray"],
        message="Circle color classification"
    )
    
    # Red-white circle
    circle.children["red-white"] = TreeNode(
        question="Does it contain numbers?",
        answers=["yes", "no"],
        message="Red-white circle type"
    )
    circle.children["red-white"].children["yes"] = speed_limit
    circle.children["red-white"].children["no"] = prohibition
    
    # Blue circle
    circle.children["blue"] = blue_circle
    
    # White-gray circle
    white_gray = circle.children["white-gray"] = TreeNode(
        question="What restriction ends?",
        answers=["speed limit", "car passing", "truck passing"],
        message="End of restrictions"
    )
    white_gray.children.update({
        "speed limit": TreeNode(message="No speed limit", possible_classes=[32]),
        "car passing": TreeNode(message="End car passing ban", possible_classes=[41]),
        "truck passing": TreeNode(message="End truck passing ban", possible_classes=[42])
    })
    
    # Other shapes
    root.children["diamond"] = TreeNode(message="Priority road", possible_classes=[12])
    root.children["inverted-triangle"] = TreeNode(message="Yield", possible_classes=[13])
    root.children["octagon"] = TreeNode(message="Stop", possible_classes=[14])

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
            print(f"{prefix}[L{level}] Final: {node.message} ({node.possible_classes})")

if __name__ == "__main__":
    print("Full Traffic Sign Classification Tree:")
    tree = build_traffic_sign_tree()
    print_tree(tree)