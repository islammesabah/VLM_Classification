class TreeNode:
    def __init__(self, question=None, answers=None, message=None, possible_classes=None):
        self.question = question
        self.answers = answers or []
        self.message = message
        self.possible_classes = possible_classes or []
        self.children = {}  # Map answers to child nodes

def build_cifar_tree():
    # ==============================================
    # Animal Subtree (Living Animals)
    # ==============================================
    
    # Fur/Hair animals subtree
    fur_animals = TreeNode(
        question="Is the animal typically shown with prominent hooves?",
        answers=["yes", "no"],
        message="Hooved vs pawed animals"
    )
    
    # Hooved animals
    hooved = fur_animals.children["yes"] = TreeNode(
        question="Does the animal have antlers visible in the image?",
        answers=["yes", "no"],
        message="Antler identification"
    )
    hooved.children["yes"] = TreeNode(message="deer", possible_classes=[4])
    hooved.children["no"] = TreeNode(message="horse", possible_classes=[7])
    
    # Pawed animals
    pawed = fur_animals.children["no"] = TreeNode(
        question="Does the animal have a noticeably longer snout or muzzle?",
        answers=["yes", "no"],
        message="Snout length identification"
    )
    pawed.children["yes"] = TreeNode(message="dog", possible_classes=[5])
    pawed.children["no"] = TreeNode(message="cat", possible_classes=[3])
    
    # Non-fur animals
    non_fur = TreeNode(
        question="Does the animal have feathers covering its body?",
        answers=["yes", "no"],
        message="Feather identification"
    )
    non_fur.children["yes"] = TreeNode(message="bird", possible_classes=[2])
    non_fur.children["no"] = TreeNode(message="frog", possible_classes=[6])
    
    # Main animal branch
    animals = TreeNode(
        question="Is the animal primarily depicted with fur or hair covering its body?",
        answers=["yes", "no"],
        message="Fur/hair identification"
    )
    animals.children["yes"] = fur_animals
    animals.children["no"] = non_fur
    
    # ==============================================
    # Non-Animal Subtree (Objects/Vehicles)
    # ==============================================
    
    # Water-capable objects
    water_objects = TreeNode(message="ship", possible_classes=[8])
    
    # Non-water objects
    non_water = TreeNode(
        question="Does the object have wings?",
        answers=["yes", "no"],
        message="Wing identification"
    )
    non_water.children["yes"] = TreeNode(message="airplane", possible_classes=[0])
    
    # Ground vehicles
    ground_vehicles = non_water.children["no"] = TreeNode(
        question="Is the vehicle larger in size and typically used for transporting goods?",
        answers=["yes", "no"],
        message="Vehicle size and purpose"
    )
    ground_vehicles.children["yes"] = TreeNode(message="truck", possible_classes=[9])
    ground_vehicles.children["no"] = TreeNode(message="automobile", possible_classes=[1])
    
    # Non-animal branch
    non_animals = TreeNode(
        question="Is the object capable of floating on water?",
        answers=["yes", "no"],
        message="Water capability"
    )
    non_animals.children["yes"] = water_objects
    non_animals.children["no"] = non_water
    
    # ==============================================
    # Main Tree Structure
    # ==============================================
    root = TreeNode(
        question="Is the object in the image a living animal?",
        answers=["yes", "no"],
        message="Root classification - Animal vs Non-animal"
    )
    
    root.children["yes"] = animals
    root.children["no"] = non_animals
    
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
            print(f"{prefix}[L{level}] Leaf Node: {node.message} (ID: {node.possible_classes[0] if node.possible_classes else 'None'})")

if __name__ == "__main__":
    print("CIFAR-10 Classification Tree:")
    tree = build_cifar_tree()
    print_tree(tree)