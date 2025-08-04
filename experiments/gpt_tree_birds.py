#!/usr/bin/env python3
"""
Full decision tree for 200 bird classes.
Each Node has:
  - question: the question asked at that node
  - answers: a dict mapping answer labels (e.g., "Yes", "Marine", "Auklet", etc.) to child Nodes
  - message: an explanatory message (could be used for logging or hints)
  - possible_classes: the list of candidate classes that remain at that node

The tree is built by first splitting the 200 classes into aquatic vs. non‑aquatic,
then further partitioning (for aquatic: marine vs. freshwater; for marine: large vs. small;
then further splits based on features; for non‑aquatic: hummingbird vs. passerine vs. other).
"""

# === The Node class ===

class Node:
    def __init__(self, question, message, possible_classes):
        """
        Initialize a decision tree node.
        
        Parameters:
            question (str): The question at this node.
            message (str): An extra message or explanation.
            possible_classes (list of str): List of candidate class names.
        """
        self.question = question
        self.message = message
        self.possible_classes = possible_classes if possible_classes is not None else []
        # answers: a dictionary mapping answer labels to child Node objects.
        self.answers = {}
    
    def add_child(self, answer_label, child_node):
        """Attach a child node corresponding to a given answer label."""
        self.answers[answer_label] = child_node

# === Helper functions to decide which group a species belongs to ===

def is_aquatic(species):
    # A species is considered aquatic if its name includes any of these keywords.
    aquatic_keywords = [
        "Albatross", "Auklet", "Cormorant", "Grebe", "Gull", "Tern",
        "Loon", "Pelican", "Merganser", "Kingfisher", "Puffin", "Jaeger",
        "Fulmar", "Frigatebird", "Gadwall", "Mallard", "Guillemot"
    ]
    return any(keyword in species for keyword in aquatic_keywords)

def is_marine(species):
    # Marine species typically have these keywords.
    marine_keywords = [
        "Albatross", "Auklet", "Cormorant", "Grebe", "Gull", "Tern",
        "Loon", "Pelican", "Puffin", "Frigatebird", "Fulmar", "Jaeger", "Guillemot"
    ]
    return any(keyword in species for keyword in marine_keywords)

def is_freshwater(species):
    # Freshwater aquatic species often include these.
    freshwater_keywords = ["Merganser", "Kingfisher", "Gadwall", "Mallard"]
    return any(keyword in species for keyword in freshwater_keywords)

def is_large_marine(species):
    # Large marine birds usually include these keywords.
    large_marine_keywords = ["Albatross", "Pelican", "Loon", "Frigatebird", "Fulmar"]
    return any(keyword in species for keyword in large_marine_keywords)

# === Build the full decision tree ===

def build_full_decision_tree(bird_classes):
    """
    Build the full decision tree for bird classification.
    
    bird_classes: a dict mapping keys "0"..."199" to bird species strings.
    """
    # Build the full list from the dictionary:
    all_species = list(bird_classes.values())
    
    # -- Root node: Is the bird primarily associated with water? --
    root = Node(
        question="Is the bird primarily associated with water?",
        message="Determine if the bird is aquatic (water-associated) or not.",
        possible_classes=all_species
    )
    
    # Partition into aquatic and non-aquatic species:
    aquatic_species = [s for s in all_species if is_aquatic(s)]
    non_aquatic_species = [s for s in all_species if not is_aquatic(s)]
    
    aquatic_node = Node(
        question="Is the bird found in marine or freshwater habitats?",
        message="Determine the type of water habitat.",
        possible_classes=aquatic_species
    )
    non_aquatic_node = Node(
        question="Which non-aquatic group does the bird belong to?",
        message="Determine non-aquatic bird group (e.g., hummingbird, passerine, or other).",
        possible_classes=non_aquatic_species
    )
    
    root.add_child("Yes", aquatic_node)
    root.add_child("No", non_aquatic_node)
    
    # -- AQUATIC BRANCH --
    # Partition aquatic species into marine vs. freshwater.
    marine_species = [s for s in aquatic_species if is_marine(s)]
    # For simplicity, assume that if a species is aquatic but does not match marine, and
    # it contains a freshwater keyword, it is freshwater.
    freshwater_species = [s for s in aquatic_species if is_freshwater(s)]
    
    marine_node = Node(
        question="Is the bird marine?",
        message="Marine aquatic birds are found in open ocean or coastal areas.",
        possible_classes=marine_species
    )
    freshwater_node = Node(
        question="Is the bird freshwater?",
        message="Freshwater birds are found on lakes, rivers, or marshes.",
        possible_classes=freshwater_species
    )
    aquatic_node.add_child("Marine", marine_node)
    aquatic_node.add_child("Freshwater", freshwater_node)
    
    # --- MARINE BRANCH ---
    # Further split marine species by size.
    large_marine_species = [s for s in marine_species if is_large_marine(s)]
    small_marine_species = [s for s in marine_species if not is_large_marine(s)]
    
    large_marine_node = Node(
        question="Is the marine bird large?",
        message="Large marine birds include species like albatrosses and pelicans.",
        possible_classes=large_marine_species
    )
    small_marine_node = Node(
        question="Is the marine bird small?",
        message="Small marine birds include species like auklets, grebes, and gulls.",
        possible_classes=small_marine_species
    )
    marine_node.add_child("Large", large_marine_node)
    marine_node.add_child("Small", small_marine_node)
    
    # --- Large Marine Branch ---
    # (1) Albatross branch: species whose name contains "Albatross".
    albatross_species = [s for s in large_marine_species if "Albatross" in s]
    albatross_node = Node(
        question="Does the bird have a long, slender bill with tube-like nostrils?",
        message="This is a key trait of albatrosses.",
        possible_classes=albatross_species
    )
    for species in albatross_species:
        leaf = Node(
            question="Final classification",
            message=f"Classified as {species}",
            possible_classes=[species]
        )
        albatross_node.add_child(species, leaf)
    
    # (2) Pelican branch: species whose name contains "Pelican".
    non_albatross_large = [s for s in large_marine_species if s not in albatross_species]
    pelican_species = [s for s in non_albatross_large if "Pelican" in s]
    pelican_node = Node(
        question="Does the bird have a very large throat pouch?",
        message="A large throat pouch is typical of pelicans.",
        possible_classes=pelican_species
    )
    for species in pelican_species:
        leaf = Node(
            question="Final classification",
            message=f"Classified as {species}",
            possible_classes=[species]
        )
        pelican_node.add_child(species, leaf)
    
    # (3) Frigatebird branch: species with "Frigatebird" in their name.
    remaining_large = [s for s in non_albatross_large if s not in pelican_species]
    frigate_species = [s for s in remaining_large if "Frigatebird" in s]
    frigate_node = Node(
        question="Does the bird have a deeply forked tail?",
        message="A deeply forked tail is characteristic of frigatebirds.",
        possible_classes=frigate_species
    )
    for species in frigate_species:
        leaf = Node(
            question="Final classification",
            message=f"Classified as {species}",
            possible_classes=[species]
        )
        frigate_node.add_child(species, leaf)
    
    # (4) Fulmar branch: species with "Fulmar" in their name.
    remaining_large = [s for s in remaining_large if s not in frigate_species]
    fulmar_species = [s for s in remaining_large if "Fulmar" in s]
    fulmar_node = Node(
        question="Is the bird compact and robust?",
        message="This trait is used to identify northern fulmars.",
        possible_classes=fulmar_species
    )
    for species in fulmar_species:
        leaf = Node(
            question="Final classification",
            message=f"Classified as {species}",
            possible_classes=[species]
        )
        fulmar_node.add_child(species, leaf)
    
    remaining_large = [s for s in remaining_large if s not in fulmar_species]
    # (5) Loon and Puffin branch:
    loon_species = [s for s in remaining_large if "Loon" in s]
    puffin_species = [s for s in remaining_large if "Puffin" in s]
    
    if loon_species:
        loon_node = Node(
            question="Is the bird streamlined for high-speed diving?",
            message="A streamlined body is typical of loons.",
            possible_classes=loon_species
        )
        for species in loon_species:
            leaf = Node(
                question="Final classification",
                message=f"Classified as {species}",
                possible_classes=[species]
            )
            loon_node.add_child(species, leaf)
    if puffin_species:
        puffin_node = Node(
            question="Is the bird not streamlined for diving?",
            message="This trait helps to identify puffins.",
            possible_classes=puffin_species
        )
        for species in puffin_species:
            leaf = Node(
                question="Final classification",
                message=f"Classified as {species}",
                possible_classes=[species]
            )
            puffin_node.add_child(species, leaf)
    
    # Attach the large marine branches to the large marine node:
    large_marine_node.add_child("Albatross", albatross_node)
    large_marine_node.add_child("Pelican", pelican_node)
    large_marine_node.add_child("Frigatebird", frigate_node)
    if fulmar_species:
        large_marine_node.add_child("Fulmar", fulmar_node)
    if loon_species:
        large_marine_node.add_child("Loon", loon_node)
    if puffin_species:
        large_marine_node.add_child("Puffin", puffin_node)
    
    # --- Small Marine Branch ---
    # Split small marine species into those adapted for diving versus surface feeders.
    diving_species = [s for s in small_marine_species if any(k in s for k in ["Auklet", "Cormorant", "Grebe"])]
    surface_species = [s for s in small_marine_species if any(k in s for k in ["Gull", "Tern", "Jaeger", "Guillemot"])]
    
    diving_node = Node(
        question="Is the bird adapted for diving?",
        message="Diving marine birds dive for prey.",
        possible_classes=diving_species
    )
    surface_node = Node(
        question="Is the bird a surface feeder?",
        message="Surface feeders forage at or near the surface.",
        possible_classes=surface_species
    )
    small_marine_node.add_child("Diving", diving_node)
    small_marine_node.add_child("Surface", surface_node)
    
    # In the diving branch, further split by key traits:
    grebe_species = [s for s in diving_species if "Grebe" in s]
    cormorant_species = [s for s in diving_species if "Cormorant" in s]
    auklet_species = [s for s in diving_species if any(k in s for k in ["Auklet", "Guillemot"])]
    
    grebe_node = Node(
        question="Does the bird have lobed feet?",
        message="Lobed feet are a signature of grebes.",
        possible_classes=grebe_species
    )
    for species in grebe_species:
        leaf = Node(
            question="Final classification",
            message=f"Classified as {species}",
            possible_classes=[species]
        )
        grebe_node.add_child(species, leaf)
    
    cormorant_node = Node(
        question="Does the bird have a hooked bill and elongated body?",
        message="This is typical of cormorants.",
        possible_classes=cormorant_species
    )
    for species in cormorant_species:
        leaf = Node(
            question="Final classification",
            message=f"Classified as {species}",
            possible_classes=[species]
        )
        cormorant_node.add_child(species, leaf)
    
    auklet_node = Node(
        question="Is the bird an auklet or guillemot?",
        message="Identifying auklets and guillemots.",
        possible_classes=auklet_species
    )
    for species in auklet_species:
        leaf = Node(
            question="Final classification",
            message=f"Classified as {species}",
            possible_classes=[species]
        )
        auklet_node.add_child(species, leaf)
    
    diving_node.add_child("Grebe", grebe_node)
    diving_node.add_child("Cormorant", cormorant_node)
    diving_node.add_child("Auklet/Guillemot", auklet_node)
    
    # In the surface branch, split into terns and gulls.
    tern_species = [s for s in surface_species if "Tern" in s]
    gull_species = [s for s in surface_species if "Gull" in s]
    
    tern_node = Node(
        question="Does the bird have a slender body with a forked tail?",
        message="This trait is typical of terns.",
        possible_classes=tern_species
    )
    for species in tern_species:
        leaf = Node(
            question="Final classification",
            message=f"Classified as {species}",
            possible_classes=[species]
        )
        tern_node.add_child(species, leaf)
    
    gull_node = Node(
        question="Is the bird a gull based on coloration and bill shape?",
        message="Gulls are distinguished by their robust bills and plumage.",
        possible_classes=gull_species
    )
    for species in gull_species:
        leaf = Node(
            question="Final classification",
            message=f"Classified as {species}",
            possible_classes=[species]
        )
        gull_node.add_child(species, leaf)
    
    surface_node.add_child("Tern", tern_node)
    surface_node.add_child("Gull", gull_node)
    
    # --- Freshwater Branch (under Aquatic) ---
    # Split freshwater species into Kingfisher versus Waterfowl.
    kingfisher_species = [s for s in freshwater_species if "Kingfisher" in s]
    waterfowl_species = [s for s in freshwater_species if any(k in s for k in ["Mallard", "Merganser", "Gadwall"])]
    
    kingfisher_node = Node(
        question="Does the bird show the typical kingfisher look (stout bill, vivid markings)?",
        message="Kingfishers are colorful and have a distinctive profile.",
        possible_classes=kingfisher_species
    )
    for species in kingfisher_species:
        leaf = Node(
            question="Final classification",
            message=f"Classified as {species}",
            possible_classes=[species]
        )
        kingfisher_node.add_child(species, leaf)
    
    waterfowl_node = Node(
        question="Does the bird have a broad, flat bill with lobed feet?",
        message="This trait is common in waterfowl.",
        possible_classes=waterfowl_species
    )
    # Further split waterfowl into Mallard, Merganser, and Gadwall.
    mallard_species = [s for s in waterfowl_species if "Mallard" in s]
    merganser_species = [s for s in waterfowl_species if "Merganser" in s]
    gadwall_species = [s for s in waterfowl_species if "Gadwall" in s]
    
    mallard_node = Node(
        question="Does the male show striking head patterns?",
        message="A key feature of mallards.",
        possible_classes=mallard_species
    )
    for species in mallard_species:
        leaf = Node(
            question="Final classification",
            message=f"Classified as {species}",
            possible_classes=[species]
        )
        mallard_node.add_child(species, leaf)
    
    merganser_node = Node(
        question="Is the bird streamlined for diving?",
        message="This helps identify mergansers.",
        possible_classes=merganser_species
    )
    for species in merganser_species:
        leaf = Node(
            question="Final classification",
            message=f"Classified as {species}",
            possible_classes=[species]
        )
        merganser_node.add_child(species, leaf)
    
    gadwall_node = Node(
        question="Is the bird a Gadwall?",
        message="Identification of Gadwall.",
        possible_classes=gadwall_species
    )
    for species in gadwall_species:
        leaf = Node(
            question="Final classification",
            message=f"Classified as {species}",
            possible_classes=[species]
        )
        gadwall_node.add_child(species, leaf)
    
    waterfowl_node.add_child("Mallard", mallard_node)
    waterfowl_node.add_child("Merganser", merganser_node)
    waterfowl_node.add_child("Gadwall", gadwall_node)
    
    freshwater_node.add_child("Kingfisher", kingfisher_node)
    freshwater_node.add_child("Waterfowl", waterfowl_node)
    
    # -- NON-AQUATIC BRANCH --
    # Split non-aquatic species into Hummingbirds, Passerines, and Others.
    hummingbird_species = [s for s in non_aquatic_species if any(k in s for k in ["Hummingbird", "Violetear"])]
    remaining_non_aquatic = [s for s in non_aquatic_species if s not in hummingbird_species]
    
    hummingbird_node = Node(
        question="Does the bird have iridescent plumage and rapid wing movement?",
        message="This is typical of hummingbirds.",
        possible_classes=hummingbird_species
    )
    for species in hummingbird_species:
        leaf = Node(
            question="Final classification",
            message=f"Classified as {species}",
            possible_classes=[species]
        )
        hummingbird_node.add_child(species, leaf)
    
    non_aquatic_node.add_child("Hummingbird", hummingbird_node)
    
    # For the remaining non-aquatic species, we split into Passerine and Other.
    passerine_keywords = [
        "Blackbird", "Bunting", "Catbird", "Sparrow", "Towhee", "Oriole",
        "Flycatcher", "Swallow", "Warbler", "Vireo", "Mockingbird", "Wren",
        "Nutcracker", "Junco", "Goldfinch", "Grosbeak"
    ]
    passerine_species = [s for s in remaining_non_aquatic if any(k in s for k in passerine_keywords)]
    other_species = [s for s in remaining_non_aquatic if s not in passerine_species]
    
    passerine_node = Node(
        question="Does the bird have the perching and singing traits of a passerine?",
        message="Passerines (songbirds) form a large group.",
        possible_classes=passerine_species
    )
    for species in passerine_species:
        leaf = Node(
            question="Final classification",
            message=f"Classified as {species}",
            possible_classes=[species]
        )
        passerine_node.add_child(species, leaf)
    
    other_node = Node(
        question="Does the bird have distinctive non-passerine traits?",
        message="This group includes cuckoos, woodpeckers, and others.",
        possible_classes=other_species
    )
    for species in other_species:
        leaf = Node(
            question="Final classification",
            message=f"Classified as {species}",
            possible_classes=[species]
        )
        other_node.add_child(species, leaf)
    
    non_aquatic_node.add_child("Passerine", passerine_node)
    non_aquatic_node.add_child("Other", other_node)
    
    return root

# === Utility: Function to print the tree recursively ===

# def print_tree(node, indent=""):
#     print(indent + "Question: " + node.question)
#     print(indent + "Message: " + node.message)
#     print(indent + "Possible Classes: " + ", ".join(node.possible_classes))
#     if node.answers:
#         for answer_label, child in node.answers.items():
#             print(indent + f"Answer Option: {answer_label}")
#             print_tree(child, indent + "    ")
#     else:
#         print(indent + "Leaf node reached.\n")

def print_tree(node, level=0):
    if node is not None:
        prefix = "  " * level
        if node.question:
            print(f"{prefix}[L{level}] Q: {node.question}")
            for ans in node.answers:
                if ans in node.answers:
                    print(f"{prefix}  -> {ans}:")
                    print_tree(node.answers[ans], level+1)
        else:
            print(f"{prefix}[L{level}] Final: {node.message} ({node.possible_classes})")
# === The full list of 200 bird classes (as provided) ===

bird_classes = {
    "0": "Black_footed_Albatross",
    "1": "Laysan_Albatross",
    "2": "Sooty_Albatross",
    "3": "Groove_billed_Ani",
    "4": "Crested_Auklet",
    "5": "Least_Auklet",
    "6": "Parakeet_Auklet",
    "7": "Rhinoceros_Auklet",
    "8": "Brewer_Blackbird",
    "9": "Red_winged_Blackbird",
    "10": "Rusty_Blackbird",
    "11": "Yellow_headed_Blackbird",
    "12": "Bobolink",
    "13": "Indigo_Bunting",
    "14": "Lazuli_Bunting",
    "15": "Painted_Bunting",
    "16": "Cardinal",
    "17": "Spotted_Catbird",
    "18": "Gray_Catbird",
    "19": "Yellow_breasted_Chat",
    "20": "Eastern_Towhee",
    "21": "Chuck_will_Widow",
    "22": "Brandt_Cormorant",
    "23": "Red_faced_Cormorant",
    "24": "Pelagic_Cormorant",
    "25": "Bronzed_Cowbird",
    "26": "Shiny_Cowbird",
    "27": "Brown_Creeper",
    "28": "American_Crow",
    "29": "Fish_Crow",
    "30": "Black_billed_Cuckoo",
    "31": "Mangrove_Cuckoo",
    "32": "Yellow_billed_Cuckoo",
    "33": "Gray_crowned_Rosy_Finch",
    "34": "Purple_Finch",
    "35": "Northern_Flicker",
    "36": "Acadian_Flycatcher",
    "37": "Great_Crested_Flycatcher",
    "38": "Least_Flycatcher",
    "39": "Olive_sided_Flycatcher",
    "40": "Scissor_tailed_Flycatcher",
    "41": "Vermilion_Flycatcher",
    "42": "Yellow_bellied_Flycatcher",
    "43": "Frigatebird",
    "44": "Northern_Fulmar",
    "45": "Gadwall",
    "46": "American_Goldfinch",
    "47": "European_Goldfinch",
    "48": "Boat_tailed_Grackle",
    "49": "Eared_Grebe",
    "50": "Horned_Grebe",
    "51": "Pied_billed_Grebe",
    "52": "Western_Grebe",
    "53": "Blue_Grosbeak",
    "54": "Evening_Grosbeak",
    "55": "Pine_Grosbeak",
    "56": "Rose_breasted_Grosbeak",
    "57": "Pigeon_Guillemot",
    "58": "California_Gull",
    "59": "Glaucous_winged_Gull",
    "60": "Heermann_Gull",
    "61": "Herring_Gull",
    "62": "Ivory_Gull",
    "63": "Ring_billed_Gull",
    "64": "Slaty_backed_Gull",
    "65": "Western_Gull",
    "66": "Anna_Hummingbird",
    "67": "Ruby_throated_Hummingbird",
    "68": "Rufous_Hummingbird",
    "69": "Green_Violetear",
    "70": "Long_tailed_Jaeger",
    "71": "Pomarine_Jaeger",
    "72": "Blue_Jay",
    "73": "Florida_Jay",
    "74": "Green_Jay",
    "75": "Dark_eyed_Junco",
    "76": "Tropical_Kingbird",
    "77": "Gray_Kingbird",
    "78": "Belted_Kingfisher",
    "79": "Green_Kingfisher",
    "80": "Pied_Kingfisher",
    "81": "Ringed_Kingfisher",
    "82": "White_breasted_Kingfisher",
    "83": "Red_legged_Kittiwake",
    "84": "Horned_Lark",
    "85": "Pacific_Loon",
    "86": "Mallard",
    "87": "Western_Meadowlark",
    "88": "Hooded_Merganser",
    "89": "Red_breasted_Merganser",
    "90": "Mockingbird",
    "91": "Nighthawk",
    "92": "Clark_Nutcracker",
    "93": "White_breasted_Nuthatch",
    "94": "Baltimore_Oriole",
    "95": "Hooded_Oriole",
    "96": "Orchard_Oriole",
    "97": "Scott_Oriole",
    "98": "Ovenbird",
    "99": "Brown_Pelican",
    "100": "White_Pelican",
    "101": "Western_Wood_Pewee",
    "102": "Sayornis",
    "103": "American_Pipit",
    "104": "Whip_poor_Will",
    "105": "Horned_Puffin",
    "106": "Common_Raven",
    "107": "White_necked_Raven",
    "108": "American_Redstart",
    "109": "Geococcyx",
    "110": "Loggerhead_Shrike",
    "111": "Great_Grey_Shrike",
    "112": "Baird_Sparrow",
    "113": "Black_throated_Sparrow",
    "114": "Brewer_Sparrow",
    "115": "Chipping_Sparrow",
    "116": "Clay_colored_Sparrow",
    "117": "House_Sparrow",
    "118": "Field_Sparrow",
    "119": "Fox_Sparrow",
    "120": "Grasshopper_Sparrow",
    "121": "Harris_Sparrow",
    "122": "Henslow_Sparrow",
    "123": "Le_Conte_Sparrow",
    "124": "Lincoln_Sparrow",
    "125": "Nelson_Sharp_tailed_Sparrow",
    "126": "Savannah_Sparrow",
    "127": "Seaside_Sparrow",
    "128": "Song_Sparrow",
    "129": "Tree_Sparrow",
    "130": "Vesper_Sparrow",
    "131": "White_crowned_Sparrow",
    "132": "White_throated_Sparrow",
    "133": "Cape_Glossy_Starling",
    "134": "Bank_Swallow",
    "135": "Barn_Swallow",
    "136": "Cliff_Swallow",
    "137": "Tree_Swallow",
    "138": "Scarlet_Tanager",
    "139": "Summer_Tanager",
    "140": "Artic_Tern",
    "141": "Black_Tern",
    "142": "Caspian_Tern",
    "143": "Common_Tern",
    "144": "Elegant_Tern",
    "145": "Forsters_Tern",
    "146": "Least_Tern",
    "147": "Green_tailed_Towhee",
    "148": "Brown_Thrasher",
    "149": "Sage_Thrasher",
    "150": "Black_capped_Vireo",
    "151": "Blue_headed_Vireo",
    "152": "Philadelphia_Vireo",
    "153": "Red_eyed_Vireo",
    "154": "Warbling_Vireo",
    "155": "White_eyed_Vireo",
    "156": "Yellow_throated_Vireo",
    "157": "Bay_breasted_Warbler",
    "158": "Black_and_white_Warbler",
    "159": "Black_throated_Blue_Warbler",
    "160": "Blue_winged_Warbler",
    "161": "Canada_Warbler",
    "162": "Cape_May_Warbler",
    "163": "Cerulean_Warbler",
    "164": "Chestnut_sided_Warbler",
    "165": "Golden_winged_Warbler",
    "166": "Hooded_Warbler",
    "167": "Kentucky_Warbler",
    "168": "Magnolia_Warbler",
    "169": "Mourning_Warbler",
    "170": "Myrtle_Warbler",
    "171": "Nashville_Warbler",
    "172": "Orange_crowned_Warbler",
    "173": "Palm_Warbler",
    "174": "Pine_Warbler",
    "175": "Prairie_Warbler",
    "176": "Prothonotary_Warbler",
    "177": "Swainson_Warbler",
    "178": "Tennessee_Warbler",
    "179": "Wilson_Warbler",
    "180": "Worm_eating_Warbler",
    "181": "Yellow_Warbler",
    "182": "Northern_Waterthrush",
    "183": "Louisiana_Waterthrush",
    "184": "Bohemian_Waxwing",
    "185": "Cedar_Waxwing",
    "186": "American_Three_toed_Woodpecker",
    "187": "Pileated_Woodpecker",
    "188": "Red_bellied_Woodpecker",
    "189": "Red_cockaded_Woodpecker",
    "190": "Red_headed_Woodpecker",
    "191": "Downy_Woodpecker",
    "192": "Bewick_Wren",
    "193": "Cactus_Wren",
    "194": "Carolina_Wren",
    "195": "House_Wren",
    "196": "Marsh_Wren",
    "197": "Rock_Wren",
    "198": "Winter_Wren",
    "199": "Common_Yellowthroat"
}

# === Main: Build and print the tree ===

if __name__ == '__main__':
    decision_tree_root = build_full_decision_tree(bird_classes)
    print_tree(decision_tree_root)
