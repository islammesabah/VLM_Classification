class TreeNode:
    def __init__(self, question=None, answers=None, message=None, possible_classes=None):
        self.question = question
        self.answers = answers or []
        self.message = message
        self.possible_classes = possible_classes or []
        self.children = {}  # Map answers to child nodes
    
    def get_all_possible_classes(self):
        """Get all possible classes that could be reached from this node."""
        if not self.question:
            # Leaf node - return its possible classes
            return set(self.possible_classes)
        
        # Internal node - collect all possible classes from children
        all_classes = set()
        for child in self.children.values():
            all_classes.update(child.get_all_possible_classes())
        return all_classes

def build_MSCOCO_tree():
    # ==============================================
    # Living Things Subtree
    # ==============================================
    
    # Hoofed Animals
    hoofed_striping = TreeNode(
        question="Does the animal have visible black-and-white striping on its body?",
        answers=["yes", "no"],
        message="Striping identification"
    )
    hoofed_striping.children["yes"] = TreeNode(message="zebra", possible_classes=[24])
    
    # Non-striped hoofed - horn check
    hoofed_horns = hoofed_striping.children["no"] = TreeNode(
        question="Does the animal have horns that curve downward or outward from the head?",
        answers=["yes", "no"],
        message="Horn identification"
    )
    
    # Curved horns - sheep vs cow
    curved_horns = hoofed_horns.children["yes"] = TreeNode(
        question="Does the animal have a white face with woolly fur covering most of its body?",
        answers=["yes", "no"],
        message="Fur texture identification"
    )
    curved_horns.children["yes"] = TreeNode(message="sheep", possible_classes=[20])
    curved_horns.children["no"] = TreeNode(message="cow", possible_classes=[21])
    
    # no curved horns - giraffe vs horse
    no_curved_horns = hoofed_horns.children["no"] = TreeNode(
        question="Does the animal have a long neck?",
        answers=["yes", "no"],
        message="Neck length identification"
    )
    no_curved_horns.children["yes"] = TreeNode(message="giraffe", possible_classes=[25])
    no_curved_horns.children["no"] = TreeNode(message="horse", possible_classes=[19])
    
    # Hoofed main branch
    hoofed_animals = TreeNode(
        question="Does the animal typically have hooves?",
        answers=["yes", "no"],
        message="Hoof identification"
    )
    hoofed_animals.children["yes"] = hoofed_striping
    
    # Not hoofed - human vs animal
    not_hoofed = hoofed_animals.children["no"] = TreeNode(
        question="Is the subject a human or an animal?",
        answers=["Human", "Animal"],
        message="Human vs animal identification"
    )
    not_hoofed.children["Human"] = TreeNode(message="person", possible_classes=[1])
    
    # Animals with fur
    fur_animals = not_hoofed.children["Animal"] = TreeNode(
        question="Is the animal covered with fur?",
        answers=["yes", "no"],
        message="Fur coverage identification"
    )
    
    # Fur animals - rounded ears check
    fur_rounded_ears = fur_animals.children["yes"] = TreeNode(
        question="Does the animal tend to have rounded ears?",
        answers=["yes", "no"],
        message="Ear shape identification"
    )
    fur_rounded_ears.children["yes"] = TreeNode(message="bear", possible_classes=[23])
    
    # Cat vs dog
    cat_dog = fur_rounded_ears.children["no"] = TreeNode(
        question="Does the animal have a long snout?",
        answers=["yes", "no"],
        message="Snout length identification"
    )
    cat_dog.children["yes"] = TreeNode(message="dog", possible_classes=[18])
    cat_dog.children["no"] = TreeNode(message="cat", possible_classes=[17])
    
    # Non-fur animals - elephant vs bird
    non_fur_animals = fur_animals.children["no"] = TreeNode(
        question="Does the animal have a trunk?",
        answers=["yes", "no"],
        message="Trunk identification"
    )
    non_fur_animals.children["yes"] = TreeNode(message="elephant", possible_classes=[22])
    non_fur_animals.children["no"] = TreeNode(message="bird", possible_classes=[16])
    
    # ==============================================
    # Indoor Objects Subtree
    # ==============================================
    
    # Furniture - sitting/resting
    sitting_backrest = TreeNode(
        question="Does the object have a backrest?",
        answers=["yes", "no"],
        message="Backrest identification"
    )
    
    # With backrest - armrests check
    with_backrest = sitting_backrest.children["yes"] = TreeNode(
        question="Does the object have armrests on both sides?",
        answers=["yes", "no"],
        message="Armrest identification"
    )
    
    # With armrests - couch vs chair
    with_armrests = with_backrest.children["yes"] = TreeNode(
        question="Is the object large enough for more than one person to sit on comfortably?",
        answers=["yes", "no"],
        message="Size identification"
    )
    with_armrests.children["yes"] = TreeNode(message="couch", possible_classes=[63])
    with_armrests.children["no"] = TreeNode(message="chair", possible_classes=[62])
    
    # Without armrests - toilet vs bench
    without_armrests = with_backrest.children["no"] = TreeNode(
        question="Does the object feature a water tank as part of its structure?",
        answers=["yes", "no"],
        message="Water tank identification"
    )
    without_armrests.children["yes"] = TreeNode(message="toilet", possible_classes=[70])
    without_armrests.children["no"] = TreeNode(message="bench", possible_classes=[15])
    
    # no backrest
    sitting_backrest.children["no"] = TreeNode(message="bed", possible_classes=[65])
    
    # Furniture for sitting vs surface
    furniture_sitting = TreeNode(
        question="Is the object primarily used for sitting or resting the body?",
        answers=["yes", "no"],
        message="Sitting vs surface identification"
    )
    furniture_sitting.children["yes"] = sitting_backrest
    
    # Surface objects
    surface_objects = furniture_sitting.children["no"] = TreeNode(
        question="Does the object primarily serve as a surface for placing items (such as a table or desk)?",
        answers=["yes", "no"],
        message="Surface functionality identification"
    )
    
    # Tables
    tables = surface_objects.children["yes"] = TreeNode(
        question="Does the object typically have a setup meant for dining, often surrounded by chairs?",
        answers=["yes", "no"],
        message="Table type identification"
    )
    tables.children["yes"] = TreeNode(message="dining table", possible_classes=[67])
    tables.children["no"] = TreeNode(message="desk", possible_classes=[69])
    
    # Non-table surfaces
    non_table_surfaces = surface_objects.children["no"] = TreeNode(
        question="Is the object typically made of transparent or glass materials?",
        answers=["yes", "no"],
        message="Material identification"
    )
    non_table_surfaces.children["yes"] = TreeNode(message="window", possible_classes=[68])
    
    # Window vs door/plant
    window_alt = non_table_surfaces.children["no"] = TreeNode(
        question="Is the object a type of plant, either in a pot or standalone?",
        answers=["yes", "no"],
        message="Plant identification"
    )
    window_alt.children["yes"] = TreeNode(message="potted plant", possible_classes=[64])
    window_alt.children["no"] = TreeNode(message="door", possible_classes=[71])
    
    # Main furniture branch
    furniture = TreeNode(
        question="Is the object primarily used for sitting, resting, or as furniture?",
        answers=["yes", "no"],
        message="Furniture identification"
    )
    furniture.children["yes"] = furniture_sitting
    
    # ==============================================
    # Wearables and Accessories
    # ==============================================
    
    # Head/face worn
    head_face = TreeNode(
        question="Is the object typically worn on the head or face?",
        answers=["yes", "no"],
        message="Head/face worn identification"
    )
    
    # Head covering
    head_covering = head_face.children["yes"] = TreeNode(
        question="Does the object primarily cover the hair or top of the head?",
        answers=["yes", "no"],
        message="Head coverage type"
    )
    head_covering.children["yes"] = TreeNode(message="hat", possible_classes=[26])
    head_covering.children["no"] = TreeNode(message="eye glasses", possible_classes=[30])
    
    # Not head/face - carried vs worn
    not_head_face = head_face.children["no"] = TreeNode(
        question="Is the object typically carried by hand rather than worn on the body?",
        answers=["yes", "no"],
        message="Carried vs worn identification"
    )
    
    # Carried items
    carried = not_head_face.children["yes"] = TreeNode(
        question="Does the object usually have two straps and is worn on the back?",
        answers=["yes", "no"],
        message="Backpack identification"
    )
    carried.children["yes"] = TreeNode(message="backpack", possible_classes=[27])
    
    # Hand-carried bags
    hand_bags = carried.children["no"] = TreeNode(
        question="Does the object commonly have wheels for easier transportation?",
        answers=["yes", "no"],
        message="Wheels identification"
    )
    hand_bags.children["yes"] = TreeNode(message="suitcase", possible_classes=[33])
    hand_bags.children["no"] = TreeNode(message="handbag", possible_classes=[31])
    
    # Worn on body
    worn_body = not_head_face.children["no"] = TreeNode(
        question="Is the object primarily worn on the foot?",
        answers=["yes", "no"],
        message="Foot worn identification"
    )
    worn_body.children["yes"] = TreeNode(message="shoe", possible_classes=[29])
    worn_body.children["no"] = TreeNode(message="tie", possible_classes=[32])
    
    # ==============================================
    # Food and Kitchen Items
    # ==============================================
    
    # Appliances - heating/cooking
    heating_cooking = TreeNode(
        question="Is the object primarily used for heating or cooking food?",
        answers=["yes", "no"],
        message="Heating/cooking identification"
    )
    
    # Heating appliances with door
    heating_door = heating_cooking.children["yes"] = TreeNode(
        question="Does the object have a door with a viewing window on the front?",
        answers=["yes", "no"],
        message="Door with window identification"
    )
    
    # Door with window - microwave vs oven
    door_window = heating_door.children["yes"] = TreeNode(
        question="Does the object usually feature a compact design suitable for countertop placement?",
        answers=["yes", "no"],
        message="Size and placement identification"
    )
    door_window.children["yes"] = TreeNode(message="microwave", possible_classes=[78])
    door_window.children["no"] = TreeNode(message="oven", possible_classes=[79])
    
    # no door window
    heating_door.children["no"] = TreeNode(message="toaster", possible_classes=[80])
    
    # Non-heating appliances
    non_heating = heating_cooking.children["no"] = TreeNode(
        question="Is the object typically used to clean or dispose of water?",
        answers=["yes", "no"],
        message="Water cleaning identification"
    )
    non_heating.children["yes"] = TreeNode(message="sink", possible_classes=[81])
    
    # Other appliances
    other_appliances = non_heating.children["no"] = TreeNode(
        question="Is the object usually a large stationary appliance?",
        answers=["yes", "no"],
        message="Appliance size identification"
    )
    other_appliances.children["yes"] = TreeNode(message="refrigerator", possible_classes=[82])
    other_appliances.children["no"] = TreeNode(message="blender", possible_classes=[83])
    
    # Appliances vs food/utensils
    appliances = TreeNode(
        question="Is the object an appliance (a powered kitchen device)?",
        answers=["yes", "no"],
        message="Appliance identification"
    )
    appliances.children["yes"] = heating_cooking
    
    # Food items
    # Fruits
    fruit_yellow = TreeNode(
        question="Is the primary color of the object yellow?",
        answers=["yes", "no"],
        message="Yellow color identification"
    )
    fruit_yellow.children["yes"] = TreeNode(message="banana", possible_classes=[52])
    
    # Non-yellow fruits
    fruit_color = fruit_yellow.children["no"] = TreeNode(
        question="Is the object primarily red or green in color?",
        answers=["yes", "no"],
        message="Fruit color identification"
    )
    fruit_color.children["yes"] = TreeNode(message="apple", possible_classes=[53])
    fruit_color.children["no"] = TreeNode(message="orange", possible_classes=[55])
    
    # Fruit vs vegetable
    fruit_veg = TreeNode(
        question="Is the object a fruit?",
        answers=["yes", "no"],
        message="Fruit vs vegetable identification"
    )
    fruit_veg.children["yes"] = fruit_yellow
    
    # Vegetables
    veg_color = fruit_veg.children["no"] = TreeNode(
        question="Is the object green in color?",
        answers=["yes", "no"],
        message="Vegetable color identification"
    )
    veg_color.children["yes"] = TreeNode(message="broccoli", possible_classes=[56])
    veg_color.children["no"] = TreeNode(message="carrot", possible_classes=[57])
    
    # Whole vs prepared food
    whole_food = TreeNode(
        question="Is the food primarily a whole fruit or vegetable?",
        answers=["yes", "no"],
        message="Food processing identification"
    )
    whole_food.children["yes"] = fruit_veg
    
    # Prepared food with bread
    bread_meat = TreeNode(
        question="Does the food contain visible meat between bread?",
        answers=["yes", "no"],
        message="Meat between bread identification"
    )
    bread_meat.children["yes"] = TreeNode(message="hot dog", possible_classes=[58])
    
    # Non-meat bread items
    bread_shape = bread_meat.children["no"] = TreeNode(
        question="Is the food item typically ring-shaped with a central hole?",
        answers=["yes", "no"],
        message="Ring shape identification"
    )
    bread_shape.children["yes"] = TreeNode(message="donut", possible_classes=[60])
    bread_shape.children["no"] = TreeNode(message="sandwich", possible_classes=[54])
    
    # Prepared food
    prepared_bread = TreeNode(
        question="Does the food typically contain a visible bread or dough component?",
        answers=["yes", "no"],
        message="Bread component identification"
    )
    prepared_bread.children["yes"] = bread_meat
    
    # Non-bread prepared
    non_bread_prepared = prepared_bread.children["no"] = TreeNode(
        question="Is the food typically round and covered with toppings such as cheese and tomato sauce?",
        answers=["yes", "no"],
        message="Pizza identification"
    )
    non_bread_prepared.children["yes"] = TreeNode(message="pizza", possible_classes=[59])
    non_bread_prepared.children["no"] = TreeNode(message="cake", possible_classes=[61])
    
    whole_food.children["no"] = prepared_bread
    
    # Food vs utensils
    food_item = TreeNode(
        question="Is the object itself an item of food?",
        answers=["yes", "no"],
        message="Food item identification"
    )
    food_item.children["yes"] = whole_food
    
    # Drinking vessels
    drinking_handle = TreeNode(
        question="Does the object have a handle?",
        answers=["yes", "no"],
        message="Handle identification"
    )
    drinking_handle.children["yes"] = TreeNode(message="cup", possible_classes=[47])
    
    # no handle drinking vessels
    drinking_stem = drinking_handle.children["no"] = TreeNode(
        question="Does the object have a long, stem-like structure supporting a bowl?",
        answers=["yes", "no"],
        message="Stem identification"
    )
    drinking_stem.children["yes"] = TreeNode(message="wine glass", possible_classes=[46])
    drinking_stem.children["no"] = TreeNode(message="bottle", possible_classes=[44])
    
    # Utensils/containers
    drinking_vessels = TreeNode(
        question="Is the object primarily used for drinking liquids?",
        answers=["yes", "no"],
        message="Drinking vessel identification"
    )
    drinking_vessels.children["yes"] = drinking_handle
    
    # Food containers
    container_shape = TreeNode(
        question="Is the container typically shallow and flat?",
        answers=["yes", "no"],
        message="Container depth identification"
    )
    container_shape.children["yes"] = TreeNode(message="plate", possible_classes=[45])
    container_shape.children["no"] = TreeNode(message="bowl", possible_classes=[51])
    
    # Container vs utensil
    container_utensil = drinking_vessels.children["no"] = TreeNode(
        question="Is the object used to contain or hold food?",
        answers=["yes", "no"],
        message="Container identification"
    )
    container_utensil.children["yes"] = container_shape
    
    # Eating utensils
    utensil_prongs = TreeNode(
        question="Does the object have prongs or tines at one end?",
        answers=["yes", "no"],
        message="Prongs identification"
    )
    utensil_prongs.children["yes"] = TreeNode(message="fork", possible_classes=[48])
    
    # Knife vs spoon
    utensil_edge = utensil_prongs.children["no"] = TreeNode(
        question="Does the object have a flat, sharp edge intended for cutting?",
        answers=["yes", "no"],
        message="Cutting edge identification"
    )
    utensil_edge.children["yes"] = TreeNode(message="knife", possible_classes=[49])
    utensil_edge.children["no"] = TreeNode(message="spoon", possible_classes=[50])
    
    container_utensil.children["no"] = utensil_prongs
    
    food_item.children["no"] = drinking_vessels
    
    appliances.children["no"] = food_item
    
    # ==============================================
    # Electronics and Other Indoor Items
    # ==============================================
    
    # Glass/reflective objects
    mirror_reflection = TreeNode(
        question="Does the object display a reflective surface used to see one's own image?",
        answers=["yes", "no"],
        message="Mirror identification"
    )
    mirror_reflection.children["yes"] = TreeNode(message="mirror", possible_classes=[66])
    
    # Clock vs vase
    clock_face = mirror_reflection.children["no"] = TreeNode(
        question="Does the object typically have a face with numbers or hands?",
        answers=["yes", "no"],
        message="Clock face identification"
    )
    clock_face.children["yes"] = TreeNode(message="clock", possible_classes=[85])
    clock_face.children["no"] = TreeNode(message="vase", possible_classes=[86])
    
    # Glass objects
    glass_objects = TreeNode(
        question="Is the object primarily made of glass or reflective surfaces?",
        answers=["yes", "no"],
        message="Glass/reflective identification"
    )
    glass_objects.children["yes"] = mirror_reflection
    
    # Electronic input devices
    input_keyboard = TreeNode(
        question="Does the object primarily consist of a flat rectangular surface with multiple buttons or keys arranged in rows?",
        answers=["yes", "no"],
        message="Keyboard identification"
    )
    input_keyboard.children["yes"] = TreeNode(message="keyboard", possible_classes=[76])
    
    # Mouse vs remote
    input_mouse = input_keyboard.children["no"] = TreeNode(
        question="Does the object have a prominent scroll wheel or curved shape designed for handheld use?",
        answers=["yes", "no"],
        message="Mouse identification"
    )
    input_mouse.children["yes"] = TreeNode(message="mouse", possible_classes=[74])
    input_mouse.children["no"] = TreeNode(message="remote", possible_classes=[75])
    
    # Input devices
    input_devices = TreeNode(
        question="Is the object used primarily for input (controlling or interacting with other electronic devices)?",
        answers=["yes", "no"],
        message="Input device identification"
    )
    input_devices.children["yes"] = input_keyboard
    
    # Handheld devices
    handheld_screen = TreeNode(
        question="Does the object have a screen or display surface?",
        answers=["yes", "no"],
        message="Screen identification"
    )
    handheld_screen.children["yes"] = TreeNode(message="cell phone", possible_classes=[77])
    handheld_screen.children["no"] = TreeNode(message="hair drier", possible_classes=[89])
    
    # Non-input electronic devices
    non_input_handheld = input_devices.children["no"] = TreeNode(
        question="Is the object mainly designed for handheld use?",
        answers=["yes", "no"],
        message="Handheld identification"
    )
    non_input_handheld.children["yes"] = handheld_screen
    
    # Stationary displays
    stationary_display = non_input_handheld.children["no"] = TreeNode(
        question="Does the object have a built-in display screen that is used for output only (not interactive)?",
        answers=["yes", "no"],
        message="Display type identification"
    )
    stationary_display.children["yes"] = TreeNode(message="tv", possible_classes=[72])
    stationary_display.children["no"] = TreeNode(message="laptop", possible_classes=[73])
    
    # Electronics
    electronics = TreeNode(
        question="Is the object an electronic device or related accessory?",
        answers=["yes", "no"],
        message="Electronic identification"
    )
    electronics.children["yes"] = input_devices
    
    # Non-electronic items
    soft_toy = TreeNode(
        question="Is the object a soft toy or plush item?",
        answers=["yes", "no"],
        message="Soft toy identification"
    )
    soft_toy.children["yes"] = TreeNode(message="teddy bear", possible_classes=[88])
    
    # Paper items
    paper_item = soft_toy.children["no"] = TreeNode(
        question="Is the object primarily made of paper?",
        answers=["yes", "no"],
        message="Paper material identification"
    )
    paper_item.children["yes"] = TreeNode(message="book", possible_classes=[84])
    
    # Scissors vs toothbrush
    scissors_blades = paper_item.children["no"] = TreeNode(
        question="Does the object have blades used for cutting?",
        answers=["yes", "no"],
        message="Blades identification"
    )
    scissors_blades.children["yes"] = TreeNode(message="scissors", possible_classes=[87])
    scissors_blades.children["no"] = TreeNode(message="toothbrush", possible_classes=[90])
    
    electronics.children["no"] = soft_toy
    
    glass_objects.children["no"] = electronics
    
    # ==============================================
    # Vehicles
    # ==============================================
    
    # Two-wheeled vehicles
    two_wheels_pedals = TreeNode(
        question="Does the vehicle have pedals for human propulsion?",
        answers=["yes", "no"],
        message="Pedals identification"
    )
    two_wheels_pedals.children["yes"] = TreeNode(message="bicycle", possible_classes=[2])
    two_wheels_pedals.children["no"] = TreeNode(message="motorcycle", possible_classes=[4])
    
    # Two wheels
    two_wheels = TreeNode(
        question="Does the vehicle have two wheels?",
        answers=["yes", "no"],
        message="Two wheels identification"
    )
    two_wheels.children["yes"] = two_wheels_pedals
    
    # More than two wheels
    water_vehicle = two_wheels.children["no"] = TreeNode(
        question="Does the vehicle primarily operate on water?",
        answers=["yes", "no"],
        message="Water operation identification"
    )
    water_vehicle.children["yes"] = TreeNode(message="boat", possible_classes=[9])
    
    # Non-water vehicles
    vehicle_wings = water_vehicle.children["no"] = TreeNode(
        question="Does the vehicle have wings?",
        answers=["yes", "no"],
        message="Wings identification"
    )
    vehicle_wings.children["yes"] = TreeNode(message="airplane", possible_classes=[5])
    
    # Ground vehicles - passenger capacity
    passenger_vehicle = vehicle_wings.children["no"] = TreeNode(
        question="Is the vehicle designed primarily for transporting many passengers?",
        answers=["yes", "no"],
        message="Passenger capacity identification"
    )
    
    # Many passengers
    tracks_vehicle = passenger_vehicle.children["yes"] = TreeNode(
        question="Does the vehicle travel on tracks?",
        answers=["yes", "no"],
        message="Tracks identification"
    )
    tracks_vehicle.children["yes"] = TreeNode(message="train", possible_classes=[7])
    tracks_vehicle.children["no"] = TreeNode(message="bus", possible_classes=[6])
    
    # Few passengers
    personal_vehicle = passenger_vehicle.children["no"] = TreeNode(
        question="Is the vehicle smaller in size and typically used for personal transport?",
        answers=["yes", "no"],
        message="Personal transport identification"
    )
    personal_vehicle.children["yes"] = TreeNode(message="car", possible_classes=[3])
    personal_vehicle.children["no"] = TreeNode(message="truck", possible_classes=[8])
    
    # ==============================================
    # Outdoor/Street Objects
    # ==============================================
    
    # Street signs
    sign_octagon = TreeNode(
        question="Is the sign octagonal in shape?",
        answers=["yes", "no"],
        message="Octagonal shape identification"
    )
    sign_octagon.children["yes"] = TreeNode(message="stop sign", possible_classes=[13])
    sign_octagon.children["no"] = TreeNode(message="street sign", possible_classes=[12])
    
    # Signs
    signs = TreeNode(
        question="Is the object primarily a sign displaying text or symbols?",
        answers=["yes", "no"],
        message="Sign identification"
    )
    signs.children["yes"] = sign_octagon
    
    # Non-sign street objects
    street_lights = signs.children["no"] = TreeNode(
        question="Does the object have visible lights or bulbs on its structure?",
        answers=["yes", "no"],
        message="Lights identification"
    )
    street_lights.children["yes"] = TreeNode(message="traffic light", possible_classes=[10])
    
    # Parking meter vs fire hydrant
    payment_slot = street_lights.children["no"] = TreeNode(
        question="Does the object typically have a coin or digital payment slot?",
        answers=["yes", "no"],
        message="Payment slot identification"
    )
    payment_slot.children["yes"] = TreeNode(message="parking meter", possible_classes=[14])
    payment_slot.children["no"] = TreeNode(message="fire hydrant", possible_classes=[11])
    
    # Street attached objects
    street_attached = TreeNode(
        question="Is the object attached to or installed on a street or roadside?",
        answers=["yes", "no"],
        message="Street attachment identification"
    )
    street_attached.children["yes"] = signs
    
    # ==============================================
    # Sports and Recreation Equipment
    # ==============================================
    
    # Snow equipment
    snow_skis = TreeNode(
        question="Does the object consist of two long narrow boards rather than a single wide board?",
        answers=["yes", "no"],
        message="Skis vs snowboard identification"
    )
    snow_skis.children["yes"] = TreeNode(message="skis", possible_classes=[35])
    snow_skis.children["no"] = TreeNode(message="snowboard", possible_classes=[36])
    
    # Boards - snow vs water/ground
    board_snow = TreeNode(
        question="Is the object typically used on snow?",
        answers=["yes", "no"],
        message="Snow use identification"
    )
    board_snow.children["yes"] = snow_skis
    
    # Non-snow boards
    board_water = board_snow.children["no"] = TreeNode(
        question="Is the object primarily used on water?",
        answers=["yes", "no"],
        message="Water use identification"
    )
    board_water.children["yes"] = TreeNode(message="surfboard", possible_classes=[42])
    board_water.children["no"] = TreeNode(message="skateboard", possible_classes=[41])
    
    # Large flat objects
    large_flat = TreeNode(
        question="Is the primary object large and flat (used for riding or standing on)?",
        answers=["yes", "no"],
        message="Large flat object identification"
    )
    large_flat.children["yes"] = board_snow
    
    # Flying objects
    flying_thrown = TreeNode(
        question="Is the object typically thrown by hand during play?",
        answers=["yes", "no"],
        message="Hand-thrown identification"
    )
    flying_thrown.children["yes"] = TreeNode(message="frisbee", possible_classes=[34])
    
    # Umbrella vs kite
    flying_fabric = flying_thrown.children["no"] = TreeNode(
        question="Does the object have a fabric or cloth surface stretched over its frame?",
        answers=["yes", "no"],
        message="Fabric purpose identification"
    )
    flying_fabric.children["yes"] = TreeNode(message="umbrella", possible_classes=[28])
    flying_fabric.children["no"] = TreeNode(message="kite", possible_classes=[38])
    
    # Air objects
    air_objects = large_flat.children["no"] = TreeNode(
        question="Is the object primarily designed to fly or be used in the air?",
        answers=["yes", "no"],
        message="Flying object identification"
    )
    air_objects.children["yes"] = flying_thrown
    
    # Non-flying sports objects
    sports_ball = air_objects.children["no"] = TreeNode(
        question="Is the object spherical or round in shape?",
        answers=["yes", "no"],
        message="Spherical shape identification"
    )
    sports_ball.children["yes"] = TreeNode(message="sports ball", possible_classes=[37])
    
    # Hand-held sports equipment
    handheld_sports = sports_ball.children["no"] = TreeNode(
        question="Is the object held in the hand during use?",
        answers=["yes", "no"],
        message="Hand-held identification"
    )
    
    # Racket vs bat
    handheld_net = handheld_sports.children["yes"] = TreeNode(
        question="Does the object have a netted or strung area as part of its design?",
        answers=["yes", "no"],
        message="Netted area identification"
    )
    handheld_net.children["yes"] = TreeNode(message="tennis racket", possible_classes=[43])
    handheld_net.children["no"] = TreeNode(message="baseball bat", possible_classes=[39])
    
    # Not hand-held
    handheld_sports.children["no"] = TreeNode(message="baseball glove", possible_classes=[40])
    
    street_attached.children["no"] = large_flat
    
    # ==============================================
    # Main Indoor Branch Assembly
    # ==============================================
    
    # Wearables branch
    wearables = TreeNode(
        question="Is the object primarily worn or carried by a person (clothing, accessories, or bags)?",
        answers=["yes", "no"],
        message="Wearable identification"
    )
    wearables.children["yes"] = head_face
    
    # Food-related branch
    food_related = wearables.children["no"] = TreeNode(
        question="Is the object typically associated with food preparation or consumption?",
        answers=["yes", "no"],
        message="Food-related identification"
    )
    food_related.children["yes"] = appliances
    food_related.children["no"] = glass_objects
    
    furniture.children["no"] = wearables
    
    # ==============================================
    # Indoor vs Outdoor Main Branch
    # ==============================================
    
    indoor = TreeNode(
        question="Is the object typically found indoors (in a home or building)?",
        answers=["yes", "no"],
        message="Indoor identification"
    )
    indoor.children["yes"] = furniture
    
    # Outdoor branch - vehicles vs other
    vehicles = TreeNode(
        question="Is the object primarily a vehicle used for transportation?",
        answers=["yes", "no"],
        message="Vehicle identification"
    )
    vehicles.children["yes"] = two_wheels
    vehicles.children["no"] = street_attached
    
    indoor.children["no"] = vehicles
    
    # ==============================================
    # Root Node
    # ==============================================
    
    root = TreeNode(
        question="Is the object a living thing (animal or person)?",
        answers=["yes", "no"],
        message="Root classification - Living vs Non-living"
    )
    
    root.children["yes"] = hoofed_animals
    root.children["no"] = indoor
    
    return root

def validate_tree_answers(node):
    """Recursively validate that answers match children keys."""
    if node is None:
        return True

    # Check if all answers are in children keys
    if not all(answer in node.children for answer in node.answers):
        print(f"Validation Error: Node with question '{node.question}' has mismatched answers and children keys.")
        return False

    # Recursively validate children
    for child in node.children.values():
        if not validate_tree_answers(child):
            return False

    return True

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
    print("MSCOCO Classification Tree:")
    tree = build_MSCOCO_tree()
    print_tree(tree)
    print("\nValidating tree structure...")
    if validate_tree_answers(tree):
        print("Tree structure is valid: All answers match children keys.")
    else:
        print("Tree structure validation failed.")